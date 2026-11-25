"""长上下文继续预训练（任务清单 T3.3；Doc 4 §6、Doc 5 §7-9）。

Context Length Curriculum：8K → 32K → 128K（Phase A–C，主干全量 attention）。
512K/1M（Phase D/E）依赖 M4 稀疏路径，不在本脚本（见任务清单 v1.1 修正）。

L = L_LM + λ·L_retrieval-aware：needle/gold token 的 CE 加权（weighted_ce）。
阶段切换时重算 RoPE（apply_rope_stage）。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import hashlib
import argparse
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.minimind_x import MiniMindXConfig
from dataset.long_context_dataset import LongContextPretrainDataset, collate_long_context
from model.minimind_x.losses import lm_loss_per_exit
from trainer.trainer_utils import (get_lr, Logger, is_main_process, lm_checkpoint,
                                   init_distributed_mode, setup_seed, SkipBatchSampler)
from trainer.x_utils import (init_model_x, save_x_checkpoint, weighted_ce, apply_rope_stage,
                             set_gradient_checkpointing)

warnings.filterwarnings('ignore')


def train_stage(stage_idx, stage_cfg, epoch, iters, wandb=None):
    """单个 curriculum 阶段的训练循环。stage_cfg: {"max_len", "rope_theta", "steps"}。"""
    Logger(f'===== Stage {stage_idx}: max_len={stage_cfg["max_len"]}, '
           f'rope_theta={stage_cfg.get("rope_theta")}, steps={stage_cfg["steps"]} =====')
    apply_rope_stage(model, max_position_embeddings=stage_cfg["max_len"],
                     rope_theta=stage_cfg.get("rope_theta"))
    train_ds = LongContextPretrainDataset(args.data_path, tokenizer,
                                          max_length=stage_cfg["max_len"],
                                          needle_ratio=args.needle_ratio,
                                          needle_weight=args.needle_weight,
                                          seed=42 + stage_idx,
                                          min_chunks=dist.get_world_size() if dist.is_initialized() else 1)
    sampler = DistributedSampler(train_ds, drop_last=True, seed=42 + stage_idx) if dist.is_initialized() else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_long_context)
    total_steps = stage_cfg["steps"]
    start_time = time.time()
    lengths, fingerprints = [], set()
    for step in range(1, total_steps + 1):
        if (step - 1) % len(loader) == 0:
            train_ds.epoch = (step - 1) // len(loader)
            if sampler is not None:
                sampler.set_epoch(train_ds.epoch)
            batches = iter(loader)
        input_ids, labels, weight = next(batches)
        for row, valid in zip(input_ids, labels.ne(-100)):
            lengths.append(int(valid.sum()))
            fingerprints.add(hashlib.blake2b(row[valid].numpy().tobytes(), digest_size=12).digest())
        input_ids, labels, weight = (input_ids.to(args.device), labels.to(args.device),
                                     weight.to(args.device))
        lr = get_lr(step, stage_cfg["steps"], args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        window_start = (step - 1) // args.accumulation_steps * args.accumulation_steps
        window_size = min(args.accumulation_steps, total_steps - window_start)
        with autocast_ctx:
            if lm_config.use_loop and args.loop_mode == "full_unroll":
                res = model(input_ids, labels=None, collect_exits=True, collect_routers=False)
                exits = res.exit_logits
                total = sum(weighted_ce(lg, labels, weight) for lg in exits) / len(exits)
                loss = (total + res.aux_loss) / window_size
            else:
                r = max(1, args.fixed_loops)
                res = model(input_ids, labels=None, num_loops=r, collect_routers=False)
                total = weighted_ce(res.logits, labels, weight)
                loss = (total + res.aux_loss) / window_size
        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0 or step == total_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step % args.log_interval == 0 or step == stage_cfg["steps"]:
            eta = (time.time() - start_time) / step * (stage_cfg["steps"] - step) // 60
            Logger(f'Stage {stage_idx}({step}/{stage_cfg["steps"]}), loss: {total.item():.4f}, '
                   f'lr: {lr:.8f}, eta: {eta:.1f}min')
            if wandb:
                wandb.log({"loss": total.item(), "stage": stage_idx,
                           "max_len": stage_cfg["max_len"], "learning_rate": lr})
        del input_ids, labels, weight, loss
    shard_stats = [(lengths, fingerprints)]
    if dist.is_initialized():
        shard_stats = [None] * dist.get_world_size()
        dist.all_gather_object(shard_stats, (lengths, fingerprints))
    if is_main_process():
        lengths = sorted(n for ns, _ in shard_stats for n in ns)
        unique = len(set.union(*(hs for _, hs in shard_stats)))
        packing = dict(examples=len(lengths), valid_tokens=sum(lengths), min_length=lengths[0],
                       median_length=lengths[len(lengths)//2], max_length=lengths[-1],
                       duplicate_fraction=1 - unique / len(lengths))
        Logger(f"Stage {stage_idx} packing {packing}")
        model.eval()
        save_x_checkpoint(model, args.save_dir, f'{args.save_weight}_stage{stage_idx}',
                          lm_config.hidden_size, extra={"packing": packing})
        raw = getattr(model, "module", model)
        raw.checkpoint_parent = raw.last_saved_checkpoint
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-X Long-Context CPT (Stage 1)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='longctx_x', type=str)
    parser.add_argument("--batch_size", type=int, default=1, help="长序列建议 1 + 梯度累积")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="CPT 用小学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument('--variant', default='dense', type=str,
                        choices=["dense", "middle_loop", "minimind_x"],
                        help="D7：Stage 1 默认 dense（loop disabled）")
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"])
    parser.add_argument('--hidden_size', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--num_hidden_layers', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument('--loop_mode', default='fixed', type=str, choices=["fixed", "full_unroll"])
    parser.add_argument('--fixed_loops', default=1, type=int)
    parser.add_argument("--stage_lens", type=str, default="8192,32768,131072",
                        help="curriculum 各阶段序列长（逗号分隔；Phase A-C）")
    parser.add_argument("--rope_thetas", type=str, default=None,
                        help="各阶段 rope theta（逗号分隔，可选；默认沿用 config）")
    parser.add_argument("--steps_per_stage", type=str, default="1000,1000,1000",
                        help="各阶段 step 数（逗号分隔）")
    parser.add_argument("--needle_ratio", type=float, default=0.1, help="needle 样本混合比例")
    parser.add_argument("--needle_weight", type=float, default=5.0, help="gold token CE 加权")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument('--from_weight', default='pretrain_x', type=str)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-LongContext")
    parser.add_argument("--grad_ckpt", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode(args.device)
    if dist.is_initialized() and torch.device(args.device).type == "cuda":
        args.device = f"cuda:{local_rank}"
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(preset=args.preset, variant=args.variant, max_loop=args.max_loop,
                                **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                         num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"LongCtx-{args.stage_lens}")

    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    save_dir=args.save_dir, device=args.device, tokenizer_path=args.tokenizer_path)
    lm_config = model.config
    if args.grad_ckpt == 1:
        set_gradient_checkpointing(model, True)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank] if device_type == "cuda" else None, broadcast_buffers=False,
                                        find_unused_parameters=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    lens = [int(x) for x in args.stage_lens.split(",")]
    steps = [int(x) for x in args.steps_per_stage.split(",")]
    thetas = [float(x) for x in args.rope_thetas.split(",")] if args.rope_thetas else [None] * len(lens)
    assert len(lens) == len(steps) >= 1
    for i, (ml, st, th) in enumerate(zip(lens, steps, thetas)):
        train_stage(i, {"max_len": ml, "steps": st, "rope_theta": th}, 0, st, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
