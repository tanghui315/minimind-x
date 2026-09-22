"""MiniMind-X 预训练（任务清单 T1.3；Doc 3 §9 Phase 0、Doc 5 Stage 0）。

loop_mode（D3/D7）：
  fixed        固定 R=args.fixed_loops（Stage 0/1 默认：求训练稳定）
  random       每 step 随机采样 R∈[1,R_max]（Doc 3 Phase 0 的随机-loop 预训练，省算力）
  full_unroll  全展开 R_max + per-exit 监督（loss@R 曲线是核心研究产出）
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import random
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.minimind_x import MiniMindXConfig
from model.minimind_x.losses import lm_loss_per_exit, consistency_kl_loss
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (get_lr, Logger, is_main_process, lm_checkpoint,
                                   init_distributed_mode, setup_seed, SkipBatchSampler)
from trainer.x_utils import (init_model_x, save_x_checkpoint, set_gradient_checkpointing,
                             build_optimizer, validate_config)
from trainer.residual_schedule import residual_progress, apply_residual_progress

warnings.filterwarnings('ignore')

# checkpoint 目录锚定到仓库根（原 '../checkpoints' 在从仓库根运行时会落到项目外）
CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))


def compute_step_loss(model, input_ids, labels, args, lm_config):
    """返回 (总损失, 日志 dict)。日志里 loss@R 逐出口记录（研究核心指标）。"""
    if not lm_config.use_loop or args.loop_mode == "fixed":
        r = max(1, args.fixed_loops)
        res = model(input_ids, labels=labels, num_loops=r)
        loss = res.loss + res.aux_loss
        return loss, {"loss": res.loss.item(), "r": r}
    if args.loop_mode == "random":
        r = random.randint(1, lm_config.max_loop)
        res = model(input_ids, labels=labels, num_loops=r)
        loss = res.loss + res.aux_loss
        return loss, {"loss": res.loss.item(), "r": r}
    if args.loop_mode != "full_unroll":
        raise ValueError(f"未知 loop_mode: {args.loop_mode}")
    res = model(input_ids, labels=None, collect_exits=True)
    lm, per_exit = lm_loss_per_exit(res.exit_logits, labels, lm_config.exit_loss_weighting)
    loss = lm + res.aux_loss
    if args.alpha_consistency > 0:
        loss = loss + args.alpha_consistency * consistency_kl_loss(res.exit_logits, labels)
    return loss, {"loss": lm.item(),
                  "per_exit": [round(l.item(), 4) for l in per_exit],
                  "r": lm_config.max_loop}


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if args.residual_w2 if False else False:
            pass
        if getattr(args, 'residual_warmup_steps', 0) > 0:
            frac = residual_progress(step, args.residual_warmup_steps)
            eta, lam = apply_residual_progress(model, frac)
            if step % args.log_interval == 0:
                Logger(f'residual warm-up: step={step} frac={frac:.3f} eta={eta:.3f} lambda={lam:.3f}')
        with autocast_ctx:
            loss, log_info = compute_step_loss(model, input_ids, labels, args, lm_config)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            exit_str = f", loss@R: {log_info.get('per_exit')}" if log_info.get("per_exit") else ""
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {log_info["loss"]:.4f}'
                   f'{exit_str}, R: {log_info["r"]}, lr: {lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                payload = {"loss": log_info["loss"], "learning_rate": lr, "epoch_time": eta_min}
                if log_info.get("per_exit"):
                    payload.update({f"loss_R{ i + 1}": v for i, v in enumerate(log_info["per_exit"])})
                wandb.log(payload)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            save_x_checkpoint(  # T5.2：完整 checkpoint
                model, args.save_dir, args.save_weight, lm_config.hidden_size,
                optimizer=optimizer, scaler=scaler,
                extra={"data_progress": {"epoch": epoch + 1, "step": step},
                       "warmup_progress": getattr(args, "residual_warmup_steps", 0),
                       "tokenizer_manifest": {"path": args.tokenizer_path},
                       "source_manifest": {"data_path": getattr(args, "data_path", None)}})
            # 保存「下一个待跑位置」：epoch 结束时存 (epoch+1, 0)，中途存 (epoch, step)
            done = step >= iters
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          scaler=scaler, epoch=epoch + 1 if done else epoch,
                          step=0 if done else step, batch_size=args.batch_size,
                          wandb=wandb, save_dir=CKPT_DIR)
            model.train()

        del input_ids, labels, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-X Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_x', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"], help="规模 preset")
    parser.add_argument('--variant', default='middle_loop', type=str,
                        choices=["dense", "full_loop", "middle_loop", "late_middle", "minimind_x"])
    parser.add_argument('--hidden_size', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--num_hidden_layers', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--prelude_layers', default=None, type=int)
    parser.add_argument('--thinking_layers', default=None, type=int)
    parser.add_argument('--coda_layers', default=None, type=int)
    parser.add_argument('--max_loop', default=4, type=int, help="R_max")
    parser.add_argument('--loop_mode', default='full_unroll', type=str,
                        choices=["fixed", "random", "full_unroll"])
    parser.add_argument('--loop_embedding', default=0, type=int, choices=[0, 1],
                        help="轮次嵌入消融（D4）。默认 0：随 MoR/Huginn 实践不加；"
                             "1：Universal-Transformer 式 step embedding（对照实验）")
    parser.add_argument('--fixed_loops', default=1, type=int, help="loop_mode=fixed 时的 R")
    parser.add_argument('--alpha_consistency', default=0.0, type=float, help="一致性 KL 系数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--prelude_attn', default=None, type=str, choices=[None, "full", "local", "sparse"])
    parser.add_argument('--core_attn', default=None, type=str, choices=[None, "full", "sparse"])
    parser.add_argument('--coda_attn', default=None, type=str, choices=[None, "full", "local", "sparse"])
    parser.add_argument('--max_seq_len', default=340, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl",
                        help="预训练数据（官方：pretrain_t2t_mini 1.2GB 快速 / pretrain_t2t 10GB 完整）")
    parser.add_argument("--tokenizer_path", type=str, default="../model",
                        help="tokenizer 目录（从仓库根目录运行时传 model/）")
    parser.add_argument('--from_weight', default='none', type=str, help="X 权重前缀（none=从头）")
    parser.add_argument('--from_minimind_weight', default='none', type=str,
                        help="原 MiniMind 权重路径（自动 remap，T0.3 训练侧入口）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-Pretrain")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--grad_ckpt", default=0, type=int, choices=[0, 1], help="梯度检查点（T1.4）")
    parser.add_argument("--residual_v2", default=0, type=int, choices=[0, 1],
                        help="启用 V2 残差改造（MHAR/UpdateGate/Reader）")
    parser.add_argument("--residual_warmup_steps", default=0, type=int,
                        help="V2 迁移热身步数（η/λ 0→1；0=从头恒 1）")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(
        preset=args.preset, variant=args.variant, use_moe=bool(args.use_moe),
        max_loop=args.max_loop, loop_embedding=bool(args.loop_embedding),
        residual_v2=bool(args.residual_v2),
        **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                 num_hidden_layers=args.num_hidden_layers,
                                 prelude_layers=args.prelude_layers,
                                 thinking_layers=args.thinking_layers,
                                 coda_layers=args.coda_layers,
                                 prelude_attn=args.prelude_attn,
                                 core_attn=args.core_attn,
                                 coda_attn=args.coda_attn).items() if v is not None})
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir=CKPT_DIR) if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"MiniMindX-{args.variant}-{args.loop_mode}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    validate_config(lm_config)  # T5.2：非法 config 启动即报错
    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    tokenizer_path=args.tokenizer_path,
                                    save_dir=args.save_dir, device=args.device)
    if args.from_minimind_weight != 'none':
        from model.minimind_x import remap_minimind_state_dict
        orig = torch.load(args.from_minimind_weight, map_location=args.device)
        mapped = remap_minimind_state_dict(orig, lm_config)
        model.load_state_dict(mapped, strict=False)
        Logger(f'remapped original MiniMind weights from {args.from_minimind_weight}')

    if args.grad_ckpt == 1:
        set_gradient_checkpointing(model, True)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model, lr=args.learning_rate)  # T4.3 参数组

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)
        # mid-epoch 续跑仅在同配置下有效：step 是 per-rank 口径，不随 world_size 缩放
        # （昨晚的 bug：8 卡存的 step 在 4 卡 resume 时被错误翻倍导致步数超界）。
        # 跨卡数/批大小恢复时，当前 epoch 从头训练（数据顺序重排，语义最干净）。
        cur_ws = dist.get_world_size() if dist.is_initialized() else 1
        if (ckp_data.get('world_size', cur_ws) != cur_ws
                or ckp_data.get('batch_size', args.batch_size) != args.batch_size):
            Logger(f'resume: 配置变化(ws {ckp_data.get("world_size", cur_ws)}->{cur_ws} / '
                   f'bs {ckp_data.get("batch_size", args.batch_size)}->{args.batch_size})，'
                   f'epoch {start_epoch} 从头训练')
            start_step = 0
        if start_epoch >= args.epochs:
            Logger(f'resume: 已完成 {start_epoch}/{args.epochs} epochs，无需训练')

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # X 模型的 router/memory 模块按训练阶段激活（Stage 0 不监督 router），
        # 必须放开 unused 检测，否则 DDP 在第二轮 reduction 报错
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
