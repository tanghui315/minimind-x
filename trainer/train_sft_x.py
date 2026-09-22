"""MiniMind-X SFT 三相训练（任务清单 T7.2；Doc 5 §11、Doc 3 Phase 1-3、Doc 4 §20）。

SFT-A：基础 instruction，固定 R=1（避免依赖 loop）
SFT-B：随机 loop 建立深度弹性（collect_exits 均匀监督 + 一致性）
SFT-C：打开 retrieval/working memory（AdaptiveController teacher-forcing）+
        router 课程（thinking → retrieval → memory，Doc 4 §20）+ 在线 oracle 标签
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
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

from model.minimind_x import MiniMindXConfig, joint_loss
from model.minimind_x.losses import lm_loss_per_exit, consistency_kl_loss
from model.minimind_x.controller import AdaptiveController
from dataset.lm_dataset import SFTDataset
from dataset.retrieval_qa_dataset import RetrievalQADataset, collate_retrieval_qa
from trainer.trainer_utils import (get_lr, Logger, is_main_process, init_distributed_mode,
                                   setup_seed, lm_checkpoint)
from trainer.x_utils import (init_model_x, save_x_checkpoint, set_gradient_checkpointing,
                             build_optimizer, validate_config)
from trainer.residual_schedule import residual_progress, apply_residual_progress
from trainer.oracle_labels import (sweep_depth, oracle_depth_from_losses, sweep_budget,
                                   oracle_budget_from_matrix, sweep_retrieval)

warnings.filterwarnings('ignore')


def sft_a_step(input_ids, labels):
    """R=1 固定（SFT-A）。"""
    res = model(input_ids, labels=labels, num_loops=1)
    return res.loss + res.aux_loss, {"loss": res.loss.item()}


def sft_b_step(input_ids, labels):
    """随机深度弹性（SFT-B）：本 step 采样 R；按概率切换全展开做 router 无关的一致性监督。"""
    if random.random() < args.full_unroll_ratio:
        res = model(input_ids, labels=None, collect_exits=True)
        lm, per_exit = lm_loss_per_exit(res.exit_logits, labels, "uniform")
        loss = lm + res.aux_loss
        if args.alpha_consistency > 0:
            loss = loss + args.alpha_consistency * consistency_kl_loss(res.exit_logits, labels)
        return loss, {"loss": lm.item(), "per_exit": [round(l.item(), 4) for l in per_exit]}
    r = random.randint(1, lm_config.max_loop)
    res = model(input_ids, labels=labels, num_loops=r)
    return res.loss + res.aux_loss, {"loss": res.loss.item(), "r": r}


def assert_sft_c_batch(input_ids):
    """SFT-C batch 契约（v1.3 guard，评审 Bug 1）：controller 逐样本建 store/检索，
    B>1 时只用到首样本上下文（静默丢弃）。DataLoader 侧已强制 batch_size=1，
    此函数是第二道防线（防未来改动）。"""
    if input_ids.shape[0] > 1:
        raise ValueError(
            f"SFT-C 暂不支持 batch>1（当前 {input_ids.shape[0]}）：controller 逐样本建 "
            f"store/检索，B>1 时只用到首样本上下文。请用 batch_size=1 + 梯度累积。")


def sft_c_step(batch):
    """终局 teacher-forcing（SFT-C / Stage 3）：controller + joint loss + 在线 oracle。

    V2 改造（T0.1/T0.2/T0.8）：query 因果边界、反事实检索 oracle、全展开 router 概率
    监督、memory 统计只进日志。
    """
    assert_sft_c_batch(batch["input_ids"])
    input_ids = batch["input_ids"].to(args.device)
    labels = batch["labels"].to(args.device)
    context_ids = batch["context_ids"][0].to(args.device)  # 研究级口径：batch 内取首样本上下文
    gold_block = batch["gold_block"].to(args.device)
    query_len = int(batch["question_len"][0].item())       # T0.1：可见前缀边界（数据集提供）
    oracle_r_file = int(batch["oracle_r"][0].item())
    oracle_ret = batch["oracle_retrieve"][0].tolist()
    if not isinstance(oracle_ret, (list, tuple)):
        oracle_ret = [int(oracle_ret)] if int(oracle_ret) >= 0 else []

    # ---- 在线 oracle（Doc 4 §3：Self-generated Supervision；小采样率）----
    oracle_r = None
    halt_valid = None  # T4.4：unresolved（全出口都错）不参与 halt 监督
    if oracle_r_file > 0:
        oracle_r = torch.full((input_ids.shape[0],), oracle_r_file, device=args.device)
    elif random.random() < args.oracle_rate:
        per_exit_loss, per_exit_correct, _ = sweep_depth(model, input_ids, labels)
        oracle_r = oracle_depth_from_losses(per_exit_loss, lm_config.oracle_epsilon)
        halt_valid = per_exit_correct.any(dim=-1)  # 有任一出口全对才算 resolved
    # Budget oracle：文件标签缺失时在线 sweep（逐档 loss → B*，Doc 4 §11）；
    # 未采到的批次不监督（budget router 仍按自身 argmax 执行，自洽）
    oracle_budget = None
    if args.train_budget_router and random.random() < args.oracle_rate:
        loss_matrix = sweep_budget(
            model, input_ids, labels, context_ids,
            controller_factory=lambda: AdaptiveController(model))
        oracle_budget = int(oracle_budget_from_matrix(loss_matrix, lm_config.oracle_epsilon)[0].item())
    if not oracle_ret and random.random() < args.oracle_rate:
        # T2.5/T4.4：三动作反事实（REUSE/RESCORE_WARM/GLOBAL_REFRESH）；
        # 无 warm cache 的动作 loss=+inf 自动 mask
        try:
            from trainer.oracle_labels import sweep_retrieval_action
            best_a, _ = sweep_retrieval_action(
                model, input_ids, labels, context_ids, query_len,
                controller_factory=lambda: AdaptiveController(model))
            oracle_ret = [int(x) for x in best_a[0].tolist()]
        except Exception:
            labels_r, _ = sweep_retrieval(model, input_ids, labels, context_ids, query_len,
                                          controller_factory=lambda: AdaptiveController(model))
            oracle_ret = labels_r[0].tolist()  # 回退：旧 0/1 口径

    ctrl = AdaptiveController(model)
    out, extras = ctrl.forward_with_oracles(
        input_ids, None, labels, context_ids, query_len=query_len,
        gold_block_idx=gold_block if (gold_block[0] >= 0).all() else None,
        oracle_retrieve=oracle_ret if oracle_ret else None,
        oracle_budget=oracle_budget)
    # T0.2：全展开 router 概率（不再要求 halt=True 才有监督信号）
    halt_probs = None
    if (oracle_r is not None and args.train_halt_router
            and out.router_probs and out.router_probs.get("halt")):
        halt_probs = out.router_probs["halt"]
    retrieve_probs = None
    if (oracle_ret and out.router_probs and out.router_probs.get("retrieval")
            and len(oracle_ret) == len(out.router_probs["retrieval"])):
        retrieve_probs = out.router_probs["retrieval"]
    # T2.5：三动作 CE（0=REUSE, 1=RESCORE_WARM, 2=GLOBAL_REFRESH 的执行口径见 oracle；
    # 训练标签统一为 0=REUSE / 1=RESCORE_WARM / 2=GLOBAL_REFRESH 时用 action CE）
    ret_action_logits = None
    ret_action_masks = None
    oracle_ret_action = None
    if (oracle_ret and out.router_probs and out.router_probs.get("retrieval_action")
            and len(oracle_ret) == len(out.router_probs["retrieval_action"])):
        ret_action_logits = out.router_probs["retrieval_action"]
        warm_ok = bool(extras.get("warm_available", True))
        n = len(ret_action_logits)
        from model.minimind_x.router.retrieval_router import (
            action_mask, ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH)
        ret_action_masks = [action_mask(warm_ok, device=ret_action_logits[0].device,
                                        batch=ret_action_logits[0].shape[0]) for _ in range(n)]
        # 归一到 ACTION_* id：旧 0/1 口径 1→GLOBAL_REFRESH；新 2→RESCORE_WARM
        def _to_action(a):
            a = int(a)
            if a == 0:
                return ACTION_REUSE
            if a == 2:
                return ACTION_RESCORE_WARM
            return ACTION_GLOBAL_REFRESH  # 1 或其它
        oracle_ret_action = torch.tensor([_to_action(a) for a in oracle_ret], device=args.device)
    # 二值 retrieve BCE（兼容旧路径）：标签 = 是否离开 REUSE
    oracle_ret_bin = None
    if oracle_ret and retrieve_probs is not None:
        oracle_ret_bin = torch.tensor(
            [0 if int(a) == 0 else 1 for a in oracle_ret], device=args.device)
    parts = joint_loss(out.exit_logits, labels,
                       halt_probs=halt_probs, oracle_r=oracle_r, halt_valid=halt_valid,
                       retrieve_probs=retrieve_probs,
                       oracle_retrieve=oracle_ret_bin,
                       budget_logits=out.budget_logits if args.train_budget_router else None,
                       oracle_budget=oracle_budget,
                       memory_size=torch.tensor([extras["memory_tokens"]]),  # T0.8：仅日志
                       config=lm_config,
                       retrieval_action_logits=ret_action_logits,
                       oracle_retrieve_action=oracle_ret_action,
                       retrieval_action_masks=ret_action_masks)
    zero = next(model.parameters()).new_zeros(())
    loss = parts["loss"] + zero + extras.get("retrieval_ce", zero)
    if "distill_loss" in extras and args.lambda_distill > 0:
        loss = loss + args.lambda_distill * extras["distill_loss"]
    return loss, {"loss": parts["loss"].item(),
                  "retrievals": extras["retrievals"],
                  "memory_tokens": extras["memory_tokens"],
                  "budget_cap/actual/unique": (extras.get("budget_cap_tokens", 0),
                                               extras.get("actual_valid_tokens", 0),
                                               extras.get("unique_blocks", 0)),
                  "distill": float(extras["distill_loss"].detach())
                  if "distill_loss" in extras else 0.0}


def main():
    global args, lm_config, model, tokenizer
    parser = argparse.ArgumentParser(description="MiniMind-X SFT (A/B/C)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='sft_x', type=str)
    parser.add_argument("--phase", type=str, required=True, choices=["a", "b", "c"],
                        help="SFT-A(R=1) / SFT-B(随机R弹性) / SFT-C(memory+retrieval)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0,
                        help=">0 时限制总步数（大数据集夜间跑部分 epoch 用）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"])
    parser.add_argument('--variant', default='minimind_x', type=str)
    parser.add_argument('--hidden_size', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--num_hidden_layers', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument('--full_unroll_ratio', default=0.25, type=float,
                        help="SFT-B 中全展开（带一致性监督）的批次比例")
    parser.add_argument('--alpha_consistency', default=0.1, type=float)
    parser.add_argument('--oracle_rate', default=0.1, type=float, help="在线 oracle 采样率")
    parser.add_argument('--train_halt_router', default=1, type=int, choices=[0, 1],
                        help="router 课程 Phase2：thinking router 开关")
    parser.add_argument('--train_budget_router', default=0, type=int, choices=[0, 1],
                        help="router 课程 Phase4：memory router 开关")
    parser.add_argument('--residual_v2', default=0, type=int, choices=[0, 1])
    parser.add_argument('--residual_warmup_steps', default=0, type=int)
    parser.add_argument('--lambda_distill', default=0.05, type=float,
                        help="fine→coarse 检索蒸馏系数（v1.3；0 关闭）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl",
                        help="SFT-A/B 数据（官方 sft_t2t_mini，conversations 格式）")
    parser.add_argument("--retrieval_data_path", type=str,
                        default="../dataset/retrieval_qa.jsonl", help="SFT-C 数据")
    parser.add_argument('--from_weight', default='pretrain_x', type=str)
    parser.add_argument("--tokenizer_path", type=str, default="../model",
                        help="tokenizer 目录（从仓库根目录运行时传 model/）")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-SFT")
    parser.add_argument("--grad_ckpt", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindXConfig(preset=args.preset, variant=args.variant, max_loop=args.max_loop,
                                residual_v2=bool(args.residual_v2),
                                **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                         num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    validate_config(lm_config)  # T5.2：非法 config 启动即报错
    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    tokenizer_path=args.tokenizer_path,
                                    save_dir=args.save_dir, device=args.device)
    model.train()
    if args.grad_ckpt == 1:
        set_gradient_checkpointing(model, True)
    # T0.5（C9 修复）：DDP 包装与数据分片（routers/manager 按阶段激活 → find_unused）
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"SFT-{args.phase}")

    if args.phase in ("a", "b"):
        ds = SFTDataset(args.data_path, tokenizer, max_length=1024)
        collate = None
    else:
        ds = RetrievalQADataset(args.retrieval_data_path, tokenizer,
                                max_context_tokens=lm_config.max_position_embeddings,
                                block_size=lm_config.block_size)
        collate = collate_retrieval_qa
        assert lm_config.use_retrieval and lm_config.use_memory, "SFT-C 需要 minimind_x 变体"
    train_sampler = DistributedSampler(ds) if dist.is_initialized() else None
    loader = DataLoader(ds, batch_size=args.batch_size if args.phase != "c" else 1,
                        shuffle=(train_sampler is None), sampler=train_sampler,
                        num_workers=args.num_workers, collate_fn=collate)
    optimizer = build_optimizer(model, lr=args.learning_rate)  # T4.3 参数组
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    iters = len(loader) * args.epochs
    step_global = 0
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        t0 = time.time()
        for step, batch in enumerate(loader, 1):
            step_global += 1
            lr = get_lr(step_global, iters, args.learning_rate)
            for pg in optimizer.param_groups:  # 参数组调度：cosine 基准 × 组倍率
                pg['lr'] = lr * pg.get("mult", 1.0)
            if getattr(args, "residual_warmup_steps", 0) > 0:
                frac = residual_progress(step_global, args.residual_warmup_steps)
                eta, lam = apply_residual_progress(model, frac)
                if step % args.log_interval == 0:
                    Logger(f'residual warm-up: step={step_global} frac={frac:.3f} '
                           f'eta={eta:.3f} lambda={lam:.3f}')

            if args.phase == "a":
                input_ids, labels = batch[0].to(args.device), batch[1].to(args.device)
                with autocast_ctx:
                    loss, info = sft_a_step(input_ids, labels)
            elif args.phase == "b":
                input_ids, labels = batch[0].to(args.device), batch[1].to(args.device)
                with autocast_ctx:
                    loss, info = sft_b_step(input_ids, labels)
            else:
                with autocast_ctx:
                    loss, info = sft_c_step(batch)
            scaler.scale(loss / args.accumulation_steps).backward()
            if step % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % args.log_interval == 0 or step == len(loader):
                eta = (time.time() - t0) / step * (len(loader) - step) // 60
                Logger(f'SFT-{args.phase.upper()} Epoch[{epoch + 1}]({step}/{len(loader)}), '
                       f'loss: {info["loss"]:.4f}, lr: {lr:.8f}, eta: {eta:.1f}min')
                if wandb:
                    wandb.log({"loss": info["loss"], "phase": args.phase})
            if args.max_steps > 0 and step_global >= args.max_steps:
                Logger(f'SFT-{args.phase.upper()} 达到 max_steps={args.max_steps}，提前结束')
                break
        if args.max_steps > 0 and step_global >= args.max_steps:
            break
        if is_main_process():
            save_x_checkpoint(  # T5.2：完整 checkpoint
                model, args.save_dir, f'{args.save_weight}_{args.phase}',
                lm_config.hidden_size, optimizer=optimizer, scaler=scaler,
                extra={"data_progress": {"epoch": epoch + 1, "step_global": step_global},
                       "warmup_progress": getattr(args, "residual_warmup_steps", 0),
                       "tokenizer_manifest": {"path": args.tokenizer_path},
                       "source_manifest": {"data_path": getattr(args, "data_path", None),
                                           "retrieval_data_path": getattr(args, "retrieval_data_path", None)}})


if __name__ == "__main__":
    main()
