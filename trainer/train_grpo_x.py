"""MiniMind-X Adaptive Compute RL（任务清单 T7.3；Doc 5 §18-25、Doc 4 §16-17）。

GRPO + Compute Penalty：
  R = R_task − λ_m·P_m − λ_r·P_r − λ_t·P_t
  R_task：math 正确性（safe_math_eval 判等）+ 轻量格式分
  P_t 使用实际循环深度，P_m 使用各轮读入的有效记忆 token，P_r 使用全局扫描和候选打分量。

默认在无截断的温度 categorical 分布上采样 token，按合法动作分布采样 halt、检索与预算。
有文档时由 controller 维护证据库和记忆。重放沿逐 token KV 历史复算相同事件及联合 logp。

RL 三相课程（Doc 5 §24）：phase1 无 compute penalty → phase2 小 penalty → phase3 全额 Pareto。
防塌缩（Doc 5 §25）：compute penalty 只在答案正确时生效 —— 简单题塌缩到 R=1 拿不到效率奖励。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import argparse
import json
import random
import re
import time
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from trainer.grpo_trajectory import trajectory, replay_logps
from model.minimind_x import MiniMindXConfig
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import (Logger, is_main_process, init_distributed_mode, setup_seed,
                                   safe_math_eval)
from trainer.x_utils import init_model_x, save_x_checkpoint, tokenize_tensors

warnings.filterwarnings('ignore')

PHASE_PENALTY = {1: (0.0, 0.0, 0.0), 2: (0.05, 0.05, 0.05), 3: (0.2, 0.1, 0.2)}  # (λ_m, λ_r, λ_t)


def extract_answer(text):
    """提取生成文本中最后一个数值/算式结果。"""
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:\s*[-+*/^]\s*-?\d+(?:\.\d+)?)*", text)
    return matches[-1].replace(" ", "").replace("^", "**") if matches else None


def math_task_reward(response, gt):
    """正确 +1；可解析但错 0；不可解析 -0.2。"""
    try:
        pred = safe_math_eval(extract_answer(response) or "")
        return 1.0 if abs(pred - float(gt)) < 1e-6 else 0.0
    except (ValueError, TypeError):
        return -0.2


_BANNED_PATTERNS = ("import os", "import sys", "import subprocess", "import socket",
                    "__import__", "open(", "exec(", "eval(", "shutil", "pathlib")


def _sandbox_limits():
    """研究级沙箱资源限制（RLIMIT）：CPU 5s、内存 512MB、无文件写出。非完全隔离。

    逐项尽力而为（preexec_fn 内不可抛异常）：Linux 生效；macOS 对部分 RLIMIT
    支持不全时跳过该项（仍有静态危险模式拒绝 + timeout 兜底）。
    """
    import resource

    def apply():
        limits = [(resource.RLIMIT_CPU, 5), (resource.RLIMIT_AS, 512 * 1024 * 1024),
                  (resource.RLIMIT_FSIZE, 0)]
        for which, soft in limits:
            try:
                hard = resource.getrlimit(which)[1]
                resource.setrlimit(which, (min(soft, hard) if hard != resource.RLIM_INFINITY
                                           else soft, hard))
            except (ValueError, OSError):
                pass
    return apply


def code_task_reward(response, gt, timeout=5):
    """代码沙箱 reward（T7.4）：提取 ```python 块，subprocess 执行，stdout 精确匹配。

    ⚠ 研究级隔离（RLIMIT + 危险模式静态拒绝），仅适用于自产/可信数据；
    不可信输入需外层容器（docker/nsjail），见 ARCHITECTURE §8。
    """
    import subprocess as sp
    import re as _re
    blocks = _re.findall(r"```python\n(.*?)```", response, _re.DOTALL)
    if not blocks:
        return -0.2
    code = blocks[-1]
    if any(p in code for p in _BANNED_PATTERNS):  # 纯计算任务无需这些能力
        return 0.0
    try:
        r = sp.run([sys.executable, "-c", code], capture_output=True, text=True,
                   timeout=timeout, preexec_fn=_sandbox_limits())
        return 1.0 if r.returncode == 0 and r.stdout.strip() == str(gt).strip() else 0.0
    except sp.TimeoutExpired:
        return 0.0


def format_reward(response):
    return 0.1 if 0 < len(response.strip()) < args.max_gen_len else -0.1


@torch.no_grad()
def rollout_group(prompt_text, num_generations, max_new_tokens, context=None, visibility=None):
    inputs = tokenize_tensors(tokenizer, prompt_text, args.device)["input_ids"]
    outs = {k: [] for k in ("ids", "completions", "loop", "mem", "ret", "records", "old_logps", "fixed_depth", "retrieval_cost")}
    policy = getattr(model, "module", model)
    for j in range(num_generations):
        depth = None
        if not args.adaptive_rollout:
            depth = min((1, 2, lm_config.max_loop)[j % 3], lm_config.max_loop) if args.mode_diversity else random.randint(1, lm_config.max_loop)
        records, old, reads, stats = trajectory(policy, inputs, context, max_new_tokens,
                                                      args.temperature, tokenizer.eos_token_id, depth,
                                                      visibility=visibility)
        gen = inputs.new_tensor([r["token"] for r in records])
        outs["ids"].append(gen)
        outs["completions"].append(tokenizer.decode(gen.tolist(), skip_special_tokens=True))
        outs["loop"].append(sum(r["depth"] for r in records) / len(records))
        outs["mem"].append(reads)
        outs["ret"].append(stats.get("retrievals", 0))
        scan = stats.get("scanned_blocks", 0)
        scored = stats.get("scored_blocks", 0)
        # Dot products per hidden dimension: full index scan plus token-wise fine ranking.
        work = scan + scored * lm_config.block_size
        blocks = (context.numel() + lm_config.block_size - 1) // lm_config.block_size if context is not None else 0
        full_search = blocks + min(blocks, lm_config.topk_coarse) * lm_config.block_size
        outs["retrieval_cost"].append(work / max(1, len(records) * full_search))
        outs["records"].append(records)
        outs["old_logps"].append(old)
        outs["fixed_depth"].append(depth)
    return outs


def compute_per_token_logps(model_, input_ids, n_keep, attention_mask=None, num_loops=1):
    """num_loops 与轨迹实际轮数对齐（否则 off-policy 偏差：生成用自适应 R、评估用 R=1）。"""
    # 评审 8.2：梯度前向必须走 DDP 包装（unwrap 会丢梯度同步）
    # 推理/无梯度可用 raw；此处要 backward，故用 model_ 本身
    target = model_
    logits = target(input_ids, attention_mask=attention_mask, num_loops=num_loops,
                       logits_to_keep=n_keep + 1, collect_routers=False).logits[:, :-1, :]
    logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        logps.append(F.log_softmax(logits_row, dim=-1).gather(
            1, ids_row.unsqueeze(1)).squeeze(1))
    return torch.stack(logps)


def grpo_x_epoch(epoch, loader, iters, ref_model, start_step=0, wandb=None):
    lam_m, lam_r, lam_t = PHASE_PENALTY[args.rl_phase]
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = [tokenizer.apply_chat_template(b["messages"], tokenize=False,
                                                 add_generation_prompt=True) for b in batch]
        gts = [b["gt"] for b in batch]
        from dataset.retrieval_qa_dataset import encode_evidence
        contexts, visibility = [], []
        for row in batch:
            doc, _, rounds = encode_evidence(row, tokenizer, lm_config.block_size)
            contexts.append(doc.to(args.device))
            visibility.append(rounds.to(args.device))
        rollouts = [rollout_group(p, args.num_generations, args.max_gen_len, doc, rounds)
                    for p, doc, rounds in zip(prompts, contexts, visibility)]
        prompt_ids = [tokenize_tensors(tokenizer, p, args.device)["input_ids"] for p in prompts]
        traj = [(i, j, ro) for i, ro in enumerate(rollouts) for j in range(args.num_generations)]

        rewards = torch.zeros(len(traj), device=args.device)
        task_fn = code_task_reward if args.reward_type == "code" else math_task_reward
        for k, (i, j, ro) in enumerate(traj):
            task_score = task_fn(ro["completions"][j], gts[i])
            task = task_score + format_reward(ro["completions"][j])
            p_t = ro["loop"][j] / max(lm_config.max_loop, 1)
            query_tokens = prompt_ids[i].numel() + len(ro["records"][j]) - 1
            p_m = ro["mem"][j] / (query_tokens * lm_config.max_loop * max(lm_config.memory_budgets))
            p_r = ro["retrieval_cost"][j]
            correct = task_score >= 1.0
            # 防塌缩：效率奖励只给「正确」的轨迹（Minimum reward constraint）
            penalty = (lam_m * p_m + lam_r * p_r + lam_t * p_t) if correct else 0.0
            rewards[k] = task - penalty

        grouped = rewards.view(-1, args.num_generations)
        mean_r = grouped.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        if args.trajectory_log:
            from pathlib import Path
            rank = dist.get_rank() if dist.is_initialized() else 0
            path = Path(f"{args.trajectory_log}.rank{rank}.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                for k, (i, j, ro) in enumerate(traj):
                    record = dict(epoch=epoch, step=step, phase=args.rl_phase, group=i, member=j,
                                  temperature=args.temperature, fixed_depth=ro["fixed_depth"][j],
                                  prompt_ids=prompt_ids[i][0].tolist(), context_tokens=contexts[i].numel(),
                                  events_by_token=ro["records"][j], reward=rewards[k].item())
                    stream.write(json.dumps(record) + "\n")

        # ---- 策略更新（逐轨迹累积，GRPO clip + KL）----
        optimizer.zero_grad(set_to_none=True)
        policy_loss_total, kl_total, n_traj = 0.0, 0.0, len(traj)
        for k, (i, j, ro) in enumerate(traj):
            replay_args = (prompt_ids[i], contexts[i], ro["records"][j], args.temperature,
                           ro["fixed_depth"][j], visibility[i])
            per_token_logps = replay_logps(model, *replay_args, verify_behavior=True)
            old = ro["old_logps"][j]
            torch.testing.assert_close(per_token_logps.detach(), old, rtol=1e-5, atol=1e-5)
            with torch.no_grad():
                ref_logps = replay_logps(ref_model, *replay_args)
            delta = ref_logps - per_token_logps
            kl = torch.expm1(delta) - delta
            ratio = torch.exp(per_token_logps - old)
            clipped = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            a1, a2 = ratio * advantages[k], clipped * advantages[k]
            token_loss = -(torch.min(a1, a2) - args.beta * kl).mean()
            (token_loss / n_traj).backward()
            policy_loss_total += token_loss.item() / n_traj
            kl_total += kl.mean().item() / n_traj
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if args.adaptive_rollout and lm_config.use_retrieval and not getattr(lm_config, "use_budget_router", False):
            # A text-only RL stage must preserve an inherited fixed-budget baseline.
            trained = any(contexts[i].numel() and grouped[i].max() > grouped[i].min()
                          for i in range(len(contexts)))
            flag = torch.tensor(int(trained), device=args.device)
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            lm_config.use_budget_router = bool(flag.item())

        evidence_reads = sum(sum(ro["mem"]) for ro in rollouts)
        retrievals = sum(sum(ro["ret"]) for ro in rollouts)
        if step % args.log_interval == 0 or step == iters:
            acc = sum(1 for k, (i, _, ro) in enumerate(traj)
                      if task_fn(ro["completions"][k % args.num_generations], gts[i]) >= 1.0) / len(traj)
            avg_loop = sum(ro["loop"][k % args.num_generations] for k, (_, _, ro) in enumerate(traj)) / len(traj)
            Logger(f'GRPO-X Epoch:[{epoch + 1}]({step}/{iters}), reward: {rewards.mean().item():.4f}, '
                   f'acc: {acc:.3f}, avg_loop: {avg_loop:.2f}, KL: {kl_total:.4f}, '
                   f'policy_loss: {policy_loss_total:.4f}, memory_token_reads: {evidence_reads}, '
                   f'retrievals: {retrievals}, phase: {args.rl_phase}')
            if wandb:
                wandb.log({"reward": rewards.mean().item(), "accuracy": acc,
                           "avg_loops": avg_loop, "kl": kl_total,
                           "policy_loss": policy_loss_total})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            save_x_checkpoint(model, args.save_dir, args.save_weight, lm_config.hidden_size)
            model.eval()
        del traj, rewards, advantages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-X Adaptive Compute GRPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default=None, type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2, help="每步 prompt 数")
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_generations", type=int, default=3, help="每 prompt 轨迹数（3 的倍数配模式循环）")
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--trajectory_log", help="Optional prefix for per-rank JSONL behavior traces")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.05, help="KL 惩罚系数")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rl_phase", type=int, default=1, choices=[1, 2, 3],
                        help="RL 课程：1 无惩罚 / 2 小惩罚 / 3 Pareto（Doc 5 §24）")
    parser.add_argument("--mode_diversity", default=1, type=int, choices=[0, 1],
                        help="轨迹按 fast/normal/deep 模式循环采样")
    parser.add_argument("--adaptive_rollout", default=1, type=int, choices=[0, 1],
                        help="1 samples and replays token/controller decisions; 0 fixes depth per trajectory")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"])
    parser.add_argument('--variant', default='minimind_x', type=str)
    parser.add_argument('--hidden_size', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--num_hidden_layers', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument('--residual_v2', default=0, type=int, choices=[0, 1],
                        help='V2 残差改造；从 V2 ckpt 加载时会自动强制打开')
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl_math.jsonl",
                        help="math RL 数据：{conversations, gt}（官方 agent_rl_math.jsonl）")
    parser.add_argument("--reward_type", type=str, default="math", choices=["math", "code"],
                        help="math=safe_math_eval 判等；code=python 沙箱 stdout 匹配（T7.4）")
    parser.add_argument('--from_weight', default=None, type=str)
    parser.add_argument("--tokenizer_path", type=str, default="../model",
                        help="tokenizer 目录（从仓库根目录运行时传 model/）")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-GRPO")
    args = parser.parse_args()
    if args.from_weight is None:
        args.from_weight = "sft_x_c" if args.rl_phase == 1 else f"grpo_x_phase{args.rl_phase - 1}"
    if args.save_weight is None:
        args.save_weight = f"grpo_x_phase{args.rl_phase}"

    local_rank = init_distributed_mode(args.device)
    if dist.is_initialized() and torch.device(args.device).type == "cuda":
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(preset=args.preset, variant=args.variant, max_loop=args.max_loop,
                                residual_v2=bool(args.residual_v2),
                                **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                         num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    device_type = "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"GRPO-X-phase{args.rl_phase}")

    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    tokenizer_path=args.tokenizer_path,
                                    save_dir=args.save_dir, device=args.device)
    lm_config = model.config
    model.eval()  # Keep dropout identical in sampling and differentiable replay.
    ref_model, _ = init_model_x(lm_config, args.from_weight,
                                tokenizer_path=args.tokenizer_path,
                                save_dir=args.save_dir, device=args.device)
    ref_model.eval().requires_grad_(False)

    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=512)

    def _collate(b):
        for x in b:  # tools=None 会卡 default_collate，GRPO 不用它
            x.setdefault("tools", [])
            if x["tools"] is None:
                x["tools"] = []
        return b

    # T0.5（C9 修复）：DDP 包装与分片。rollout 走 raw（no_grad、generate 无 hook），
    # 策略梯度的 logps 前向必须过 DDP 包装（train_grpo 同款注释）。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                        sampler=train_sampler, collate_fn=_collate)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank] if device_type == "cuda" else None,
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)
    iters = len(loader)
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        grpo_x_epoch(epoch, loader, iters, ref_model, 0, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
