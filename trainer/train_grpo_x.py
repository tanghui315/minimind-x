"""MiniMind-X Adaptive Compute RL（任务清单 T7.3；Doc 5 §18-25、Doc 4 §16-17）。

GRPO + Compute Penalty：
  R = R_task − λ_m·P_m − λ_r·P_r − λ_t·P_t
  R_task：math 正确性（safe_math_eval 判等）+ 轻量格式分
  P_t = avg_loops/R_max（来自 generate 统计）；P_m/P_r 来自 controller 统计（如有）

轨迹多样性：同一 prompt 的生成组按 [fast, normal, deep] 模式循环 → (memory, loop) 维度的组内对比。

RL 三相课程（Doc 5 §24）：phase1 无 compute penalty → phase2 小 penalty → phase3 全额 Pareto。
防塌缩（Doc 5 §25）：compute penalty 只在答案正确时生效 —— 简单题塌缩到 R=1 拿不到效率奖励。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import argparse
import math
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

from model.minimind_x import MiniMindXConfig
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import (Logger, is_main_process, init_distributed_mode, setup_seed,
                                   safe_math_eval)
from trainer.x_utils import init_model_x, save_x_checkpoint, tokenize_tensors

warnings.filterwarnings('ignore')

MODE_CYCLE = ["fast", "normal", "deep"]
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
def rollout_group(prompt_text, num_generations, max_new_tokens):
    """一个 prompt 的生成组：按 [fast, normal, deep] 循环产生 (memory, loop) 多样性。

    默认（on-policy，v1.2 修正）：每条轨迹固定采样 R（fast=1/normal=2/deep=R_max，
    无 mode 多样性时随机 R∈[1,R_max]），生成与 logps 重算的计算路径严格一致。
    --adaptive_rollout 1 为消融路径：adaptive halting 采样，轨迹内各 token 的 R 不同，
    logps 只能用单一 R 近似重算（残余 off-policy 偏差），仅用于对比实验。
    """
    inputs = tokenize_tensors(tokenizer, prompt_text, args.device)
    outs = {"ids": [], "completions": [], "loop": [], "mem": [], "ret": []}
    for j in range(num_generations):
        mode = MODE_CYCLE[j % len(MODE_CYCLE)] if args.mode_diversity else None
        if args.adaptive_rollout == 1:
            res = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens, do_sample=True, temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id, mode=mode,
                use_halting=True, return_stats=True)
        else:
            r = {"fast": 1, "normal": 2}.get(mode, lm_config.max_loop) if mode else \
                random.randint(1, lm_config.max_loop)
            r = max(1, min(r, lm_config.max_loop))
            res = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens, do_sample=True, temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id, fixed_loops=r,
                use_halting=False, return_stats=True)
        gen_ids = res["generated_ids"][0, inputs["input_ids"].shape[1]:]
        outs["ids"].append(gen_ids)
        outs["completions"].append(tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True))
        outs["loop"].append(res["stats"]["avg_loops"])  # 固定 R 时严格 == r
        outs["mem"].append(res["stats"].get("memory_steps", 0))
        outs["ret"].append(res["stats"].get("re_retrievals", 0))
    return outs


def compute_per_token_logps(model_, input_ids, n_keep, attention_mask=None, num_loops=1):
    """num_loops 与轨迹实际轮数对齐（否则 off-policy 偏差：生成用自适应 R、评估用 R=1）。"""
    unwrapped = model_.module if isinstance(model_, DistributedDataParallel) else model_
    logits = unwrapped(input_ids, attention_mask=attention_mask, num_loops=num_loops,
                       logits_to_keep=n_keep + 1).logits[:, :-1, :]
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
        rollouts, old_logps, old_loop = [], [], []
        prompt_lens = []
        for p in prompts:
            ro = rollout_group(p, args.num_generations, args.max_gen_len)
            rollouts.append(ro)
        # 拼 batch：每条轨迹独立（变长 → 逐条算 logps，研究级实现）
        traj = []  # (prompt_idx, ids, stats)
        prompt_base_ids = [tokenize_tensors(tokenizer, p, args.device)["input_ids"][0]
                           for p in prompts]
        for i, ro in enumerate(rollouts):
            base = prompt_base_ids[i]
            for j in range(args.num_generations):
                traj.append((i, torch.cat([base, ro["ids"][j].cpu()]).to(args.device), ro))
        old_per_token, logp_lens, traj_loops = [], [], []
        with torch.no_grad():
            for (i, ids, ro) in traj:
                n_keep = ids.numel() - prompt_base_ids[i].numel()
                j = len(old_per_token) % args.num_generations
                r = max(1, int(round(ro["loop"][j])))  # 轨迹实际轮数对齐
                old_per_token.append(compute_per_token_logps(model, ids.unsqueeze(0), n_keep, num_loops=r)[0])
                logp_lens.append(n_keep)
                traj_loops.append(r)

        rewards = torch.zeros(len(traj), device=args.device)
        task_fn = code_task_reward if args.reward_type == "code" else math_task_reward
        for k, (i, ids, ro) in enumerate(traj):
            j = k % args.num_generations
            task = task_fn(ro["completions"][j], gts[i]) + \
                format_reward(ro["completions"][j])
            p_t = ro["loop"][j] / max(lm_config.max_loop, 1)
            p_m = min(ro["mem"][j] / args.max_gen_len, 1.0)
            p_r = ro["ret"][j] / 4.0
            correct = task >= 1.0
            # 防塌缩：效率奖励只给「正确」的轨迹（Minimum reward constraint）
            penalty = (lam_m * p_m + lam_r * p_r + lam_t * p_t) if correct else 0.0
            rewards[k] = task - penalty

        grouped = rewards.view(-1, args.num_generations)
        mean_r = grouped.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # ---- 策略更新（逐轨迹累积，GRPO clip + KL）----
        optimizer.zero_grad(set_to_none=True)
        policy_loss_total, kl_total, n_traj = 0.0, 0.0, len(traj)
        for k, (i, ids, ro) in enumerate(traj):
            n_keep = logp_lens[k]
            per_token_logps = compute_per_token_logps(model, ids.unsqueeze(0), n_keep,
                                                      num_loops=traj_loops[k])[0]
            with torch.no_grad():
                ref_logps = compute_per_token_logps(ref_model, ids.unsqueeze(0), n_keep,
                                                    num_loops=traj_loops[k])[0]
            kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
            ratio = torch.exp(per_token_logps - old_per_token[k].to(args.device))
            clipped = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            a1, a2 = ratio * advantages[k], clipped * advantages[k]
            token_loss = -(torch.min(a1, a2) - args.beta * kl).mean()
            (token_loss / n_traj).backward()
            policy_loss_total += token_loss.item() / n_traj
            kl_total += kl.mean().item() / n_traj
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0 or step == iters:
            acc = sum(1 for k, (i, _, ro) in enumerate(traj)
                      if math_task_reward(ro["completions"][k % args.num_generations], gts[i]) >= 1.0) / len(traj)
            avg_loop = sum(ro["loop"][k % args.num_generations] for k, (_, _, ro) in enumerate(traj)) / len(traj)
            Logger(f'GRPO-X Epoch:[{epoch + 1}]({step}/{iters}), reward: {rewards.mean().item():.4f}, '
                   f'acc: {acc:.3f}, avg_loop: {avg_loop:.2f}, KL: {kl_total:.4f}, '
                   f'policy_loss: {policy_loss_total:.4f}, phase: {args.rl_phase}')
            if wandb:
                wandb.log({"reward": rewards.mean().item(), "accuracy": acc,
                           "avg_loops": avg_loop, "kl": kl_total,
                           "policy_loss": policy_loss_total})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            save_x_checkpoint(model, args.save_dir, args.save_weight, lm_config.hidden_size)
            model.train()
        del traj, rewards, advantages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-X Adaptive Compute GRPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='grpo_x', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2, help="每步 prompt 数")
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_generations", type=int, default=3, help="每 prompt 轨迹数（3 的倍数配模式循环）")
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.05, help="KL 惩罚系数")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rl_phase", type=int, default=1, choices=[1, 2, 3],
                        help="RL 课程：1 无惩罚 / 2 小惩罚 / 3 Pareto（Doc 5 §24）")
    parser.add_argument("--mode_diversity", default=1, type=int, choices=[0, 1],
                        help="轨迹按 fast/normal/deep 模式循环采样")
    parser.add_argument("--adaptive_rollout", default=0, type=int, choices=[0, 1],
                        help="0=固定采样 R（默认，on-policy）；1=adaptive halting rollout"
                             "（logps 用单一 R 近似，残余 off-policy 偏差，消融用）")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"])
    parser.add_argument('--variant', default='minimind_x', type=str)
    parser.add_argument('--hidden_size', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--num_hidden_layers', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl_math.jsonl",
                        help="math RL 数据：{conversations, gt}（官方 agent_rl_math.jsonl）")
    parser.add_argument("--reward_type", type=str, default="math", choices=["math", "code"],
                        help="math=safe_math_eval 判等；code=python 沙箱 stdout 匹配（T7.4）")
    parser.add_argument('--from_weight', default='sft_x_c', type=str)
    parser.add_argument("--tokenizer_path", type=str, default="../model",
                        help="tokenizer 目录（从仓库根目录运行时传 model/）")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-GRPO")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(preset=args.preset, variant=args.variant, max_loop=args.max_loop,
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
    model.train()
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
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)
    iters = len(loader) * args.epochs
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        grpo_x_epoch(epoch, loader, iters, ref_model, 0, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
