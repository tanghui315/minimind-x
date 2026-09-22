"""Zone 对比实验（任务清单 T1.5；Doc 3 §6/§13 实验 1-3）。

在等 FLOPs 预算下对比：Baseline(dense R=1) vs Full Loop vs Middle Loop vs Late-Middle。
- 等预算换算：full_loop 的 R_full = ceil(T_mid × R_mid / L)，使有效深度 FLOPs 对齐
- 输出：各 variant 的 val loss、平均 FLOPs/step、结果表 + experiments/ 记录

在进程内跑小规模对比（CPU/单卡均可），大规模请用 train_pretrain_x.py 逐个跑后聚合。
"""
import os
import sys
import json
import argparse
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import optim

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, estimate_flops
from trainer.trainer_utils import Logger

VARIANTS = ["dense", "full_loop", "middle_loop", "late_middle"]


def loops_for_equal_flops(cfg, target_flops_per_token, seq_len):
    """给定目标 backbone FLOPs，求该 variant 的等价 R（dense 固定 1）。"""
    if not cfg.use_loop:
        return 1
    r = 1
    while r < cfg.max_loop:
        f = estimate_flops(cfg, seq_len, loops=r)["total"]
        if f >= target_flops_per_token:
            break
        r += 1
    return r


def run_variant(variant, args, train_tokens, val_tokens):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # ---- 数据：优先真实语料（随机 token 无可学习结构，loss 停在熵上限，分不出 variant）----
    use_real = args.data_path and os.path.exists(args.data_path)
    if use_real:
        from transformers import AutoTokenizer
        from dataset.lm_dataset import PretrainDataset
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        args.vocab_size = len(tokenizer)  # vocab 以 tokenizer 为准（6400），覆盖默认
        ds = PretrainDataset(args.data_path, tokenizer, max_length=args.seq_len)
        rng = random.Random(args.seed)
        print(f"[data] real corpus: {args.data_path} ({len(ds)} samples)")
    else:
        print(f"[warn] {args.data_path} 不存在，退化为随机 token（仅冒烟，无架构信号）")

    def batch():
        if use_real:
            xs, ys = zip(*(ds[rng.randrange(len(ds))] for _ in range(args.batch_size)))
            return torch.stack(xs).to(args.device), torch.stack(ys).to(args.device)
        ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
        return ids.to(args.device), ids.clone().to(args.device)

    kw = dict(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
              variant=variant, max_loop=args.max_loop, flash_attn=False,
              vocab_size=args.vocab_size, loop_embedding=bool(args.loop_embedding))
    cfg = MiniMindXConfig(**kw)
    model = MiniMindXForCausalLM(cfg).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    # 参照预算：middle_loop R=2 的 backbone FLOPs
    ref_cfg = MiniMindXConfig(**dict(kw, variant="middle_loop"))
    budget = estimate_flops(ref_cfg, args.seq_len, loops=2)["total"]
    r = loops_for_equal_flops(cfg, budget, args.seq_len)

    model.train()
    t0, losses = time.time(), []
    for step in range(args.steps):
        ids, labels = batch()
        res = model(ids, labels=labels, num_loops=r)
        loss = res.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(args.val_steps):
            ids, labels = batch()
            val_losses.append(model(ids, labels=labels, num_loops=r).loss.item())
    flops = estimate_flops(cfg, args.seq_len, loops=r, train=True)["total"] * args.steps
    return {"variant": variant, "loops": r, "train_loss": sum(losses) / len(losses),
            "val_loss": sum(val_losses) / len(val_losses), "total_flops": flops,
            "sec": round(time.time() - t0, 1), "data": "real" if use_real else "random"}


def main():
    p = argparse.ArgumentParser(description="Doc 3 实验 1-3：zone 位置对比（等 FLOPs）")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--val_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--max_loop", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))
    p.add_argument("--loop_embedding", type=int, default=0, choices=[0, 1],
                   help="轮次嵌入消融（D4）：对比加/不加 step embedding 的循环收益")
    p.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl",
                   help="真实语料（强烈建议）；缺省时退化为随机 token（仅冒烟）")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda / cpu（等 FLOPs 对比建议 cuda）")
    p.add_argument("--out", type=str, default="experiments/zone_experiments.json")
    args = p.parse_args()

    results = []
    for v in args.variants.split(","):
        Logger(f"===== running variant={v} =====")
        results.append(run_variant(v, args, None, None))
    print("\n==== Zone Experiment Results (equal-FLOPs vs middle_loop R=2) ====")
    print(f"{'variant':<14}{'R':>3}{'train_loss':>12}{'val_loss':>10}{'GFLOPs':>10}{'sec':>7}")
    for r in results:
        print(f"{r['variant']:<14}{r['loops']:>3}{r['train_loss']:>12.4f}"
              f"{r['val_loss']:>10.4f}{r['total_flops'] / 1e9:>10.2f}{r['sec']:>7}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
