"""Needle / 检索式任务评测（任务清单 T3.5；Doc 1 §26 Context Ability）。

三种口径：
  单针     ：--mode retrieval|backbone，深度网格（默认）
  多针     ：--needles K>1（Doc 1 §26 Multi Needle）
  数据集   ：--data xxx.jsonl（直接评 multihop / repo / planning 等检索式数据，
             覆盖 Doc 1 §26 的 Multi-hop QA 与 Long Code）

EM 判定：answer（精确字符串）出现在生成文本中。
"""
import os
import sys
import json
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), "data_gen"))

import torch
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from trainer.x_utils import x_weight_path, tokenize_tensors


def run_one(model, tokenizer, ctrl, args, cfg, sample):
    ctx_ids = tokenizer(sample["context"], add_special_tokens=False).input_ids
    q_ids = tokenizer(sample["question"], add_special_tokens=False).input_ids
    ids = torch.tensor([q_ids], device=args.device)
    if args.mode == "backbone":
        total = (ctx_ids + q_ids)[:cfg.max_position_embeddings - 1]
        ids = torch.tensor([total], device=args.device)
        out = model.generate(inputs=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=args.max_new_tokens, do_sample=False,
                             eos_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True)
        stats = {}
    else:
        out = ctrl.generate_with_context(
            ids, attention_mask=torch.ones_like(ids),
            context_ids=torch.tensor(ctx_ids, device=args.device),
            max_new_tokens=args.max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out["generated_ids"][0][ids.shape[1]:].tolist(),
                                skip_special_tokens=True)
        stats = out["stats"]
    return int(str(sample["answer"]) in resp), stats


def main():
    p = argparse.ArgumentParser(description="needle / multi-needle / 检索式任务评测")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="longctx_x_stage2")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="minimind_x")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--mode", type=str, default="retrieval", choices=["backbone", "retrieval"])
    p.add_argument("--data", type=str, default=None,
                   help="检索式 jsonl（needle/multihop/repo/planning）；给出则忽略深度网格")
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--depths", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--needles", type=int, default=1, help="多针评测的针数（Doc 1 §26）")
    p.add_argument("--n_per_cell", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/needle.json")
    args = p.parse_args()

    cfg = MiniMindXConfig(variant=args.variant, hidden_size=args.hidden_size,
                          num_hidden_layers=args.num_hidden_layers)
    model = MiniMindXForCausalLM(cfg)
    model.load_state_dict(torch.load(x_weight_path(args.weight_dir, args.weight, cfg.hidden_size),
                                     map_location="cpu"), strict=False)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    rng = random.Random(7)
    ctrl = AdaptiveController(model) if args.mode == "retrieval" else None

    cells = []
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()][:args.n_per_cell]
        correct, agg = 0, {"avg_loops": 0.0, "retrievals": 0, "memory_tokens": 0, "n": 0}
        for s in samples:
            hit, stats = run_one(model, tokenizer, ctrl, args, cfg, s)
            correct += hit
            if stats:
                agg["avg_loops"] += stats.get("avg_loops", 0)
                agg["retrievals"] += stats.get("retrievals", 0)
                agg["memory_tokens"] += stats.get("memory_tokens_final", 0)
                agg["n"] += 1
        n = max(len(samples), 1)
        row = {"cell": os.path.basename(args.data), "n": n, "acc": correct / n}
        if agg["n"]:
            row.update({k: agg[k] / agg["n"] for k in ("avg_loops", "retrievals", "memory_tokens")})
        cells.append(row)
        print(f"[{row['cell']}] n={n} acc={row['acc']:.2f}" +
              (f" loops={row.get('avg_loops', 0):.2f} retr={row.get('retrievals', 0):.1f}" if agg["n"] else ""))
    else:
        from gen_needle import gen_sample
        depths_list = [float(x) for x in args.depths.split(",")]
        for depth in depths_list:
            correct = 0
            for _ in range(args.n_per_cell):
                s = (gen_sample(rng, args.ctx_tokens, depths_list, args.needles, cfg.block_size)
                     if args.needles > 1 else
                     gen_sample(rng, args.ctx_tokens, [depth], 1, cfg.block_size))
                hit, _ = run_one(model, tokenizer, ctrl, args, cfg, s)
                correct += hit
            acc = correct / args.n_per_cell
            cells.append({"cell": f"depth={depth:.1f}", "depth": depth,
                          "needles": args.needles, "acc": acc})
            print(f"depth={depth:.1f} needles={args.needles}  acc={acc:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "ctx_tokens": args.ctx_tokens, "cells": cells,
                   "config": vars(args)}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
