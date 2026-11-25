"""Needle / 检索式任务评测（任务清单 T3.5；Doc 1 §26 Context Ability）。

三种口径：
  单针     ：--mode retrieval|backbone，深度网格（默认）
  多针     ：--needles K>1（Doc 1 §26 Multi Needle）
  数据集   ：--data xxx.jsonl（直接评 multihop / repo / planning 等检索式数据，
             覆盖 Doc 1 §26 的 Multi-hop QA 与 Long Code）

EM 判定：Unicode 规范化并整理空白后比较整条答案。
"""
import os
import sys
import json
import argparse
import random
import unicodedata
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), "data_gen"))

import torch
from transformers import AutoTokenizer

from dataset.retrieval_qa_dataset import encode_evidence
from model.minimind_x.controller import AdaptiveController
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval, tokenize_tensors


def exact_match(prediction, answer):
    def normalize(value):
        return " ".join(unicodedata.normalize("NFKC", str(value)).split())
    return normalize(prediction) == normalize(answer)


def run_one(model, tokenizer, args, cfg, sample):
    ctx_ids, gold, visibility = encode_evidence(sample, tokenizer, cfg.block_size)
    q_ids = tokenizer(sample["question"], add_special_tokens=False).input_ids
    ids = torch.tensor([q_ids], device=args.device)
    if args.mode == "backbone":
        if visibility.any():
            raise ValueError("delayed evidence requires retrieval mode")
        available = cfg.max_position_embeddings - args.max_new_tokens - len(q_ids)
        if available < 0:
            raise ValueError("question and generation budget exceed the backbone window")
        total = ctx_ids[:available].tolist() + q_ids
        ids = torch.tensor([total], device=args.device)
        out = model.generate(inputs=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=args.max_new_tokens, do_sample=False,
                             eos_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True)
        stats = {}
    else:
        ctrl = AdaptiveController(model, block_visible_from=visibility)
        out = ctrl.generate_with_context(
            ids, attention_mask=torch.ones_like(ids),
            context_ids=ctx_ids.to(args.device),
            max_new_tokens=args.max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out["generated_ids"][0][ids.shape[1]:].tolist(),
                                skip_special_tokens=True)
        stats = out["stats"]
    stats["context_tokens"] = len(ctx_ids)
    stats["evaluated_context_tokens"] = min(len(ctx_ids), available) if args.mode == "backbone" else len(ctx_ids)
    stats["gold_block"] = gold
    stats["evidence_depth"] = None
    if gold >= 0 and len(ctx_ids):
        from dataset.retrieval_qa_dataset import token_span
        span = sample.get("gold_span")
        position = min(len(ctx_ids) - 1, (gold + 0.5) * cfg.block_size)
        if span and span[0] >= 0:
            offsets = tokenizer(sample["context"], add_special_tokens=False, return_offsets_mapping=True).offset_mapping
            bounds = token_span(offsets, span)
            if bounds:
                position = (bounds[0] + bounds[1] - 1) / 2
        stats["evidence_depth"] = position / len(ctx_ids)
    return int(exact_match(resp, sample["answer"])), stats


def main():
    p = argparse.ArgumentParser(description="needle / multi-needle / 检索式任务评测")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="longctx_x_stage2")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="minimind_x")
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--num_hidden_layers", type=int, default=None)
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

    model = load_x_for_eval(args.weight_dir, args.weight, device=args.device,
                            hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                            variant=args.variant)
    cfg = model.config
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bind_x_tokenizer(model, tokenizer)
    rng = random.Random(7)

    groups = defaultdict(list)
    if args.data:
        def examples():
            with open(args.data, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        if sample.get("split", "val") == "val":
                            yield "data", sample
    else:
        from gen_needle import gen_sample
        def examples():
            for depth in map(float, args.depths.split(",")):
                for _ in range(args.n_per_cell):
                    positions = [depth] + [rng.random() for _ in range(args.needles - 1)]
                    yield depth, gen_sample(rng, args.ctx_tokens, positions, args.needles,
                                            cfg.block_size, tokenizer, target_needle=0)
    for cell, sample in examples():
        hit, stats = run_one(model, tokenizer, args, cfg, sample)
        actual = stats["evidence_depth"]
        depth_bin = min(9, int(actual * 10)) if actual is not None else None
        length = stats["context_tokens"]
        key = (cell, sample.get("family", "needle" if not args.data else "unspecified"),
               length.bit_length(), depth_bin)
        groups[key].append((hit, stats))
    if not groups:
        raise ValueError("No evaluation samples in the requested file")
    cells = []
    for (cell, family, bits, depth_bin), rows in groups.items():
        hits = sum(hit for hit, _ in rows)
        reported = {"cell": cell, "family": family, "min_context_tokens": min(s["context_tokens"] for _, s in rows),
                    "max_context_tokens": max(s["context_tokens"] for _, s in rows),
                    "depth_decile": depth_bin, "n": len(rows), "correct": hits, "acc": hits / len(rows),
                    "truncated_contexts": sum(s["evaluated_context_tokens"] < s["context_tokens"] for _, s in rows)}
        for metric in ("avg_loops", "retrievals", "actual_valid_tokens"):
            if metric in rows[0][1]:
                reported[metric] = sum(s[metric] for _, s in rows) / len(rows)
        cells.append(reported)
        print(reported)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "ctx_tokens": args.ctx_tokens, "cells": cells,
                   "config": vars(args)}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
