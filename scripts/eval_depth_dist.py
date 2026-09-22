"""Adaptive Depth 评测（任务清单 T2.5；Doc 3 §14 router 指标）。

合成难度分级的算术题 → 逐难度统计：
  - oracle R*（固定 R 逐档生成取首个 correct，Doc 3 §12）
  - halting 决策的 loop 分布（Doc 6 §10 Dynamic Depth 曲线）
  - stop accuracy（决策轮 == oracle 轮）
"""
import os
import sys
import json
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from trainer.x_utils import x_weight_path, tokenize_tensors
from trainer.trainer_utils import safe_math_eval


def make_arith_prompt(rng, difficulty):
    """难度 1~5：运算步数与数值规模递增。返回 (prompt, answer)。"""
    n_terms = difficulty + 1
    terms = [rng.randint(2, 9 + difficulty * 10) for _ in range(n_terms)]
    ops = [rng.choice("+-*") for _ in range(n_terms - 1)]
    expr = str(terms[0])
    for op, t in zip(ops, terms[1:]):
        expr += f" {op} {t}"
    value = safe_math_eval(expr)
    prompt = f"计算：{expr} = ?"
    return prompt, value


def correctness(response, gt):
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", response)
    if not nums:
        return False
    try:
        return abs(float(nums[-1]) - float(gt)) < 1e-6
    except ValueError:
        return False


def main():
    p = argparse.ArgumentParser(description="动态深度评测：loop 分布 vs 难度")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="sft_x_b")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="middle_loop")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--max_loop", type=int, default=4)
    p.add_argument("--n_per_difficulty", type=int, default=20)
    p.add_argument("--difficulties", type=str, default="1,2,3,4,5")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/depth_dist.json")
    args = p.parse_args()

    cfg_kw = dict(variant=args.variant, max_loop=args.max_loop, preset=args.preset,
                  **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                           num_hidden_layers=args.num_hidden_layers).items() if v or k == "variant"})
    cfg = MiniMindXConfig(**{k: v for k, v in cfg_kw.items() if v is not None})
    model = MiniMindXForCausalLM(cfg)
    path = x_weight_path(args.weight_dir, args.weight, cfg.hidden_size)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    rng = random.Random(0)
    report = {"per_difficulty": [], "config": vars(args)}
    print(f"{'diff':>5}{'acc':>7}{'oracle_R':>10}{'halting_R':>10}{'stop_acc':>9}")
    for d in [int(x) for x in args.difficulties.split(",")]:
        oracle_rs, halt_rs, correct_fixed, correct_halt = [], [], 0, 0
        for _ in range(args.n_per_difficulty):
            prompt, gt = make_arith_prompt(rng, d)
            ids = tokenize_tensors(tokenizer, prompt, args.device)
            # oracle：固定 R 逐档，取首个 correct
            oracle = cfg.max_loop
            for r in range(1, cfg.max_loop + 1):
                out = model.generate(inputs=ids["input_ids"], attention_mask=ids["attention_mask"],
                                     max_new_tokens=args.max_new_tokens, do_sample=False,
                                     eos_token_id=tokenizer.eos_token_id, fixed_loops=r,
                                     use_halting=False)
                resp = tokenizer.decode(out[0][ids["input_ids"].shape[1]:].tolist(),
                                        skip_special_tokens=True)
                if correctness(resp, gt):
                    oracle = r
                    correct_fixed += 1
                    break
            oracle_rs.append(oracle)
            # halting 决策
            out = model.generate(inputs=ids["input_ids"], attention_mask=ids["attention_mask"],
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 use_halting=True, return_stats=True)
            resp = tokenizer.decode(out["generated_ids"][0][ids["input_ids"].shape[1]:].tolist(),
                                    skip_special_tokens=True)
            halt_rs.append(out["stats"]["avg_loops"])
            correct_halt += int(correctness(resp, gt))
        n = args.n_per_difficulty
        stop_acc = sum(1 for o, h in zip(oracle_rs, halt_rs) if abs(o - h) < 0.5) / n
        row = {"difficulty": d, "acc_fixed": correct_fixed / n, "acc_halt": correct_halt / n,
               "oracle_mean_R": sum(oracle_rs) / n,
               "halt_mean_R": sum(halt_rs) / n, "stop_accuracy": stop_acc,
               "oracle_hist": {str(r): oracle_rs.count(r) for r in set(oracle_rs)},
               "halt_hist": {str(r): halt_rs.count(r) for r in set(halt_rs)}}
        report["per_difficulty"].append(row)
        print(f"{d:>5}{row['acc_fixed']:>7.2f}{row['oracle_mean_R']:>10.2f}"
              f"{row['halt_mean_R']:>10.2f}{stop_acc:>9.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
