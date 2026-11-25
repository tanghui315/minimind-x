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

from trainer.x_utils import bind_x_tokenizer, load_x_for_eval, tokenize_tensors
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


@torch.no_grad()
def token_decisions(model, prompt_ids, target_ids):
    """Compare a routed token to fixed exits with the identical gold prefix and routed KV history."""
    cfg = model.config
    past, chunk = None, prompt_ids
    counts = dict(tokens=0, solvable=0, matched_earliest=0, routed_correct=0)
    for t, target in enumerate(target_ids):
        first = None
        cap = cfg.max_loop if cfg.use_loop else 1
        for r in range(1, cap + 1):
            fixed = model(chunk, past_key_values=past, num_loops=r, logits_to_keep=1)
            if first is None and int(fixed.logits[0, -1].argmax()) == target:
                first = r
        halt = t > 0 or getattr(cfg, "prefill_policy", "fixed_cap") != "fixed_cap"
        routed = model(chunk, past_key_values=past, num_loops=cap, halt=halt, use_cache=True,
                       logits_to_keep=1)
        counts["tokens"] += 1
        counts["solvable"] += int(first is not None)
        counts["matched_earliest"] += int(first == routed.num_loops_executed)
        counts["routed_correct"] += int(int(routed.logits[0, -1].argmax()) == target)
        past, chunk = routed.past_key_values, chunk.new_tensor([[target]])
    return counts


def main():
    p = argparse.ArgumentParser(description="动态深度评测：loop 分布 vs 难度")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="sft_x_b")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="middle_loop")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--num_hidden_layers", type=int, default=None)
    p.add_argument("--max_loop", type=int, default=4)
    p.add_argument("--n_per_difficulty", type=int, default=20)
    p.add_argument("--difficulties", type=str, default="1,2,3,4,5")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/depth_dist.json")
    args = p.parse_args()

    model = load_x_for_eval(args.weight_dir, args.weight, device=args.device,
                            hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                            variant=args.variant, preset=args.preset, max_loop=args.max_loop)
    cfg = model.config
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bind_x_tokenizer(model, tokenizer)

    rng = random.Random(0)
    report = {"per_difficulty": [], "config": vars(args)}
    for d in map(int, args.difficulties.split(",")):
        fixed_hits = [0] * (cfg.max_loop if cfg.use_loop else 1)
        first_correct, halt_rs, correct_halt = [], [], 0
        token_counts = dict(tokens=0, solvable=0, matched_earliest=0, routed_correct=0)
        for _ in range(args.n_per_difficulty):
            prompt, gt = make_arith_prompt(rng, d)
            ids = tokenize_tensors(tokenizer, prompt, args.device)
            prompt_ids = ids["input_ids"]
            generation = dict(inputs=prompt_ids, attention_mask=ids["attention_mask"],
                              max_new_tokens=args.max_new_tokens, do_sample=False,
                              eos_token_id=tokenizer.eos_token_id)
            solved = []
            for r in range(1, len(fixed_hits) + 1):
                result = model.generate(**generation, fixed_loops=r)
                hit = correctness(tokenizer.decode(result[0, prompt_ids.size(1):].tolist(), skip_special_tokens=True), gt)
                fixed_hits[r-1] += int(hit)
                if hit:
                    solved.append(r)
            if solved:
                first_correct.append(solved[0])
            out = model.generate(**generation, return_stats=True)
            resp = tokenizer.decode(out["generated_ids"][0, prompt_ids.size(1):].tolist(), skip_special_tokens=True)
            halt_rs.append(out["stats"]["avg_loops"])
            correct_halt += int(correctness(resp, gt))
            target_ids = tokenizer(str(int(gt)), add_special_tokens=False).input_ids
            for field, value in token_decisions(model, prompt_ids, target_ids).items():
                token_counts[field] += value
        n = args.n_per_difficulty
        row = dict(difficulty=d, n=n, fixed_correct=fixed_hits, acc_fixed_by_R={
                   str(r+1): count / n for r, count in enumerate(fixed_hits)},
                   solvable=len(first_correct), unresolved=n-len(first_correct),
                   oracle_upper_bound=len(first_correct)/n, acc_halt=correct_halt/n,
                   halt_mean_R=sum(halt_rs)/n, oracle_hist={str(r): first_correct.count(r) for r in set(first_correct)},
                   token_decisions=token_counts, token_stop_accuracy=token_counts["matched_earliest"] /
                   token_counts["solvable"] if token_counts["solvable"] else None)
        report["per_difficulty"].append(row)
        print(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
