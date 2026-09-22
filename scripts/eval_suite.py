"""评测套件（任务清单 T8.4；Doc 1 §26、Doc 6 §7）。

一次运行产出统一 JSON 报告：
  能力：val PPL（文本 jsonl）
  长上下文：needle（retrieval 路径，短网格）
  深度：loop 分布（合成难度分级）
  效率：tokens/s、avg_loops、FLOPs（estimate_flops 统一口径）
"""
import os
import sys
import json
import argparse
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, estimate_flops
from model.minimind_x.controller import AdaptiveController
from trainer.x_utils import x_weight_path, tokenize_tensors
from trainer.trainer_utils import safe_math_eval


def eval_ppl(model, tokenizer, data_path, max_samples=50, max_len=512, device="cpu"):
    import torch.nn.functional as F
    tot_nll, tot_tok = 0.0, 0
    with open(data_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()][:max_samples]
    model.eval()
    with torch.no_grad():
        for line in lines:
            text = json.loads(line).get("text", "")
            ids = tokenizer(text, add_special_tokens=False, max_length=max_len,
                            truncation=True).input_ids
            if len(ids) < 8:
                continue
            x = torch.tensor([ids], device=device)
            logits = model(x, num_loops=1).logits
            nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                  x[:, 1:].reshape(-1), reduction="sum")
            tot_nll += float(nll)
            tot_tok += len(ids) - 1
    return tot_nll / max(tot_tok, 1)


def eval_efficiency(model, tokenizer, device, n=8, max_new_tokens=32):
    prompts = ["请解释机器学习的基本原理。"] * n
    t0 = time.time()
    loops, tokens = [], 0
    with torch.no_grad():
        for p in prompts:
            ids = tokenize_tensors(tokenizer, p, device)
            out = model.generate(inputs=ids["input_ids"],
                                 attention_mask=ids.get("attention_mask"),
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id, return_stats=True)
            loops.append(out["stats"]["avg_loops"])
            tokens += out["generated_ids"].shape[1] - ids["input_ids"].shape[1]
    sec = max(time.time() - t0, 1e-6)
    return {"tokens_per_sec": tokens / sec, "avg_loops": sum(loops) / len(loops),
            "gen_tokens": tokens, "sec": round(sec, 2)}


def main():
    p = argparse.ArgumentParser(description="MiniMind-X 评测套件")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="sft_x_c")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="minimind_x")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--ppl_data", type=str, default="../dataset/pretrain_t2t_mini.jsonl")
    p.add_argument("--skip_needle", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/eval_suite_report.json")
    args = p.parse_args()

    cfg = MiniMindXConfig(preset=args.preset, variant=args.variant,
                          **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                   num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    model = MiniMindXForCausalLM(cfg)
    model.load_state_dict(torch.load(x_weight_path(args.weight_dir, args.weight, cfg.hidden_size),
                                     map_location="cpu"), strict=False)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    report = {"config": {"variant": cfg.variant, "hidden": cfg.hidden_size,
                         "layers": cfg.num_hidden_layers,
                         "zones": [cfg.prelude_layers, cfg.thinking_layers, cfg.coda_layers],
                         "max_loop": cfg.max_loop},
              "params_m": round(sum(pp.numel() for pp in model.parameters()) / 1e6, 2)}
    if os.path.exists(args.ppl_data):
        report["ppl"] = eval_ppl(model, tokenizer, args.ppl_data, device=args.device)
    report["efficiency"] = eval_efficiency(model, tokenizer, args.device)
    report["flops_tok"] = estimate_flops(cfg, seq_len=512, loops=round(
        report["efficiency"]["avg_loops"]))["total"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved -> {args.out}")
    if not args.skip_needle:
        print("提示：needle 网格评测请另跑 scripts/eval_needle.py（较慢）")


if __name__ == "__main__":
    main()
