"""评测套件（任务清单 T8.4；Doc 1 §26、Doc 6 §7）。

一次运行产出统一 JSON 报告：
  能力：val PPL（文本 jsonl）
  长上下文：needle（retrieval 路径，短网格）
  深度：loop 分布（合成难度分级）
  效率：tokens/s、avg_loops、FLOPs（estimate_flops 统一口径）
"""
import os
import sys
import math
import json
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer

from model.minimind_x import estimate_flops
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval, tokenize_tensors


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
    if not tot_tok:
        raise ValueError("PPL requires at least one scored token")
    return math.exp(tot_nll / tot_tok)


def eval_efficiency(model, tokenizer, device, n=8, max_new_tokens=32):
    loops, tokens = [], 0
    prefill_seconds, decode_seconds = [], []
    prefill_flops = decode_flops = 0
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    def timestamp():
        if cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    started = timestamp()
    with torch.no_grad():
        for _ in range(n):
            ids = tokenize_tensors(tokenizer, "请解释机器学习的基本原理。", device)
            last = timestamp()
            def observe(size, past, rounds):
                nonlocal last, prefill_flops, decode_flops
                now = timestamp()
                cost = estimate_flops(model.config, size, rounds, past_tokens=past)["total"]
                if past:
                    decode_seconds.append(now - last)
                    decode_flops += cost
                else:
                    prefill_seconds.append(now - last)
                    prefill_flops += cost
                last = now
            out = model.generate(inputs=ids["input_ids"], attention_mask=ids.get("attention_mask"),
                                 max_new_tokens=max_new_tokens, do_sample=False, step_observer=observe,
                                 eos_token_id=tokenizer.eos_token_id, return_stats=True)
            loops.append(out["stats"]["loop_sum"])
            tokens += out["stats"]["steps"]
    sec = timestamp() - started
    return {"tokens_per_sec": tokens / sec, "avg_loops": sum(loops) / tokens,
            "gen_tokens": tokens, "sec": sec, "requests": len(prefill_seconds),
            "decode_tokens": len(decode_seconds), "ttft_ms": 1000 * sum(prefill_seconds) / n,
            "tpot_ms": 1000 * sum(decode_seconds) / len(decode_seconds) if decode_seconds else None,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
            "estimated_prefill_flops": prefill_flops, "estimated_decode_flops": decode_flops}


@torch.no_grad()
def profile_request(model, tokenizer, device, document, max_new_tokens=4):
    """Measure a single cold request; profiler FLOPs cover only supported operator families."""
    from model.minimind_x.controller import AdaptiveController
    from torch.profiler import profile, ProfilerActivity
    from contextlib import ExitStack
    cfg = model.config
    ctrl = AdaptiveController(model) if document is not None else None
    prompt = tokenize_tensors(tokenizer, "资料中的阈值是多少？", device)["input_ids"]
    ctx = torch.tensor(tokenizer(document, add_special_tokens=False).input_ids, device=device,
                       dtype=torch.long) if ctrl else None
    activities = [ProfilerActivity.CPU]
    cuda = torch.device(device).type == "cuda"
    if cuda:
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    reader = getattr(model.model.thinking_core, "memory_reader", None)
    reads, projections, legacy_reads, chunks = [], [], [], []
    def reading(_, inputs):
        h, kv = inputs
        reads.append((h.shape[1], kv.valid.size(1)))
    def legacy_reading(_, inputs, kwargs):
        mem = kwargs.get("memory_kv")
        if mem is not None:
            legacy_reads.append(inputs[0].shape[1] * mem[0].shape[1])

    def projecting(_, inputs):
        projections.append(inputs[0].shape[1])
    def observe(size, past, rounds):
        if cuda:
            torch.cuda.synchronize(device)
        chunks.append((size, past, rounds, time.perf_counter()))
    with ExitStack() as stack:
        if reader is not None:
            stack.enter_context(reader.register_forward_pre_hook(reading))
            stack.enter_context(reader.k_proj.register_forward_pre_hook(projecting))
        elif ctrl is not None:
            for layer in model.model.thinking_core.layers:
                stack.enter_context(layer.self_attn.register_forward_pre_hook(legacy_reading, with_kwargs=True))
        with profile(activities=activities, with_flops=True) as prof:
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            opts = dict(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id,
                        step_observer=observe, return_stats=True)
            if ctrl:
                out = ctrl.generate_with_context(prompt, context_ids=ctx, **opts)
            else:
                out = model.generate(inputs=prompt, **opts)
    stats = out["stats"]
    estimate = [estimate_flops(cfg, size, rounds, past_tokens=past)["total"] for size, past, rounds, _ in chunks]
    reader_flops = 0
    for size, m in reads:
        f = estimate_flops(cfg, size, 1, memory_blocks=m / cfg.block_size, memory_projections=0)
        reader_flops += f["reader_attn"] + f["reader_gate"] + f["reader_proj"]
    qh = cfg.reader_query_heads or max(2, cfg.hidden_size // 64)
    hd = max(8, min(cfg.reader_head_dim, cfg.hidden_size // qh))
    reader_flops += sum(projections) * 4 * cfg.hidden_size * cfg.reader_kv_heads * hd
    extra = estimate_flops(cfg, 1, indexed_blocks=stats.get("indexed_blocks", 0),
                           encoded_blocks=stats.get("encoded_blocks", 0))
    search = (stats.get("scanned_blocks", 0) + stats.get("scored_blocks", 0) * cfg.block_size) * 2 * cfg.hidden_size
    legacy = (sum(legacy_reads) * 4 * cfg.num_attention_heads * cfg.head_dim +
              stats.get("legacy_projected_tokens", 0) * 4 * cfg.thinking_layers * cfg.hidden_size *
              cfg.num_key_value_heads * cfg.head_dim)
    components = dict(prefill_backbone_and_residual=estimate[0], decode_backbone_and_residual=sum(estimate[1:]),
                      index_build=extra["index_build"], candidate_encoding=extra["candidate_encoding"],
                      retrieval_dot_products=search, reader=reader_flops, legacy_memory=legacy)
    return dict(cold_request=True, generated_tokens=len(chunks), ttft_ms=1000*(chunks[0][3] - start),
                tpot_ms=1000*(chunks[-1][3]-chunks[0][3])/(len(chunks)-1) if len(chunks)>1 else None,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
                profiled_operator_flops=sum(e.flops or 0 for e in prof.key_averages()),
                estimated_components=components, reader_calls=len(reads), kv_projection_calls=len(projections),
                stats=stats)


def main():
    p = argparse.ArgumentParser(description="MiniMind-X 评测套件")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="sft_x_c")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="minimind_x")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--num_hidden_layers", type=int, default=None)
    p.add_argument("--ppl_data", type=str, default="../dataset/pretrain_t2t_mini.jsonl")
    p.add_argument("--skip_needle", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/eval_suite_report.json")
    p.add_argument("--profile", action="store_true", help="Profile one cold request separately from throughput")
    p.add_argument("--profile_context", help="External text document for the profiled request")
    args = p.parse_args()

    model = load_x_for_eval(args.weight_dir, args.weight, device=args.device,
                            hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                            variant=args.variant, preset=args.preset)
    cfg = model.config
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bind_x_tokenizer(model, tokenizer)

    report = {"config": {"architecture_version": cfg.architecture_version(), "variant": cfg.variant, "hidden": cfg.hidden_size,
                         "layers": cfg.num_hidden_layers,
                         "zones": [cfg.prelude_layers, cfg.thinking_layers, cfg.coda_layers],
                         "max_loop": cfg.max_loop},
              "params_m": round(sum(pp.numel() for pp in model.parameters()) / 1e6, 2)}
    if os.path.exists(args.ppl_data):
        report["ppl"] = eval_ppl(model, tokenizer, args.ppl_data, device=args.device)
    report["efficiency"] = eval_efficiency(model, tokenizer, args.device)
    efficiency = report["efficiency"]
    report["flops_tok"] = (efficiency["estimated_prefill_flops"] + efficiency["estimated_decode_flops"]) / efficiency["gen_tokens"]
    if args.profile or args.profile_context:
        document = None
        if args.profile_context:
            with open(args.profile_context, encoding="utf8") as source:
                document = source.read()
        report["profile"] = profile_request(model, tokenizer, args.device, document)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved -> {args.out}")
    if not args.skip_needle:
        print("提示：needle 网格评测请另跑 scripts/eval_needle.py（较慢）")


if __name__ == "__main__":
    main()
