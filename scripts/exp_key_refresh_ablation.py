"""Compare a stale index and periodic rebuilds under identical seeded embedding perturbations.

Reports coarse recall, RMS key drift relative to a fresh index, and measured rebuild time.
"""
import os
import sys
import json
import copy
import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController


def recall_at_k(store, model, queries, gold_blocks, ks=(1, 8)):
    """粗检 Recall@K（Frozen Index 评估口径）。"""
    retriever = model.model.retriever
    hits = {k: 0 for k in ks}
    n = 0
    for q, g in zip(queries, gold_blocks):
        if g is None or int(g) < 0:
            continue
        n += 1
        qv = retriever.project_query(q)
        scores = retriever.coarse_scores(qv, store)
        rank = scores[0].argsort(descending=True)
        pos = int((rank == int(g)).nonzero().flatten().item())
        for k in ks:
            if pos < k:
                hits[k] += 1
    return {f"recall@{k}": hits[k] / max(n, 1) for k in ks} | {"n": n}


def run_arm(name, refresh_interval, model, ctx, gold_blocks, query_hiddens, steps=3):
    model = copy.deepcopy(model)
    model.config.key_refresh_interval = refresh_interval
    ctrl = AdaptiveController(model)
    ctrl.begin(ctx)
    # Identical deterministic embedding drift in both arms. The index remains request-local in training.
    rng = torch.Generator(device=ctx.device).manual_seed(73)
    elapsed = 0.0
    for step in range(1, steps + 1):
        with torch.no_grad():
            weight = model.model.embed_tokens.weight
            weight.add_(torch.randn(weight.shape, generator=rng, device=weight.device) * .01)
        start = time.perf_counter()
        if ctrl.maybe_refresh_index(step):
            elapsed += time.perf_counter() - start
    fresh = AdaptiveController(model)
    fresh.begin(ctx)
    r = recall_at_k(ctrl.store, model, query_hiddens, gold_blocks)
    r["key_rms_drift"] = float((fresh.store.block_keys - ctrl.store.block_keys).square().mean().sqrt())
    r["rebuild_seconds"] = elapsed
    r["drift_source"] = "seeded_embedding_perturbation"
    r["arm"] = name
    r["key_refresh_interval"] = refresh_interval
    r["index_manifest"] = ctrl.store.metadata
    r["index_refreshes"] = ctrl.stats.get("index_refreshes", 0)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/key_refresh_ablation.jsonl")
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--n_queries", type=int, default=8)
    p.add_argument("--refresh_interval", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(0)
    cfg = MiniMindXConfig(hidden_size=args.hidden_size, num_hidden_layers=6,
                          num_attention_heads=2, num_key_value_heads=1, vocab_size=64,
                          flash_attn=False, variant="minimind_x", block_size=8,
                          topk_coarse=4, topk_fine=2, freeze_key=True,
                          key_refresh_interval=args.refresh_interval)
    model = MiniMindXForCausalLM(cfg).eval()
    ctx = torch.randint(3, 60, (8 * 6,))
    golds = list(range(6))
    qs = [model.model.embed_tokens(torch.full((1, 4), 3 + i)) for i in range(args.n_queries)]

    results = [
        run_arm("A_freeze_forever", 0, model, ctx, golds + [None] * 2, qs),
        run_arm("B_refresh_every_N", args.refresh_interval, model, ctx,
                golds + [None] * 2, qs, steps=3),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in results:
        print(r["arm"], {k: r[k] for k in r if k.startswith("recall")}, 
              "refreshes=", r["index_refreshes"])


if __name__ == "__main__":
    main()
