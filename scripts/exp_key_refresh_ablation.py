"""P3 消融：Frozen Key vs Periodic Refresh vs（可选）联训预备。

对比三组（同数据/同步数）：
  A freeze_forever     key_refresh_interval=0
  B refresh_every_N    key_refresh_interval=N
  C refresh_adaptive   N 较大 + 监控 recall 漂移（简化：仍按步数）

输出：Recall@1/@8 与（若有 QA 标签）answer match，写 jsonl。
不启动长训——用小模型 + 合成 evidence 作冒烟/趋势观察。
"""
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.retrieval import LongContextStore


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
    ctrl = AdaptiveController(model)
    # 按 arm 覆盖 interval（勿用全局 cfg 串味）
    prev = getattr(model.config, "key_refresh_interval", 0)
    model.config.key_refresh_interval = int(refresh_interval)
    try:
        ctrl.begin(ctx)
        if refresh_interval > 0:
            for s in range(1, steps + 1):
                ctrl.maybe_refresh_index(s * max(1, refresh_interval))
        else:
            ctrl.maybe_refresh_index(1)  # interval=0 → 不 refresh
    finally:
        model.config.key_refresh_interval = prev
    r = recall_at_k(ctrl.store, model, query_hiddens, gold_blocks)
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
