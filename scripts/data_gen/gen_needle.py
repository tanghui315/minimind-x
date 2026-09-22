"""Needle 数据生成器（任务清单 T3.1；Doc 4 §7.3、Doc 5 §8）。

产出检索式 jsonl：{context, question, answer, gold_block}。
- 单针/多针（--needles 1~8）
- 深度网格：needle 插入位置比例从 --depths 取值（0.1~0.9 对应 10K/100K/500K/1M 的相对深度）
- gold_block：needle 所在块号（按字符比例折算，训练/评测时以实际 tokenize 为准）
"""
import os
import sys
import json
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataset.long_context_dataset import FILLER_WORDS


def filler_sentences(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        sent = "".join(rng.choice(FILLER_WORDS) for _ in range(rng.randint(8, 20))) + "。"
        parts.append(sent)
        total += len(sent)
    return parts


def gen_sample(rng, ctx_tokens, depths, n_needles, block_size):
    num = rng.randint(1000, 9999)
    needles = [f"重要提示：第{i + 1}条记录的编号是 {num + i}。" for i in range(n_needles)]
    target_chars = int(ctx_tokens * 1.5)
    parts = filler_sentences(rng, target_chars)
    n_blocks = max(1, round(ctx_tokens / block_size))
    # 多针：针数与深度网格 zip（避免越界）；单针：取第一个深度
    pairs = list(zip(depths, needles)) if n_needles > 1 else [(depths[0], needles[0])]
    gold_blocks = []
    for d, needle in pairs:
        pos = max(1, int(len(parts) * d))
        parts.insert(pos, needle)
        gold_blocks.append(min(int(d * n_blocks), n_blocks - 1))
    q_idx = rng.randrange(len(pairs))  # 只在实际插入的针里提问（深度数可小于针数）
    question = f"第{q_idx + 1}条记录的编号是多少？"
    answer = str(num + q_idx)
    return {"context": "".join(parts), "question": question, "answer": answer,
            "gold_block": gold_blocks[q_idx], "gold_blocks": gold_blocks, "needles": needles}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 needle 检索式训练/评测数据")
    p.add_argument("--out", type=str, required=True, help="输出 jsonl 路径")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--needles", type=int, default=1, help="每条样本针数（1=单针）")
    p.add_argument("--depths", type=str, default="0.1,0.3,0.5,0.7,0.9",
                   help="插入深度比例网格（逗号分隔）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    depths = [float(x) for x in args.depths.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.ctx_tokens, depths,
                                          args.needles, args.block_size), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
