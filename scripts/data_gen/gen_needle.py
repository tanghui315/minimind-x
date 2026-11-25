"""Needle 数据生成器（任务清单 T3.1；Doc 4 §7.3、Doc 5 §8）。

产出检索式 jsonl：{context, question, answer, gold_block}。
- 单针/多针（--needles 1~8）
- 深度网格：needle 插入位置比例从 --depths 取值（0.1~0.9 对应 10K/100K/500K/1M 的相对深度）
- gold_block：needle 所在块号（按完整上下文的 tokenizer offsets 定位）
"""
import os
import sys
import json
import random
import argparse
from functools import lru_cache
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataset.long_context_dataset import FILLER_WORDS
from dataset.retrieval_qa_dataset import token_span


def filler_sentences(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        sent = "".join(rng.choice(FILLER_WORDS) for _ in range(rng.randint(8, 20))) + "。"
        parts.append(sent)
        total += len(sent)
    return parts


@lru_cache(maxsize=1)
def default_tokenizer():
    return AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), "../../model"))


def gen_sample(rng, ctx_tokens, depths, n_needles, block_size, tokenizer=None, target_needle=None):
    tokenizer = tokenizer or default_tokenizer()
    if not depths or not all(0 <= d <= 1 for d in depths) or min(n_needles, block_size, ctx_tokens) < 1:
        raise ValueError("invalid needle count, length, block size or depth")
    num = rng.randint(1000, 9999)
    needles = [f"重要提示：第{i + 1}条记录的编号是 {num + i}。" for i in range(n_needles)]
    # Reserve space for every needle, including ones placed at the very end.
    budget = ctx_tokens - sum(len(tokenizer(n, add_special_tokens=False).input_ids) for n in needles)
    if budget < 0:
        raise ValueError("ctx_tokens is smaller than the requested needles")
    filler = ""
    offsets = []
    while len(offsets) < budget + 8:
        filler += "".join(filler_sentences(rng, max(32, 2 * (budget + 8 - len(offsets)))))
        offsets = tokenizer(filler, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    insertions = sorted((int(depths[i % len(depths)] * budget), i) for i in range(n_needles))
    parts, previous = [], 0
    for pos, i in insertions:
        char_pos = offsets[pos][0]
        parts.extend((filler[previous:char_pos], needles[i]))
        previous = char_pos
    parts.append(filler[previous:offsets[budget][0]])
    context = "".join(parts)
    def encode(text):
        return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    encoded = encode(context)
    # Joining fragments can retokenize a boundary. Correct the length using only filler.
    while len(encoded.input_ids) > ctx_tokens:
        starts = sorted((context.index(n), len(n)) for n in needles)
        gaps = list(zip([0] + [a + size for a, size in starts], [a for a, _ in starts] + [len(context)]))
        a, b = max(gaps, key=lambda gap: gap[1] - gap[0])
        if a == b:
            raise ValueError("ctx_tokens is smaller than the assembled needles")
        trim = min(b - a, len(encoded.input_ids) - ctx_tokens)
        context = context[:b-trim] + context[b:]
        encoded = encode(context)
    while len(encoded.input_ids) < ctx_tokens:
        for suffix in ("。", "甲", "1", " A"):
            candidate = encode(context + suffix)
            if len(encoded.input_ids) < len(candidate.input_ids) <= ctx_tokens:
                context += suffix
                encoded = candidate
                break
        else:
            raise ValueError("tokenizer cannot fill the final token budget")
    spans = [(context.index(n), context.index(n) + len(n)) for n in needles]
    token_ranges = [token_span(encoded.offset_mapping, span) for span in spans]
    gold_blocks = [(a + b - 1) // 2 // block_size for a, b in token_ranges]
    q = rng.randrange(n_needles) if target_needle is None else target_needle
    return {"context": context, "question": f"第{q + 1}条记录的编号是多少？",
            "answer": str(num + q), "gold_span": spans[q], "gold_block": gold_blocks[q],
            "gold_blocks": gold_blocks, "needles": needles}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 needle 检索式训练/评测数据")
    p.add_argument("--tokenizer_path", default=os.path.join(os.path.dirname(__file__), "../../model"))
    p.add_argument("--out", type=str, required=True, help="输出 jsonl 路径")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--needles", type=int, default=1, help="每条样本针数（1=单针）")
    p.add_argument("--depths", type=str, default="0.1,0.3,0.5,0.7,0.9",
                   help="插入深度比例网格（逗号分隔）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    rng = random.Random(args.seed)
    depths = [float(x) for x in args.depths.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.ctx_tokens, depths,
                                          args.needles, args.block_size, tokenizer), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
