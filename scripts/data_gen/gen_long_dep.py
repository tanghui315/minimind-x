"""长依赖合成数据（任务清单 T3.1；Doc 4 §7.2）。

前部定义变量/规则，远端（约 depth 比例处）要求使用 —— 训练长距离回看。
产出 pretrain jsonl：{"text": ...}（配合 LongContextPretrainDataset packing）。
"""
import os
import json
import random
import argparse

WORDS = ("变量 函数 模块 参数 结构 配置 状态 结果 记录 序号 编号 数值 阈值 上限 标识").split()


def filler(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(10, 25))) + " 。"
        parts.append(sent)
        total += len(sent)
    return parts


def gen_sample(rng, target_tokens, depth):
    name = f"X{rng.randint(100, 999)}"
    val = rng.randint(1000, 999999)
    definition = f"定义：{name} 的初始值为 {val}，后续所有引用必须使用该值。"
    use = f"计算：{name} 的值加上 {rng.randint(1, 100)}，结果以「{name} 的最终取值是」开头给出：{name} 的最终取值是 {val}。"
    parts = filler(rng, int(target_tokens * 1.5))
    pos = max(1, int(len(parts) * depth))
    parts.insert(0, definition)
    parts.insert(pos + 1, use)
    return {"text": "\n".join(parts)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成长依赖预训练数据")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--depth", type=float, default=0.9, help="使用点深度比例")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.ctx_tokens, args.depth), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
