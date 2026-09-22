"""Memory 管理监督数据（任务清单 T5.5；Doc 4 §19）。

构造 Keep/Drop/Replace 标签：给定原 memory blocks、新块、query，
标签 = 按「含 query 关键词 + 去重」规则应保留的块（可学习规则）。
jsonl：{memory_blocks, new_blocks, query, keep}
"""
import os
import json
import random
import argparse

TOPICS = ["数据库", "网络", "安全", "编译", "调度", "存储", "前端", "算法", "监控", "部署"]


def gen_block(rng, topic=None):
    topic = topic or rng.choice(TOPICS)
    return f"{topic}模块：{topic}的{rng.choice(['配置', '状态', '日志', '依赖', '接口'])}" \
           f"记录 {rng.randint(100, 999)}，{rng.choice(['正常', '已更新', '待处理', '已归档'])}。"


def gen_sample(rng, mem_n, new_n, keep_n):
    topic = rng.choice(TOPICS)
    query = f"当前任务需要查询{topic}相关的信息"
    blocks = [gen_block(rng) for _ in range(mem_n)]
    news = [gen_block(rng, topic if rng.random() < 0.7 else None) for _ in range(new_n)]
    # 标签规则：与 query 同主题（含关键词）优先保留，重复内容去重（只留首个）
    def score(b):
        return (1 if topic in b else 0) + rng.random() * 0.01
    seen, pool = set(), []
    for i, b in enumerate(blocks + news):
        key = b.split("记录")[0]
        if key in seen:
            continue
        seen.add(key)
        pool.append((score(b), f"{i}"))
    pool.sort(reverse=True)
    keep = [i for _, i in pool[:keep_n]]
    return {"memory_blocks": blocks, "new_blocks": news, "query": query, "keep": keep}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 memory 管理监督数据")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--mem_blocks", type=int, default=8)
    p.add_argument("--new_blocks", type=int, default=4)
    p.add_argument("--keep", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.mem_blocks, args.new_blocks, args.keep),
                               ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
