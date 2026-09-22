"""Multi-hop 推理数据（任务清单 T3.1；Doc 4 §8）。

构造实体链条 A→B→C→D…，事实句打乱铺入上下文，问题要求跨信息组合。
产出检索式 jsonl：{context, question, answer, gold_block}。
"""
import os
import json
import random
import argparse

SURNAMES = "赵 钱 孙 李 周 吴 郑 王 冯 陈 蒋 沈 韩 杨".split()
ROLES = ["负责人", "对接人", "审核人", "导师", "合作者"]
ENT_FORMATS = [
    "{a} 的{role}是 {b}。",
    "记录显示，{a} 由 {b} 负责。",
    "根据登记表，{a} 的联系人是 {b}。",
]
FILLER = ["该季度项目按计划推进。", "会议记录已归档。", "相关审批流程已完成。",
          "预算执行情况正常。", "系统运行状态良好。", "文档已同步至知识库。"]


def gen_sample(rng, hops, ctx_tokens, block_size):
    chain = [rng.choice(SURNAMES) + rng.choice("明芳军婷宇晨静磊") for _ in range(hops + 1)]
    facts = []
    for i in range(hops):
        facts.append(rng.choice(ENT_FORMATS).format(a=chain[i], role=rng.choice(ROLES), b=chain[i + 1]))
    target_chars = int(ctx_tokens * 1.5)
    parts = [rng.choice(FILLER) for _ in range(max(10, target_chars // 12))]
    insert_at = sorted(rng.sample(range(len(parts)), min(hops, len(parts))), reverse=True)
    for pos, fact in zip(insert_at, facts):
        parts.insert(pos, fact)
    rng.shuffle(insert_at)  # gold 块按最终位置折算（简化：取第一条事实）
    first_fact = facts[0]
    q = chain[0]
    for i in range(1, hops):
        q += f" 的{ROLES[0]}"
    question = f"{'起点 ' + chain[0]}经过 {hops} 跳关系后指向谁？"
    answer = chain[-1]
    ctx = "".join(parts)
    gold_ratio = ctx.find(first_fact) / max(len(ctx), 1)
    n_blocks = max(1, round(ctx_tokens / block_size))
    return {"context": ctx, "question": question, "answer": answer,
            "gold_block": min(int(gold_ratio * n_blocks), n_blocks - 1), "chain": chain}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 multi-hop 检索式数据")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--hops", type=str, default="2", help="跳数（可逗号分隔混合 1,2,4,8）")
    p.add_argument("--ctx_tokens", type=int, default=4096)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    hop_pool = [int(x) for x in args.hops.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            f.write(json.dumps(gen_sample(rng, hop_pool[i % len(hop_pool)],
                                          args.ctx_tokens, args.block_size), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
