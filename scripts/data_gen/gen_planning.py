"""Planning / Agent 任务数据（任务清单补 Doc 1 §23「Planning / Agent Task」）。

多步任务规划：上下文含任务描述 + 资源/约束清单（散布），问题要求给出执行计划。
产出检索式 jsonl：{context, question, answer, gold_block}。
"""
import os
import json
import random
import argparse

RESOURCES = ["服务器A", "服务器B", "数据库C", "缓存D", "网关E", "队列F", "镜像仓库G"]
CONSTRAINTS = ["每日 02:00 维护窗口", "带宽上限 1Gbps", "磁盘余量 20%", "并发上限 500",
               "审计要求全量日志", "回滚需双人审批"]
# 真值修正：步骤有硬依赖（必须按此偏序取子序列，禁止无放回乱序）
STEPS_ORDERED = ["备份当前状态", "更新配置", "扩容资源", "切换流量",
                 "验证健康检查", "通知相关方", "归档日志", "清理临时文件"]


def filler(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        parts.append(rng.choice(["例行巡检记录正常。", "容量报表已生成。", "告警面板无待办。",
                                 "变更单已关闭。", "值班交接完成。", "备份任务成功。"]))
        total += 12
    return parts


def gen_sample(rng, n_steps, ctx_tokens, block_size):
    target_chars = int(ctx_tokens * 1.5)
    parts = filler(rng, target_chars)
    res = rng.sample(RESOURCES, k=min(3, len(RESOURCES)))
    cons = rng.sample(CONSTRAINTS, k=2)
    # 取有序子序列（保序），保证步骤依赖成立
    n_steps = max(2, min(n_steps, len(STEPS_ORDERED)))
    idxs = sorted(rng.sample(range(len(STEPS_ORDERED)), k=n_steps))
    plan = [STEPS_ORDERED[i] for i in idxs]
    assert plan == [STEPS_ORDERED[i] for i in idxs] and idxs == sorted(idxs)
    # 资源/约束散布进上下文（需检索组合）
    doc = [f"资源清单：本次可使用 {('、'.join(res))}。"] + \
          [f"约束{i + 1}：{c}。" for i, c in enumerate(cons)]
    insert_at = sorted(rng.sample(range(len(parts)), len(doc)), reverse=True)
    for pos, line in zip(insert_at, doc):
        parts.insert(pos, line)
    question = (f"目标：完成一次上线变更，涉及 {res[0]} 与 {res[-1]}，需遵守全部约束，"
                f"共 {n_steps} 步。请给出按序执行计划。")
    # 答案必须落实约束（否则目标不完整）
    answer = ("；".join(f"第{i + 1}步：{s}" for i, s in enumerate(plan)) +
              f"。执行时遵守：{'；'.join(cons)}。")
    ctx = "".join(parts)
    first = doc[0]
    gold_ratio = ctx.find(first) / max(len(ctx), 1)
    n_blocks = max(1, round(ctx_tokens / block_size))
    return {"context": ctx, "question": question, "answer": answer,
            "gold_block": min(int(gold_ratio * n_blocks), n_blocks - 1)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 Planning/Agent 检索式数据（Doc 1 §23）")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--steps", type=str, default="3,5", help="计划步数（可逗号分隔混合）")
    p.add_argument("--ctx_tokens", type=int, default=4096)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    step_pool = [int(x) for x in args.steps.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            f.write(json.dumps(gen_sample(rng, step_pool[i % len(step_pool)],
                                          args.ctx_tokens, args.block_size),
                               ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
