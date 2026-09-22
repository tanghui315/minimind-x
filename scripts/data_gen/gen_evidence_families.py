"""七族证据数据生成器（任务清单 T4.5；V2 文档 §16.1-16.2）。

样本族（每族一个 family 字段，训练/验收目标见 V2 表）：
  no_evidence      memory=None —— Reader 不应扰动无 memory 路径
  evidence_needed  答案含本批随机事实，prompt 无法推断 —— Reader 必须用外部信息
  irrelevant       与问题无关、格式相似的资料 —— 不因存在就注入
  similar_wrong_ver 相同术语、不同版本/对象 —— 学适用性而非关键词
  new_evidence     关键约束只在「后续轮才检索到」的位置 —— Gate 可重新打开
  multi_hop        第二跳信息依赖第一跳结果 —— Think→Retrieve→Think 闭环
  insufficient     确实缺少必要字段 —— 允许承认不足而非强行回答

机制验收（V2 §16.2 随机事实任务）：evidence_needed/new_evidence 每条随机换外部事实，
train/val 用不同随机种子（事实组合不重叠）；交换/移除 memory 后答案应随之变化。

输出 jsonl：{context, question, answer, family, gold_block, split}
"""
import os
import json
import random
import argparse

DEVICES = ["网关A", "网关B", "传感节点C", "执行器D", "边缘盒E", "协处理器F"]
VERSIONS = ["v1.2.0", "v1.3.0", "v2.0.1", "v2.1.0-rc"]


def filler(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        parts.append(rng.choice(["例行遥测正常。", "固件校验通过。", "链路心跳 200ms。",
                                 "缓冲区水位 42%。", "任务队列空。", "看门狗复位计数 0。"]))
        total += 10
    return "".join(parts)


def _block_ctx(rng, facts, n_blocks=6, gold_ratio=0.5):
    """事实句散布进填充文本；返回 (context, gold_block)。"""
    parts = [filler(rng, 120) for _ in range(n_blocks)]
    n_blocks_real = len(parts)
    gold_idx = max(0, min(n_blocks_real - 1, int(gold_ratio * n_blocks_real)))
    parts[gold_idx] = facts[0]
    for i, f in enumerate(facts[1:], start=1):
        pos = (gold_idx + 2 * i) % n_blocks_real
        parts[pos] = f
    return "".join(parts), gold_idx


def gen_sample(rng, family):
    dev = rng.choice(DEVICES)
    ver, ver_wrong = rng.sample(VERSIONS, 2)
    val = rng.randint(1000, 9999)
    if family == "no_evidence":
        return {"context": "", "question": f"{dev} 的默认采样率是多少？",
                "answer": f"{dev} 的默认采样率为 {rng.choice(['10Hz', '20Hz', '50Hz'])}。"}
    if family == "evidence_needed":
        facts = [f"配置记录：{dev} 本次部署的阈值设定为 {val}。"]
        ctx, gold = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"本次部署中 {dev} 的阈值是多少？",
                "answer": str(val), "gold_block": gold}
    if family == "irrelevant":
        facts = [f"配置记录：{rng.choice(DEVICES)} 的阈值设定为 {val}。"]  # 其它设备
        ctx, gold = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} 的阈值是多少？",
                "answer": "上下文未提供该设备的阈值信息。", "gold_block": -1}
    if family == "similar_wrong_ver":
        facts = [f"版本说明：{dev} {ver_wrong} 的阈值为 {val}。"]
        ctx, gold = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} {ver} 的阈值是多少？",
                "answer": f"上下文只包含 {ver_wrong} 的信息，缺少 {ver} 的阈值。",
                "gold_block": -1}
    if family == "new_evidence":
        # 关键约束放在文本后段（后续轮检索更可能命中的位置）
        facts = [f"补充约束（生效中）：{dev} 的实际阈值为 {val}，覆盖早期文档值。",
                 f"早期文档：{dev} 的阈值为 {rng.randint(100, 999)}（已废弃）。"]
        parts = [filler(rng, 150) for _ in range(6)]
        parts[2], parts[4] = facts[1], facts[0]
        return {"context": "".join(parts), "question": f"{dev} 当前的有效阈值是多少？",
                "answer": str(val), "gold_block": 4}
    if family == "multi_hop":
        dev2 = rng.choice([d for d in DEVICES if d != dev])
        facts = [f"链路表：{dev} 的上行通道绑定到 {dev2}。",
                 f"配置记录：{dev2} 的阈值设定为 {val}。"]
        ctx, gold = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} 所绑定设备的阈值是多少？",
                "answer": str(val), "gold_block": gold}
    if family == "insufficient":
        facts = [f"配置记录：{dev} 的阈值设定为（待填写）。"]
        ctx, gold = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} 的阈值是多少？",
                "answer": "记录中该字段为空，无法给出阈值。", "gold_block": gold}
    raise ValueError(family)


FAMILIES = ["no_evidence", "evidence_needed", "irrelevant", "similar_wrong_ver",
            "new_evidence", "multi_hop", "insufficient"]


def main():
    p = argparse.ArgumentParser(description="生成七族证据数据（V2 §16.1）")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n_per_family", type=int, default=200)
    p.add_argument("--seed_train", type=int, default=42)
    p.add_argument("--seed_val", type=int, default=1042)  # 与 train 不重叠的事实随机流
    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for split, seed in (("train", args.seed_train), ("val", args.seed_val)):
            rng = random.Random(seed)
            for fam in FAMILIES:
                for _ in range(args.n_per_family):
                    rec = gen_sample(rng, fam)
                    rec["family"] = fam
                    rec["split"] = split
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"written {2 * len(FAMILIES) * args.n_per_family} samples "
          f"(train/val 事实流隔离) -> {args.out}")


if __name__ == "__main__":
    main()
