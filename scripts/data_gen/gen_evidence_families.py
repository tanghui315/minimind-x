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
train/val 使用互斥部署组；交换/移除 memory 后答案应随之变化。

输出 jsonl：{context, question, answer, family, gold_block, gold_span, split}
gold_span: 证据句在 context 中的 [start, end) 字符偏移；token id 标注由 dataset 按
当前 tokenizer/block_size 转换（参数变更须重新生成，V2 §16.2）。
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
    """事实句散布进填充文本；返回 (context, gold_block, gold_span)。"""
    parts = [filler(rng, 120) for _ in range(n_blocks)]
    n_blocks_real = len(parts)
    gold_idx = max(0, min(n_blocks_real - 1, int(gold_ratio * n_blocks_real)))
    parts[gold_idx] = facts[0]
    for i, f in enumerate(facts[1:], start=1):
        pos = (gold_idx + 2 * i) % n_blocks_real
        parts[pos] = f
    ctx = "".join(parts)
    # gold_span：facts[0] 在 context 中的 [start, end)（T4.5；token id 由 dataset 转换）
    start = ctx.find(facts[0])
    span = [start, start + len(facts[0])] if start >= 0 else [-1, -1]
    return ctx, gold_idx, span


def gen_sample(rng, family, devices=DEVICES):
    dev = rng.choice(devices)
    ver, ver_wrong = rng.sample(VERSIONS, 2)
    val = rng.randint(1000, 9999)
    if family == "no_evidence":
        # 真值修正：输入无任何依据时目标必须是「不可知」，禁止随机编造采样率
        return {"context": "", "question": f"{dev} 的默认采样率是多少？",
                "answer": "上下文未提供默认采样率信息。",
                "gold_span": [-1, -1], "gold_block": -1, "unanswerable": True}
    if family == "evidence_needed":
        facts = [f"配置记录：{dev} 本次部署的阈值设定为 {val}。"]
        ctx, gold, span = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"本次部署中 {dev} 的阈值是多少？",
                "answer": str(val), "gold_block": gold, "gold_span": span}
    if family == "irrelevant":
        # 真值修正：干扰设备必须 ≠ 目标设备，且 context 不得出现目标设备名
        others = [d for d in devices if d != dev]
        other = rng.choice(others)
        facts = [f"配置记录：{other} 的阈值设定为 {val}。"]
        ctx, gold, span = _block_ctx(rng, facts)
        assert dev not in ctx, f"irrelevant 泄漏目标设备: {dev} in ctx"
        assert str(val) in ctx  # 数值只属于 other，答案仍不可知
        return {"context": ctx, "question": f"{dev} 的阈值是多少？",
                "answer": "上下文未提供该设备的阈值信息。", "gold_block": -1, "gold_span": [-1, -1],
                "unanswerable": True}
    if family == "similar_wrong_ver":
        facts = [f"版本说明：{dev} {ver_wrong} 的阈值为 {val}。"]
        ctx, gold, span = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} {ver} 的阈值是多少？",
                "answer": f"上下文只包含 {ver_wrong} 的信息，缺少 {ver} 的阈值。",
                "gold_block": -1, "gold_span": [-1, -1], "unanswerable": True}
    if family == "new_evidence":
        # 真值修正：证据可见时间线——生效约束在 late_block，首轮检索默认不可见
        facts = [f"补充约束（生效中）：{dev} 的实际阈值为 {val}，覆盖早期文档值。",
                 f"早期文档：{dev} 的阈值为 {rng.randint(100, 999)}（已废弃）。"]
        parts = [filler(rng, 150) for _ in range(6)]
        parts[2], parts[4] = facts[1], facts[0]
        ctx = "".join(parts)
        start = ctx.find(facts[0])
        span = [start, start + len(facts[0])] if start >= 0 else [-1, -1]
        return {"context": ctx, "question": f"{dev} 当前的有效阈值是多少？",
                "answer": str(val), "gold_block": 4, "gold_span": span,
                # 可见时间线：生效证据块在第 1 轮后才允许进 Hot Memory
                "evidence_visible_from_round": 1,
                "stale_block": 2, "gold_block_late": 4}
    if family == "multi_hop":
        dev2 = rng.choice([d for d in devices if d != dev])
        facts = [f"链路表：{dev} 的上行通道绑定到 {dev2}。",
                 f"配置记录：{dev2} 的阈值设定为 {val}。"]
        ctx, gold, span = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} 所绑定设备的阈值是多少？",
                "answer": str(val), "gold_block": gold, "gold_span": span}
    if family == "insufficient":
        facts = [f"配置记录：{dev} 的阈值设定为（待填写）。"]
        ctx, gold, span = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} 的阈值是多少？",
                "answer": "记录中该字段为空，无法给出阈值。", "gold_block": gold, "gold_span": span,
                "unanswerable": True}
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
    deployments = list(range(256))
    random.Random(0).shuffle(deployments)
    groups = {"train": deployments[:192], "val": deployments[192:]}
    used = {"train": set(), "val": set()}
    with open(args.out, "w", encoding="utf-8") as f:
        for split, seed in (("train", args.seed_train), ("val", args.seed_val)):
            rng = random.Random(seed)
            for fam in FAMILIES:
                for _ in range(args.n_per_family):
                    site = rng.choice(groups[split])
                    devices = [f"{base}-{site:03d}" for base in DEVICES]
                    rec = gen_sample(rng, fam, devices)
                    rec["deployment_id"] = site
                    used[split].add(site)
                    rec["family"] = fam
                    rec["split"] = split
                    # 真值自检（评审 7.3）：irrelevant 不得泄漏目标设备
                    if fam == "irrelevant":
                        q = rec.get("question", "")
                        # question 含目标设备名是合法的；context 不得含
                        dev_tok = [d for d in devices if d in q]
                        for d in dev_tok:
                            assert d not in rec.get("context", ""), f"irrelevant 泄漏 {d}"
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"written {2 * len(FAMILIES) * args.n_per_family} samples "
          f"(train groups={len(used['train'])}, val groups={len(used['val'])}, "
          f"overlap={len(used['train'] & used['val'])}) -> {args.out}")


if __name__ == "__main__":
    main()
