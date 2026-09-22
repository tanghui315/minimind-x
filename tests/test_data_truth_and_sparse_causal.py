"""合成数据真值 + sparse 块级因果（评审 7.3 / 9.5）。"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from scripts.data_gen.gen_evidence_families import gen_sample, DEVICES
from scripts.data_gen.gen_long_dep import gen_sample as gen_long
from scripts.data_gen.gen_planning import gen_sample as gen_plan, STEPS_ORDERED
from scripts.data_gen.gen_repo_context import gen_sample as gen_repo
import random


def test_irrelevant_never_contains_target_device():
    rng = random.Random(0)
    for i in range(200):
        rec = gen_sample(rng, "irrelevant")
        q = rec["question"]
        # context 不得出现 question 中的目标设备
        for d in DEVICES:
            if d in q:
                assert d not in rec["context"], rec
        assert rec.get("unanswerable") is True


def test_no_evidence_unanswerable_not_random_target():
    rng = random.Random(1)
    answers = set()
    for _ in range(50):
        rec = gen_sample(rng, "no_evidence")
        answers.add(rec["answer"])
        assert rec.get("unanswerable") is True
        assert rec["context"] == ""
    assert len(answers) == 1, f"no_evidence 目标必须确定不可知，得到 {answers}"


def test_new_evidence_has_visibility_timeline():
    rng = random.Random(2)
    rec = gen_sample(rng, "new_evidence")
    assert rec.get("evidence_visible_from_round", 0) >= 1
    assert rec["gold_span"][0] >= 0


def test_long_dep_target_is_x_plus_positive():
    rng = random.Random(3)
    rec = gen_long(rng, 200, 0.5)
    text = rec["text"]
    # 解析「加上 k」与「最终取值是 y」，y 必须 > 初值
    assert "加上" in text and "最终取值是" in text
    import re
    m_add = re.search(r"加上 (\d+)", text)
    m_fin = re.search(r"最终取值是 (\d+)", text)
    m_init = re.search(r"初始值为 (\d+)", text)
    assert m_add and m_fin and m_init
    k, y, x = int(m_add.group(1)), int(m_fin.group(1)), int(m_init.group(1))
    assert y == x + k, f"目标应为 X+k，得到 {x}+{k}!={y}"


def test_planning_ordered_and_constraints_in_answer():
    rng = random.Random(4)
    rec = gen_plan(rng, 4, 200, 8)
    ans = rec["answer"]
    # 步骤必须是 STEPS_ORDERED 的保序子序列
    steps = []
    for name in STEPS_ORDERED:
        if f"：{name}" in ans or f":{name}" in ans or name in ans:
            steps.append(name)
    idxs = [STEPS_ORDERED.index(s) for s in steps if s in STEPS_ORDERED]
    assert idxs == sorted(idxs), f"步骤乱序: {steps}"
    assert "遵守" in ans or "约束" in ans


def test_repo_patch_uses_real_args_no_ellipsis():
    rng = random.Random(5)
    rec = gen_repo(rng, 2, 2, 200, 8)
    ans = rec["answer"]
    assert "..." not in ans
    assert "if not args" not in ans  # 未定义 args


def test_sparse_future_token_cannot_change_past_output():
    """评审 9.5：改未来 token 不得改变早期位置输出（块选择不得泄未来）。"""
    from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
    from model.minimind_x.retrieval.sparse_attn import SparseBlockAttention
    torch.manual_seed(60)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="middle_loop", core_attn="sparse",
                          block_size=8, sparse_topk_blocks=2, attn_window=16)
    attn = SparseBlockAttention(cfg).eval()
    B, S = 1, 16
    x1 = torch.randn(B, S, 32)
    x2 = x1.clone()
    x2[:, 8:] = torch.randn(B, S - 8, 32)  # 只改未来（位置 ≥8）
    # RoPE
    from model.model_minimind import precompute_freqs_cis
    cos, sin = precompute_freqs_cis(dim=cfg.head_dim, end=32)
    cos, sin = cos[:S], sin[:S]
    with torch.no_grad():
        o1, _ = attn(x1, (cos, sin), use_cache=False)
        o2, _ = attn(x2, (cos, sin), use_cache=False)
    # 位置 0..5（完整在第一个块且远早于改动点）应不变
    diff = (o1[:, :5] - o2[:, :5]).abs().max().item()
    assert diff < 1e-5, f"未来 token 影响了早期输出 Δ={diff}"
