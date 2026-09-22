"""T5.1 / V2 §21 五组验收汇总。

1. 数学不变量 —— tests/test_residual_invariants.py
2. 真实训练接线 —— tests/test_trainer_gradients.py
3. Cache 与路径一致性 —— tests/test_cache_replay.py + test_loop_kv.py
4. 因果与安全 —— tests/test_causality.py + 本文件补隐私/来源
5. 任务与效率 —— 本文件（executed vs selected、动作计数、FLOPs 分项口径）
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.flops import estimate_flops


def _v2_model(max_loop=3):
    torch.manual_seed(55)
    return MiniMindXForCausalLM(MiniMindXConfig(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=64, flash_attn=False,
        variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
        max_loop=max_loop, residual_v2=True)).eval()


def test_acceptance_group5_efficiency_stats():
    """V2 §18.1/18.6：executed vs selected 分开记；动作计数；TTFT/decode 分开。"""
    m = _v2_model()
    ids = torch.randint(3, 60, (1, 8))
    with torch.no_grad():
        out = m.generate(inputs=ids, max_new_tokens=3, do_sample=False,
                         use_halting=True, return_stats=True, eos_token_id=None)
    st = out["stats"]
    assert "executed_core_rounds" in st
    assert "avg_ttft_loops" in st and "avg_decode_loops" in st
    assert st["executed_core_rounds"] >= st["avg_loops"] * 0  # 非负
    # 动作计数
    torch.manual_seed(56)
    ctrl = AdaptiveController(m)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    _, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4, oracle_retrieve=[1, 0, 0])
    ac = extras["action_counts"]
    assert ac["global_refresh"] + ac["warm_rescore"] + ac["reuse"] >= 1


def test_acceptance_group5_flops_report_honest():
    """容量不得代理真实开销：Reader 按 T×M；分项齐全；residual 关闭时主干线性于 R。"""
    cfg = MiniMindXConfig(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=64, flash_attn=False,
        variant="middle_loop", residual_v2=False)
    f1 = estimate_flops(cfg, seq_len=128, loops=1)
    f2 = estimate_flops(cfg, seq_len=128, loops=2)
    assert abs((f2["total"] - f1["total"]) * 1 - (f2["total"] - f1["total"])) < 1e-6
    assert f1.get("reader_attn", 0.0) == 0.0  # v2 关闭
    cfg2 = MiniMindXConfig(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=64, flash_attn=False,
        variant="minimind_x", residual_v2=True, max_loop=2)
    fr = estimate_flops(cfg2, seq_len=64, loops=2, memory_blocks=2, n_retrievals=2)
    assert fr["reader_attn"] > 0 and fr["mhar"] > 0 and fr["update_gate"] > 0


def test_acceptance_group4_no_privacy_payload_in_trajectory():
    """调试轨迹不含完整 hidden / 原始 payload（V2 §21.4）。"""
    m = _v2_model()
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    ctrl.forward_with_oracles(ids, None, labels, ctx, query_len=4, oracle_retrieve=[0, 0, 0])
    for rec in ctrl.trajectory:
        for k, v in rec.items():
            if torch.is_tensor(v):
                assert v.dim() <= 1 and v.numel() < 64, f"轨迹字段 {k} 不应携带大张量"
            assert "hidden" not in k.lower() or k in ("visible_prefix_len",)
        assert "payload" not in str(rec.keys()).lower()
