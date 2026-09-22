"""外部评审 G1 修复回归（query 泄漏 / 动作编号 / memory NaN / 空库 / max_steps 语义）。"""
import sys, os, inspect
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.memory.memory_manager import MemoryManager
from model.minimind_x.memory.working_memory import WorkingMemory
from model.minimind_x.router.retrieval_router import (
    ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH,
    normalize_oracle_actions)


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=2, residual_v2=True)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_action_schema_no_swap():
    """评审 4.1：0/1/2 与枚举一致；v1 二值只在边界转换一次。"""
    assert (ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH) == (0, 1, 2)
    assert normalize_oracle_actions([0, 1, 2], 2) == [0, 1, 2]
    # v1: 1=REFRESH → v2 GLOBAL=2；不得变成 WARM=1
    assert normalize_oracle_actions([0, 1], 1) == [0, 2]
    with pytest.raises(ValueError):
        normalize_oracle_actions([3], 2)


def test_sweep_budget_requires_query_len():
    """评审 3.1：sweep_budget 缺 query_len 必须报错。"""
    from trainer.oracle_labels import sweep_budget
    with pytest.raises(ValueError):
        sweep_budget(None, None, None, None, lambda: None)


def test_forward_requires_query_len_in_training():
    """评审 3.1/3.2：训练态缺 query_len 拒绝默认全序列。"""
    torch.manual_seed(1)
    m = MiniMindXForCausalLM(_cfg()).train()
    ctrl = AdaptiveController(m)
    ids = torch.randint(3, 60, (1, 8))
    y = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (32,))
    with pytest.raises(ValueError):
        ctrl.forward_with_oracles(ids, None, y, ctx, query_len=None)


def test_single_block_score_finite():
    """评审 6.1：单块 utility 不得 Inf/NaN。"""
    torch.manual_seed(2)
    cfg = _cfg(variant="middle_loop", residual_v2=False)
    mm = MemoryManager(cfg)
    mem = WorkingMemory(torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32), block_size=8)
    u1 = mm.score(torch.randn(1, 32), mem)
    assert torch.isfinite(u1).all(), u1
    mem2 = mem.with_persistence_ema(u1)
    u2 = mm.score(torch.randn(1, 32), mem2)
    assert torch.isfinite(u2).all(), u2


def test_usage_hits_correct_source_after_dedup():
    """评审 6.2：命中 source 10 不得加到 source 30。"""
    torch.manual_seed(3)
    cfg = _cfg(variant="middle_loop", residual_v2=False)
    mm = MemoryManager(cfg)
    mem = WorkingMemory(torch.randn(1, 3, 8, 32), torch.randn(1, 3, 32), block_size=8,
                        source_ids=torch.tensor([[10, 20, 30]]))
    mem2, _, _ = mm.update(mem, torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32),
                           torch.randn(1, 32), 8,
                           hit_new=torch.ones(1, 1), source_ids=torch.tensor([[10]]))
    # source 10 应被 bump，30 不应
    u10 = mem2.usage[0, (mem2.source_ids[0] == 10).nonzero().item()].item()
    u30 = mem2.usage[0, (mem2.source_ids[0] == 30).nonzero().item()].item()
    assert u10 > u30, (mem2.source_ids, mem2.usage)


def test_empty_context_no_crash():
    """评审 6.3：空 context → 合法空 memory，不 torch.cat 崩。"""
    torch.manual_seed(4)
    m = MiniMindXForCausalLM(_cfg()).eval()
    ctrl = AdaptiveController(m)
    ctrl.begin(torch.zeros(0, dtype=torch.long))
    q = m.model.prelude_hidden(torch.randint(3, 60, (1, 4)), None)
    mem = ctrl.retrieve_into_memory(q)
    assert mem.is_empty()
    out, extras = ctrl.forward_with_oracles(
        torch.randint(3, 60, (1, 8)), None, torch.randint(3, 60, (1, 8)),
        torch.zeros(0, dtype=torch.long), query_len=4, oracle_retrieve=[0, 0])
    assert torch.isfinite(out.logits).all()


def test_oracle_action_ids_execute_as_labeled():
    """评审 4.1：oracle=1 必须执行 RESCORE_WARM（不是 GLOBAL）。"""
    torch.manual_seed(5)
    m = MiniMindXForCausalLM(_cfg(max_loop=3)).eval()
    ids = torch.randint(3, 60, (1, 8))
    y = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (32,))
    ctrl = AdaptiveController(m)
    ctrl.forward_with_oracles(ids, None, y, ctx, query_len=4,
                              oracle_retrieve=normalize_oracle_actions([1, 0], 2))
    events = [t["event"] for t in ctrl.trajectory]
    assert "rescore_warm" in events, events
    assert ctrl.stats.get("warm_rescore", 0) >= 1


def test_router_train_input_matches_execution_prefix():
    """评审 3.2：监督用 [:query_len]，换 gold 后缀不改 halt/action logits。"""
    torch.manual_seed(6)
    m = MiniMindXForCausalLM(_cfg()).eval()
    ids = torch.randint(3, 60, (1, 10))
    ids2 = ids.clone()
    ids2[0, 5:] = torch.randint(3, 60, (5,))  # 改 gold 后缀
    q = 5
    with torch.no_grad():
        o1 = m(ids, collect_exits=True, num_loops=2, query_len=q)
        o2 = m(ids2, collect_exits=True, num_loops=2, query_len=q)
    assert o1.router_probs and o2.router_probs
    h1, h2 = o1.router_probs["halt"][0], o2.router_probs["halt"][0]
    assert torch.allclose(h1, h2, atol=1e-5), (h1, h2)
    a1 = o1.router_probs["retrieval_action"][0]
    a2 = o2.router_probs["retrieval_action"][0]
    assert torch.allclose(a1, a2, atol=1e-5)
