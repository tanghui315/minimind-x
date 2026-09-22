"""缓存契约与重放测试（任务清单 P3 验收：T3.1/T3.3/T3.4；V2 §21.3）。

1. R_max 固定槽 + per-token last-valid：混合深度序列 4→1→3 不丢先前深轮 KV，
   浅执行 token 的深槽复制其最后实际轮（只填该 token 位置）
2. memory 版本与轨迹：refresh 递增版本；轨迹含 (事件, 轮, version)；内容变更后投影缓存失效
3. checkpoint 纯化：梯度检查点开/关——检索计数不变（无副作用重放）、loss 一致、梯度仍通
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController


def _model(max_loop=4, **over):
    torch.manual_seed(41)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="middle_loop", max_loop=max_loop)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def _core_cache(m, past):
    """取 core 层（past[P:P+T]）的轮次槽。"""
    cfg = m.config
    return past[cfg.prelude_layers:cfg.prelude_layers + cfg.thinking_layers]


def test_rmax_slots_and_per_token_last_valid():
    """深度序列 4→1→3：槽恒 4；浅 token 深槽=其最后实际轮；先前深槽不丢。"""
    m = _model(max_loop=4)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out = m(ids[:, :3], use_cache=True, num_loops=4)
        past = out.past_key_values
        cc = _core_cache(m, past)
        for slots in cc:
            assert len(slots) == 4, "prefill 后应恒 4 槽"
            assert all(s is not None for s in slots)
        # step4：R=1（浅执行）
        out1 = m(ids[:, 3:4], past_key_values=past, use_cache=True, num_loops=1)
        past = out1.past_key_values
        cc = _core_cache(m, past)
        for slots in cc:
            assert len(slots) == 4, "浅执行后槽数仍 4（R_max 契约）"
            k_new_shallow = slots[0][0][:, -1]          # 该 token 真实轮 0
            for r in (1, 2, 3):                          # 深槽 = last-valid 填充
                assert torch.equal(slots[r][0][:, -1], k_new_shallow), \
                    "浅执行 token 的深槽应复制其最后实际轮"
            assert slots[3][0].shape[1] == 4             # 历史 3+1 token
        # step5：R=3（读到 prefill token 的真实深轮 + step4 token 的填充轮）
        out2 = m(ids[:, 4:5], past_key_values=past, use_cache=True, num_loops=3)
        past = out2.past_key_values
        cc = _core_cache(m, past)
        for slots in cc:
            assert len(slots) == 4
            assert torch.isfinite(slots[2][0]).all()
        assert torch.isfinite(out2.logits).all()


def test_deep_slots_survive_shallow_steps():
    """早停（R=1）不得删除先前 token 的深轮 KV（V2 §11.3 preserve_past_deep_slots）。"""
    m = _model(max_loop=4)
    ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        out = m(ids[:, :4], use_cache=True, num_loops=4)
        deep_kv_before = [s[3][0].clone() for s in _core_cache(m, out.past_key_values)]
        out1 = m(ids[:, 4:5], past_key_values=out.past_key_values, use_cache=True, num_loops=1)
        for l, slots in enumerate(_core_cache(m, out1.past_key_values)):
            assert torch.equal(slots[3][0][:, :4], deep_kv_before[l]), "先前 token 深槽被覆盖"


def test_memory_version_and_trajectory():
    m = _model(variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2, max_loop=2)
    torch.manual_seed(42)
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4,
        oracle_retrieve=[2, 0])   # schema v2：第 0 轮后 GLOBAL_REFRESH
    # T3.3/V2 §9.3：版本随「槽位内容」变更递增；同 source 去重只 bump usage
    assert ctrl.memory.memory_version >= 1, "至少经过初始检索"
    assert len(set(ctrl.memory.source_ids[0].tolist())) == ctrl.memory.size, "source_ids 唯一"
    events = [t["event"] for t in ctrl.trajectory]
    assert "initial_retrieval" in events and "refresh" in events
    vers = [t["memory_version"] for t in ctrl.trajectory]
    assert vers == sorted(vers), "轨迹版本单调不减"
    refresh = [t for t in ctrl.trajectory if t["event"] == "refresh"]
    assert refresh and refresh[0]["round"] == 0, "oracle_retrieve[0]=1 应在轮 0 后 refresh"


def test_projection_cache_invalidated_on_refresh():
    from model.minimind_x.memory import WorkingMemory
    torch.manual_seed(43)
    m = _model(variant="minimind_x", block_size=8, max_loop=2)
    mem = WorkingMemory(torch.randn(1, 3, 8, 32), torch.randn(1, 3, 32), block_size=8)
    kv = mem.get_kv(m.model.thinking_core)
    assert mem._kv_cache is not None
    mem2 = mem.with_added(torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32))
    assert mem2.memory_version == mem.memory_version + 1
    assert mem2._kv_cache is None, "内容变更后投影缓存必须失效"


def test_checkpoint_purity_no_retrieval_side_effects():
    """梯度检查点开/关：检索计数一致（checkpoint 重放无副作用）、loss 一致。"""
    from trainer.x_utils import set_gradient_checkpointing
    torch.manual_seed(44)
    cfg_kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                  num_key_value_heads=1, vocab_size=64, flash_attn=False,
                  variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
                  max_loop=2, residual_v2=True, mhar_strength=1.0, update_progress=1.0)

    def run(ckpt):
        torch.manual_seed(44)
        m = MiniMindXForCausalLM(MiniMindXConfig(**cfg_kw)).train()
        set_gradient_checkpointing(m, ckpt)
        ctrl = AdaptiveController(m)
        ids = torch.randint(3, 60, (1, 8))
        labels = torch.randint(3, 60, (1, 8))
        ctx = torch.randint(3, 60, (8 * 4,))
        out, extras = ctrl.forward_with_oracles(ids, None, labels, ctx, query_len=4)
        loss = sum(l.mean() for l in out.exit_logits)
        loss.backward()
        return ctrl.stats["retrievals"], loss.item(), \
            m.model.thinking_core.update_gate.fc2.weight.grad.abs().sum().item()

    r_off = run(False)
    r_on = run(True)
    assert r_off[0] == r_on[0], "checkpoint 开关不得改变检索计数（纯化契约）"
    assert abs(r_off[1] - r_on[1]) < 1e-4, "loss 应一致"
    assert r_on[2] > 0, "checkpoint 下梯度仍须流动"
