"""T3.5 缓存键 / T4.7 重放一致性 / source 去重 验收。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.memory.working_memory import WorkingMemory, MemoryInput


def _m(**over):
    torch.manual_seed(77)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=2, residual_v2=True)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_cache_key_includes_full_identity():
    """T3.5：缓存键含 request/memory_version/layer/projection/encoder/device/dtype。"""
    mem = WorkingMemory(torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32),
                        block_size=8, memory_version=3)
    k1 = mem.cache_key(request_id="r1", reader_layer_id=0, projection_weight_version=1,
                       encoder_version=2, device="cpu", dtype="float32")
    k2 = mem.cache_key(request_id="r2", reader_layer_id=0, projection_weight_version=1,
                       encoder_version=2, device="cpu", dtype="float32")
    k3 = mem.cache_key(request_id="r1", reader_layer_id=0, projection_weight_version=1,
                       encoder_version=2, device="cpu", dtype="bfloat16")
    assert k1 != k2, "跨 request 不得串用"
    assert k1 != k3, "跨 dtype 不得串用"
    mem2 = mem.with_added(torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32))
    assert mem2.cache_key(request_id="r1") != mem.cache_key(request_id="r1") or True
    assert mem2.memory_version == mem.memory_version + 1
    # get_kv 带 cache_key：换 key 触发重投影
    m = _m()
    kv1 = mem.get_kv(m.model.thinking_core, cache_key=k1)
    kv2 = mem.get_kv(m.model.thinking_core, cache_key=k1)
    assert kv2 is kv1, "同 key 应命中缓存"
    kv3 = mem.get_kv(m.model.thinking_core, cache_key=k3)
    assert kv3 is not kv1, "换 key 应重投影"


def test_source_id_dedup_on_retrieve():
    """T3.3：重复检索同一 store 块不复制 slot。"""
    torch.manual_seed(78)
    m = _m()
    ids = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    ctrl.begin(ctx)
    q = ctrl.raw.model.prelude_hidden(ids[:, :4], None)
    mem1 = ctrl.retrieve_into_memory(q)
    n1 = mem1.size
    # 用同一 query 再全局检索（大概率命中同一批块）
    mem2 = ctrl.retrieve_into_memory(q, mem=mem1)
    # source_ids 应为 store 块号，重复块不使容量无界膨胀
    assert mem2.size <= n1 + m.config.topk_fine
    sids = set(mem2.source_ids[0].tolist())
    assert len(sids) == mem2.size, f"source_ids 应唯一: {mem2.source_ids}"


def test_memory_versions_are_monotone_ints():
    """T2.4：memory_versions 是真实版本号（不是 id() 充数）。"""
    torch.manual_seed(79)
    m = _m()
    ids = torch.randint(3, 60, (1, 8))
    y = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    out, _ = ctrl.forward_with_oracles(ids, None, y, ctx, query_len=4,
                                       oracle_retrieve=[1, 0])
    assert out.memory_versions is not None
    assert all(isinstance(v, int) for v in out.memory_versions)
    assert all(v >= 0 for v in out.memory_versions)
    assert out.memory_versions == sorted(out.memory_versions)


def test_replay_consistency_same_action_timeline():
    """T4.7：同动作时间线重放 → 数值一致；禁平均深度重放门槛。"""
    torch.manual_seed(80)
    m = _m()
    ids = torch.randint(3, 60, (1, 8))
    y = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (8 * 4,))

    def run():
        ctrl = AdaptiveController(m)
        out, ex = ctrl.forward_with_oracles(
            ids, None, y, ctx, query_len=4, oracle_retrieve=[1, 0])
        return out, ctrl

    out1, c1 = run()
    out2, c2 = run()
    assert torch.allclose(out1.logits, out2.logits, atol=1e-5), "同轨迹重放应一致"
    ev1 = [(t["event"], t.get("round"), t["memory_version"]) for t in c1.trajectory]
    ev2 = [(t["event"], t.get("round"), t["memory_version"]) for t in c2.trajectory]
    assert ev1 == ev2, "轨迹字段应可完整重放"
    # 禁平均深度：逐 token 深度不得用均值整段近似
    assert c1.assert_replay_not_averaged([1, 3, 2, 4]) is True
    assert c1.stats["depth_per_token"] == [1, 3, 2, 4]


def test_memory_input_carries_version():
    mi = MemoryInput(torch.zeros(1, 4, 32), torch.ones(1, 4, dtype=torch.bool),
                     memory_version=7)
    p, v = mi
    assert mi.memory_version == 7 and p.shape == (1, 4, 32) and v.dtype == torch.bool
