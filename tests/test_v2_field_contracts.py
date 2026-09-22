"""T2.1 halt 统计 / T2.4 OutputV2 / T3.3 WorkingMemoryV2 / T5.2 完整 checkpoint 验收。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.memory.working_memory import WorkingMemory


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=2, residual_v2=True)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_halt_receives_detached_gate_stats():
    """T2.1：halt 读 committed + detached 统计（ḡ/Δ^cand/Δ^commit），零初始化旁路。"""
    torch.manual_seed(1)
    m = MiniMindXForCausalLM(_cfg()).eval()
    x = torch.randint(3, 60, (2, 8))
    with torch.no_grad():
        out = m(x, num_loops=2, halt=True, collect_exits=True)
    assert out.residual_stats is not None and len(out.residual_stats) >= 1
    rs = out.residual_stats[0]
    for k in ("gate_mean", "delta_cand", "delta_commit"):
        assert k in rs and torch.isfinite(rs[k]).all()
    # 旁路零初始化：无统计 vs 零统计 应一致；有统计时 logit 可被推动
    core = m.model.thinking_core
    h = torch.randn(2, 8, 32)
    hr = m.model.halt_router
    p0 = hr(h, None)
    p_zero = hr(h, {"gate_mean": torch.zeros(2), "delta_cand": torch.zeros(2),
                    "delta_commit": torch.zeros(2)})
    assert torch.allclose(p0, p_zero, atol=1e-6)
    # 旧权重兼容：proj 仍是 hidden_size 输入
    assert hr.proj.in_features == 32


def test_output_v2_fields():
    """T2.4：core_committed / coda_exit / halt_logits / memory_versions / residual_stats。"""
    torch.manual_seed(2)
    m = MiniMindXForCausalLM(_cfg()).train()
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4, oracle_retrieve=[0, 0])
    assert out.core_committed_states is not None
    assert out.coda_exit_states is not None
    assert out.halt_logits is not None and len(out.halt_logits) == len(out.core_committed_states)
    assert out.memory_versions is not None
    assert out.residual_stats is not None
    assert out.router_trace is not None and len(out.router_trace) >= 1
    # Router 用 committed（T2.1 顺序），committed ≠ 空
    assert out.core_committed_states[0].shape[-1] == 32


def test_working_memory_v2_valid_metadata_dedup():
    """T3.3：valid/metadata + source_id 去重（同 source 不复制 slot）。"""
    torch.manual_seed(3)
    mem = WorkingMemory(torch.randn(1, 2, 8, 32), torch.randn(1, 2, 32),
                        block_size=8, source_ids=torch.tensor([[10, 11]]),
                        metadata={"origin": "test"})
    assert mem.valid.shape == (1, 2) and mem.valid.all()
    assert mem.metadata["origin"] == "test"
    assert mem.token_valid().shape == (1, 16)
    # 并入已有 source_id=10 → 不新增 slot（去重）
    mem2 = mem.with_added(torch.randn(1, 2, 8, 32), torch.randn(1, 2, 32),
                          source_ids=torch.tensor([[10, 12]]))
    assert mem2.size == 3, f"去重后应 3 槽（10/11/12），得到 {mem2.size}"
    assert set(mem2.source_ids[0].tolist()) == {10, 11, 12}
    assert mem2.memory_version == mem.memory_version + 1
    assert mem2.valid.shape[1] == 3


def test_full_checkpoint_roundtrip():
    """T5.2：optimizer/RNG/进度 manifest 落盘并可恢复。"""
    from trainer.x_utils import (save_x_checkpoint, load_x_checkpoint_meta,
                                 restore_training_state, build_optimizer, param_manifest)
    torch.manual_seed(4)
    m = MiniMindXForCausalLM(_cfg()).train()
    opt = build_optimizer(m, lr=1e-3)
    # 走一步产生 optimizer state
    x = torch.randint(3, 60, (1, 6))
    y = torch.randint(3, 60, (1, 6))
    loss = m(x, labels=y, num_loops=1).loss
    loss.backward()
    opt.step()
    man = param_manifest(m)
    assert man["total"] > 0 and "param_groups" in man
    assert man["param_groups"]["residual_decay"] + man["param_groups"]["residual_no_decay"] > 0
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = save_x_checkpoint(
            m, d, "ckpt", 32, optimizer=opt,
            extra={"data_progress": {"step": 7}, "warmup_progress": 0.5,
                   "tokenizer_manifest": {"name": "test"},
                   "source_manifest": {"dataset": "evidence"}})
        weights, meta = load_x_checkpoint_meta(path)
        assert meta["architecture_version"] == "minimind_x_residual_v2"
        assert meta["param_manifest"]["total"] == man["total"]
        assert meta["optimizer"] is not None
        assert meta["rng_torch"] is not None
        assert meta["data_progress"]["step"] == 7
        assert meta["warmup_progress"] == 0.5
        assert meta["tokenizer_manifest"]["name"] == "test"
        opt2 = build_optimizer(m, lr=1e-3)
        info = restore_training_state(meta, optimizer=opt2)
        assert "optimizer" in info["restored"] and "rng_torch" in info["restored"]
        assert info["warmup_progress"] == 0.5
