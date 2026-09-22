"""V2 Core 集成测试（任务清单 T2.1-T2.3 验收）。

1. 严格兼容模式（T2.3）：η=0 + update_progress=0 + reader_scale=0 的 V2 模型，
   加载同一套主干权重后，与 V1 模型 logits 逐位一致（迁移回归锚点）
2. V2 前向（含 memory_payload）+ 生成可用
3. 三模块在真实模型路径中全部获得梯度（T2.1/R-06 的模型侧前置）
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def _cfg(v2=False, **over):
    kw = dict(hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
              num_key_value_heads=2, vocab_size=128, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=3, residual_v2=v2)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_v2_strict_compat_equivalence():
    """T2.3：严格兼容（η=0/λ=0/scale=0）下 V2 == V1（同主干权重）。"""
    torch.manual_seed(31)
    v1 = MiniMindXForCausalLM(_cfg(v2=False)).eval()
    v2 = MiniMindXForCausalLM(_cfg(
        v2=True, mhar_strength=0.0, update_progress=0.0,
        reader_scale_init=0.0)).eval()
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith(("model.thinking_core.loop_mhar",
                             "model.thinking_core.update_gate",
                             "model.thinking_core.memory_reader"))
               for k in missing), missing
    x = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        for r in (1, 2, 3):
            l1 = v1(x, num_loops=r).logits
            l2 = v2(x, num_loops=r).logits
            assert torch.allclose(l1, l2, atol=1e-5), f"R={r} 不等价: {(l1-l2).abs().max()}"


def test_v2_forward_with_memory_payload():
    torch.manual_seed(32)
    m = MiniMindXForCausalLM(_cfg(v2=True)).eval()
    x = torch.randint(0, 128, (2, 8))
    payload = torch.randn(2, 3 * 8, m.config.hidden_size)   # 3 块 × 8 token
    valid = torch.ones(2, payload.shape[1], dtype=torch.bool)
    with torch.no_grad():
        out = m(x, num_loops=2, memory_payload=(payload, valid))
        out_none = m(x, num_loops=2)
    assert torch.isfinite(out.logits).all()
    assert not torch.allclose(out.logits, out_none.logits, atol=1e-3)  # reader 生效


def test_v2_modules_get_gradients_in_model_path():
    torch.manual_seed(33)
    m = MiniMindXForCausalLM(_cfg(v2=True))
    x = torch.randint(0, 128, (2, 8))
    y = torch.randint(0, 128, (2, 8))
    payload = torch.randn(2, 16, m.config.hidden_size)
    valid = torch.ones(2, 16, dtype=torch.bool)
    out = m(x, labels=y, num_loops=2, memory_payload=(payload, valid))
    out.loss.backward()
    core = m.model.thinking_core
    assert core.update_gate.fc2.weight.grad is not None and \
        core.update_gate.fc2.bias.grad is not None
    assert core.memory_reader.o_proj.weight.grad is not None and \
        core.memory_reader.k_proj.weight.grad is not None
    # MHAR：λ=1 且 η=1 时 query/type_bias 均在路径上（第 1 轮只有 anchor 单来源，
    # 第 2 轮起 query 参与混合 → 梯度经由第 2 轮传导）
    assert core.loop_mhar.query.grad is not None or core.loop_mhar.type_bias.grad is not None


def test_v2_generate_and_controller_end_to_end():
    """终局 controller × V2：payload 注入 + router 再检索 + 生成全链路。"""
    from model.minimind_x.controller import AdaptiveController
    torch.manual_seed(34)
    m = MiniMindXForCausalLM(_cfg(v2=True)).eval()
    ctrl = AdaptiveController(m)
    ids = torch.randint(0, 128, (1, 6))
    ctx = torch.randint(0, 128, (m.config.block_size * 4,))
    out = ctrl.generate_with_context(ids, context_ids=ctx, max_new_tokens=4,
                                     do_sample=False, eos_token_id=None)
    assert out["generated_ids"].shape[1] == 10
    assert out["stats"]["retrievals"] >= 1
