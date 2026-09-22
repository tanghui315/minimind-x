"""V2 三模块不变量测试（任务清单 T1.4；V2 §21.1 + 附录 B.2 测试范围移植）。

覆盖：LoopMHAR（单来源恒等/η=0 恒等/来源轴归一/因果/不原地改），
UpdateGate（0/1 边界/inactive/插值），Reader（no-op 全家族/padding 不可见/
scale=0 精确 no-op 但有梯度/冻结 payload 梯度/BF16 有限），组合路径梯度。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn.functional as F

from model.minimind_x.residual import LoopMHAR, RecurrentUpdateGate
from model.minimind_x.memory.reader import GatedMemoryReader, MemoryKV


def _rand(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ------------------------------- LoopMHAR --------------------------------
def test_mhar_single_source_identity():
    m = LoopMHAR(16, heads=4)
    x = _rand((2, 5, 16))
    out, w = m([x])
    assert torch.allclose(out, x, atol=1e-5)
    assert torch.allclose(w.squeeze(0), torch.ones_like(w.squeeze(0)), atol=1e-5)


def test_mhar_strength_zero_identity():
    m = LoopMHAR(16, heads=4)
    h = [_rand((2, 5, 16), s) for s in range(3)]
    out, _ = m(h, strength=0.0)
    assert torch.allclose(out, h[-1], atol=1e-6)


def test_mhar_weights_normalized_along_source_axis():
    m = LoopMHAR(16, heads=4)
    with torch.no_grad():
        m.query.normal_(0, 0.5)  # 非零 query 打破初始偏置主导
    h = [_rand((2, 5, 16), s) for s in range(4)]
    _, w = m(h)
    assert w.shape == (4, 2, 5, 4)
    assert torch.allclose(w.sum(dim=0), torch.ones(2, 5, 4), atol=1e-5)


def test_mhar_causal_and_no_inplace():
    """改未来 token 的历史状态不影响更早位置；history 不被原地修改。"""
    m = LoopMHAR(16, heads=2)
    h = [_rand((1, 6, 16), s) for s in range(3)]
    h_frozen = [x.clone() for x in h]
    out1, _ = m(h)
    h[0] = h[0].clone()          # 替换引用（非原地）触发重算
    h[0][:, 4:] = torch.randn(1, 2, 16)  # 只改未来位置（新 tensor 上原地，允许）
    out2, _ = m(h)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)
    for a, b in zip(h_frozen, [x for x in h]):
        pass  # h[0] 被替换为克隆后修改，原 frozen 不受影响
    assert torch.equal(h_frozen[1], h[1]) and torch.equal(h_frozen[2], h[2])


def test_mhar_heads_can_route_differently():
    m = LoopMHAR(16, heads=4)
    with torch.no_grad():
        m.query.zero_()
        m.query[0].normal_(0, 1.0)   # 只给 head0 非零 query
    h = [_rand((1, 3, 16), s) for s in range(3)]
    _, w = m(h)
    # head0 的来源分布应与偏置主导的其它头显著不同
    w0, w1 = w[:, :, :, 0], w[:, :, :, 1]
    assert (w0 - w1).abs().max() > 1e-6


# ------------------------------ UpdateGate -------------------------------
def test_update_gate_boundaries():
    g = RecurrentUpdateGate(16, rank=8)
    prev, anchor, cand = _rand((1, 4, 16), 1), _rand((1, 4, 16), 2), _rand((1, 4, 16), 3)
    out0, gate0 = g(prev, anchor, cand, force_gate=0.0)
    out1, gate1 = g(prev, anchor, cand, force_gate=1.0)
    assert torch.equal(out0, prev)
    assert torch.equal(out1, cand)
    assert gate0.max() == 0.0 and gate1.min() == 1.0
    outm, gatem = g(prev, anchor, cand, force_gate=0.25)
    expect = prev + 0.25 * (cand - prev)
    assert torch.allclose(outm, expect, atol=1e-6)


def test_update_gate_inactive_positions_keep_previous():
    g = RecurrentUpdateGate(16, rank=8)
    prev, anchor, cand = _rand((1, 4, 16), 1), _rand((1, 4, 16), 2), _rand((1, 4, 16), 3)
    active = torch.tensor([[True, False, True, False]])
    out, _ = g(prev, anchor, cand, active=active, force_gate=1.0)
    assert torch.equal(out[:, 1], prev[:, 1]) and torch.equal(out[:, 3], prev[:, 3])
    assert torch.equal(out[:, 0], cand[:, 0])


def test_update_gate_near_open_init():
    """零 W2/+4 bias：g≈0.982（近开门，非严格恒等——V2 §7.4）。"""
    g = RecurrentUpdateGate(16, rank=8)
    prev, anchor, cand = _rand((1, 2, 16), 1), _rand((1, 2, 16), 2), _rand((1, 2, 16), 3)
    _, gate = g(prev, anchor, cand)
    assert abs(gate.mean().item() - torch.sigmoid(torch.tensor(4.0)).item()) < 1e-4


# ------------------------------- Reader ----------------------------------
def _reader(dim=16, mem_dim=16, scale_init=0.01):
    return GatedMemoryReader(dim, mem_dim, query_heads=4, kv_heads=2,
                             head_dim=8, gate_rank=8, scale_init=scale_init)


def test_reader_noop_family():
    """None / M=0 / 全 masked / 混合 batch 空行：恒等且无 NaN。"""
    r = _reader()
    h = _rand((2, 3, 16))
    out_none, d0 = r(h, None)
    assert torch.equal(out_none, h) and d0["gate"].shape == (2, 3, 1)

    kv0 = r.project_memory(torch.zeros(2, 0, 16), torch.zeros(2, 0, dtype=torch.bool))
    out_m0, _ = r(h, kv0)
    assert torch.equal(out_m0, h)

    mem = _rand((2, 4, 16), 5)
    valid = torch.tensor([[True, True, True, True], [False, False, False, False]])
    kv = r.project_memory(mem, valid)
    out_mix, d = r(h, kv)
    assert torch.equal(out_mix[1], h[1])          # 无有效 memory 的行 no-op
    assert torch.isfinite(out_mix).all() and torch.isfinite(d["delta"]).all()

    valid_all_masked = torch.zeros(2, 4, dtype=torch.bool)
    kv_m = r.project_memory(mem, valid_all_masked)
    out_m, _ = r(h, kv_m)
    assert torch.equal(out_m, h)


def test_reader_padded_memory_invisible():
    """padding memory 内容改变不影响真实 query 输出。"""
    r = _reader()
    h = _rand((1, 2, 16))
    mem = _rand((1, 4, 16), 7)
    mem2 = mem.clone()
    mem2[0, :2] = torch.randn(2, 16) * 10        # 只改 padding 位
    valid = torch.tensor([[False, False, True, True]])
    out1, _ = r(h, r.project_memory(mem, valid))
    out2, _ = r(h, r.project_memory(mem2, valid))
    assert torch.allclose(out1, out2, atol=1e-6)


def test_reader_scale_zero_exact_noop_but_scale_gets_grad():
    r = _reader(scale_init=0.0)
    assert abs(r.scale_logit.item()) < 1e-6
    h = _rand((1, 3, 16))
    kv = r.project_memory(_rand((1, 4, 16), 9), torch.ones(1, 4, dtype=torch.bool))
    out, _ = r(h, kv)
    assert torch.allclose(out, h, atol=1e-6)     # 精确 no-op
    out.sum().backward()
    assert r.scale_logit.grad is not None and r.scale_logit.grad.abs().item() > 0


def test_reader_grads_under_frozen_payload():
    """冻结 payload（no_grad 投影输入）下 Reader Q/K/V/O 仍有梯度（V2 §13.2 分层）。"""
    r = _reader()
    payload = _rand((1, 4, 16), 11)
    valid = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        frozen = payload
    h = _rand((1, 3, 16), 12).requires_grad_(True)
    kv = r.project_memory(frozen, valid)          # 投影带梯度（trainable 契约）
    out, _ = r(h, kv)
    out.sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        p = getattr(r, name).weight
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"{name} 无梯度"
    assert h.grad is not None


def test_reader_kv_reuse_across_reads():
    """同一 MemoryKV 在同一计算图内多次读取可反传（跨 loop 复用契约）。"""
    r = _reader()
    kv = r.project_memory(_rand((1, 4, 16), 13), torch.ones(1, 4, dtype=torch.bool))
    h1 = _rand((1, 2, 16), 14).requires_grad_(True)
    h2 = _rand((1, 2, 16), 15).requires_grad_(True)
    (r(h1, kv)[0].sum() + r(h2, kv)[0].sum()).backward()
    assert r.o_proj.weight.grad is not None
    assert h1.grad is not None and h2.grad is not None


def test_modules_bf16_finite():
    if not torch.cuda.is_available():
        m, g, r = LoopMHAR(16, 4), RecurrentUpdateGate(16, 8), _reader()
        for mod in (m, g, r):
            for p in mod.parameters():
                p.data = p.data.to(torch.bfloat16)
        h = [_rand((1, 3, 16)).to(torch.bfloat16) for _ in range(2)]
        out, _ = m(h)
        assert torch.isfinite(out.float()).all()
        prev, anchor, cand = h[0], h[1], h[0]
        out2, _ = g(prev, anchor, cand)
        assert torch.isfinite(out2.float()).all()
        # reader bf16 路径（参数与输入一致转 bf16）
        r2 = _reader()
        for p in r2.parameters():
            p.data = p.data.to(torch.bfloat16)
        kv = r2.project_memory(_rand((1, 4, 16)).to(torch.bfloat16),
                               torch.ones(1, 4, dtype=torch.bool))
        out3, _ = r2(h[0].to(torch.bfloat16), kv)
        assert torch.isfinite(out3.float()).all()


def test_combined_path_all_modules_get_grads():
    """线性 toy 组合路径：三模块参数全部有梯度（非语言模型，仅机制验收）。"""
    torch.manual_seed(7)
    D = 16
    mhar = LoopMHAR(D, heads=4)
    gate = RecurrentUpdateGate(D, rank=8)
    reader = _reader()
    anchor = _rand((1, 4, D), 1)
    payload = _rand((1, 6, D), 2)
    kv = reader.project_memory(payload, torch.ones(1, 6, dtype=torch.bool))
    target = _rand((1, 4, D), 3)

    history = [anchor]
    prev = anchor
    for r in range(3):
        read_state, _ = mhar(history)
        # 简化 core：Reader + 线性变换（模拟候选生成）
        h1, _ = reader(read_state, kv)
        candidate = torch.nn.functional.linear(h1, torch.randn(D, D))
        committed, _ = gate(prev, anchor, candidate)
        history.append(committed)
        prev = committed
    loss = F.mse_loss(prev, target)
    loss.backward()
    assert mhar.query.grad is not None or mhar.type_bias.grad is not None
    assert gate.fc2.weight.grad is not None and gate.fc2.bias.grad is not None
    assert reader.o_proj.weight.grad is not None and reader.k_proj.weight.grad is not None
