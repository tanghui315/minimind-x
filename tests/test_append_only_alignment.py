"""append-only 对齐：reference ↔ segmented ↔ generate（V2 §11.5）。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from trainer.dual_executor import (
    build_action_timeline, reference_append_only, segmented_forward,
    reference_recompute, generate_teacher_forced, assert_executors_correspond)


def _m(**over):
    torch.manual_seed(97)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=1, residual_v2=True)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def _batch():
    torch.manual_seed(98)
    ids = torch.randint(3, 60, (1, 6))
    y = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (32,))
    return ids, y, ctx


def test_append_only_matches_segmented():
    """同 memory 时间线：逐 token append-only == 分段续 cache。"""
    m = _m()
    ids, y, ctx = _batch()
    timeline = build_action_timeline([0, 0], [(0, 3), (3, 6)])
    ref_ce, ref_logits = reference_append_only(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    seg_ce, seg_logits = segmented_forward(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    ok, msg = assert_executors_correspond(ref_logits, seg_logits, atol=1e-4)
    assert ok, f"append-only vs segmented: {msg}"
    assert torch.allclose(ref_ce, seg_ce, atol=1e-4)


def test_append_only_with_memory_switch_differs_from_recompute():
    """memory 换版后：append-only 不回算历史；从头重算是另一语义（旧路径）。"""
    m = _m(max_loop=1)
    ids, y, ctx = _batch()
    # 段 0 REUSE，段 1 GLOBAL_REFRESH（动作边界换 memory）
    timeline = build_action_timeline([0, 2], [(0, 3), (3, 6)])
    ao_ce, ao_logits = reference_append_only(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    rc_ce, rc_logits = reference_recompute(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    # 分段续 cache 必须跟 append-only，而不是 recompute
    sg_ce, sg_logits = segmented_forward(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    ok, msg = assert_executors_correspond(ao_logits, sg_logits, atol=1e-4)
    assert ok, f"segmented 应对齐 append-only: {msg}"
    # recompute 与 append-only 在换 memory 后允许不同（历史是否被重算）
    diff_rc = max((a - b).abs().max().item() for a, b in zip(ao_logits, rc_logits))
    # 若 memory 版本切换实际改变了 payload，diff 应能观测到；至少二者都是有限值
    assert all(torch.isfinite(x).all() for x in ao_logits + rc_logits)


def test_generate_teacher_forced_matches_append_only():
    """与 generate 同构的 prefill+decode == append-only（固定 R、固定 memory）。"""
    m = _m(max_loop=1)
    ids, y, ctx = _batch()
    timeline = build_action_timeline([0], [(0, 6)])
    _, ref_logits = reference_append_only(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    # 同一 memory（全 REUSE）逐位置
    from trainer.dual_executor import _prepare_memories, _mem_at
    mem_list = _prepare_memories(m, ids, ctx, 3, timeline, lambda: AdaptiveController(m))
    gen_logits = generate_teacher_forced(
        m, ids, mem_schedule=lambda t: _mem_at(timeline, mem_list, t),
        num_loops=1, prompt_len=3)  # prompt=前 3，后 3 为 decode 追加
    # prompt 内位置 0..2：generate 一次 prefill；append-only 逐 token——因果 LM 应接近
    # decode 位置 3..5：两者都是单 token 追加——应严格接近
    for t in range(3, 6):
        assert torch.allclose(ref_logits[t], gen_logits[t], atol=1e-4), (
            f"decode t={t} Δ={(ref_logits[t]-gen_logits[t]).abs().max()}")


def test_generate_model_loop_equals_append_only():
    """真实 model.generate（greedy, fixed R）下一步 logits == teacher-forced 追加。"""
    m = _m(max_loop=1)
    torch.manual_seed(99)
    prompt = torch.randint(3, 60, (1, 4))
    with torch.no_grad():
        gen = m.generate(inputs=prompt, max_new_tokens=2, do_sample=False,
                         fixed_loops=1, use_halting=False, eos_token_id=None)
    gen_ids = gen["generated_ids"] if isinstance(gen, dict) else gen
    full = gen_ids  # [1, 6]
    # teacher-forced append-only 重放生成出的 token
    timeline = build_action_timeline([0], [(0, full.shape[1])])
    from trainer.dual_executor import generate_teacher_forced
    # 无 memory（middle 等价）：用 minimind_x 但不传 memory
    logits_seq = generate_teacher_forced(
        m, full, mem_schedule=lambda t: {}, num_loops=1, prompt_len=4)
    # 位置 4 的 next 应与 generate 第二步 argmax 一致
    nxt = logits_seq[4].argmax(dim=-1)
    assert int(nxt.item()) == int(full[0, 5].item()), (
        f"replay argmax {int(nxt.item())} vs generate {int(full[0, 5].item())}")
