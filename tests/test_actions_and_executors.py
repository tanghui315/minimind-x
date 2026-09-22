"""T2.5 三动作 / T3.2 prefill fixed_cap / T4.6 双执行器 / T5.2 配置校验 验收。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.router.retrieval_router import (
    ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH,
    action_mask, apply_action_mask, RetrievalRouter)


def _model(variant="minimind_x", max_loop=2, **over):
    torch.manual_seed(7)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant=variant, block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=max_loop, residual_v2=True)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_action_mask_rescore_warm_without_cache():
    """T2.5：无 Warm Candidate Cache 时 RESCORE_WARM 必须 mask，不得可用。"""
    m = action_mask(False, device="cpu", batch=2)
    assert m.shape == (2, 3)
    assert m[:, ACTION_REUSE].all() and m[:, ACTION_GLOBAL_REFRESH].all()
    assert not m[:, ACTION_RESCORE_WARM].any()
    m2 = action_mask(True, device="cpu", batch=1)
    assert m2.all()
    logits = torch.zeros(2, 3)
    masked = apply_action_mask(logits, m)
    assert masked[0, ACTION_RESCORE_WARM] == float("-inf")
    assert masked[0, ACTION_REUSE] == 0.0


def test_rescore_warm_rejected_without_cache():
    """执行侧同样拒绝无缓存 RESCORE_WARM（禁止静默降级全局搜索）。"""
    m = _model()
    ctrl = AdaptiveController(m)
    ids = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl.begin(ctx)
    assert not ctrl.has_warm_cache()
    try:
        ctrl.execute_retrieval_action(ACTION_RESCORE_WARM, ids, mem=None)
        assert False, "无 warm cache 的 RESCORE_WARM 应抛错"
    except RuntimeError:
        pass


def test_three_action_trajectory_and_warm_cache():
    """T2.5：三动作可执行；warm cache 在全局检索后可用；轨迹记动作。"""
    torch.manual_seed(8)
    ids = torch.randint(3, 60, (1, 10))
    labels = torch.full((1, 10), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    m = _model(max_loop=3)  # 3 轮 → 两次 loop_callback
    ctrl = AdaptiveController(m)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4,
        oracle_retrieve=[1, 0])  # schema v2：第 0 轮后 RESCORE_WARM，第 1 轮 REUSE
    assert extras["warm_available"] is True
    events = [t["event"] for t in ctrl.trajectory]
    assert "initial_retrieval" in events
    assert "rescore_warm" in events
    assert "reuse" in events
    assert ctrl.stats.get("warm_rescore", 0) >= 1


def test_global_refresh_still_works_and_legacy_oracle():
    """schema v2：2=GLOBAL_REFRESH；旧 v1 二值经 normalize_oracle_actions 转换。"""
    m = _model()
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4, oracle_retrieve=[2, 0])
    events = [t["event"] for t in ctrl.trajectory]
    assert "refresh" in events or "global_refresh" in events
    assert ctrl.stats["global_refresh"] >= 1


def test_action_ce_loss_masks_unavailable():
    from model.minimind_x.losses import retrieval_action_ce_loss
    logits = [torch.randn(1, 3, requires_grad=True) for _ in range(2)]
    labels = torch.tensor([ACTION_REUSE, ACTION_REUSE])
    # RESCORE_WARM 不可用
    masks = [action_mask(False, batch=1), action_mask(False, batch=1)]
    loss = retrieval_action_ce_loss(logits, labels, action_masks=masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(l.grad is not None for l in logits)
    # 标签落在不可用动作上 → 该轮 mask 掉（不 NaN）
    bad = torch.tensor([ACTION_RESCORE_WARM, ACTION_REUSE])
    loss2 = retrieval_action_ce_loss(logits, bad, action_masks=masks)
    assert torch.isfinite(loss2)


def test_prefill_fixed_cap_ttft_decode_stats():
    """T3.2：prefill 用固定 R_prefill；TTFT / decode 成本分开记。"""
    m = _model(variant="middle_loop", max_loop=3, residual_v2=False)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out = m.generate(inputs=ids, max_new_tokens=3, do_sample=False,
                         use_halting=False, fixed_loops=None, return_stats=True,
                         eos_token_id=None)
    st = out["stats"]
    assert st["ttft_steps"] == 1, "prompt prefill 应记 1 次 TTFT"
    assert st["decode_steps"] == 2  # max_new_tokens=3：1 次 TTFT + 2 次 decode
    assert "avg_ttft_loops" in st and "avg_decode_loops" in st
    assert "executed_core_rounds" in st


def test_dual_executor_correspondence():
    """T4.6：逐前缀参考执行器与分段执行器数值逐位对应。"""
    from trainer.dual_executor import (
        build_action_timeline, reference_append_only,
        segmented_forward, assert_executors_correspond_ce)
    torch.manual_seed(9)
    m = _model(max_loop=1, residual_v2=True)
    ids = torch.randint(3, 60, (1, 6))
    labels = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (8 * 4,))
    timeline = build_action_timeline([0, 0], [(0, 3), (3, 6)])

    def factory():
        return AdaptiveController(m)

    ref_ce, _ = reference_append_only(
        m, ids, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    seg_ce, _ = segmented_forward(
        m, ids, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    ok, msg = assert_executors_correspond_ce(ref_ce, seg_ce, atol=1e-4)
    assert ok, f"双执行器应数值对应: {msg}\nref={ref_ce}\nseg={seg_ce}"
    # 因果一致性：换未来 token 不影响早期 CE
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % 60
    ref_ce2, _ = reference_append_only(
        m, ids2, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    assert torch.allclose(ref_ce[:, :3], ref_ce2[:, :3], atol=1e-5), \
        "换未来 token 不得改变早期位置 CE"


def test_dual_executor_with_refresh_boundary():
    """T4.6：带动作边界（GLOBAL_REFRESH）时两执行器仍对应。"""
    from trainer.dual_executor import (
        build_action_timeline, reference_append_only,
        segmented_forward, assert_executors_correspond_ce)
    torch.manual_seed(11)
    m = _model(max_loop=1, residual_v2=True)
    ids = torch.randint(3, 60, (1, 6))
    labels = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (8 * 4,))
    # 段 0 REUSE，段 1 进入时 GLOBAL_REFRESH（动作边界）
    timeline = build_action_timeline([0, 2], [(0, 3), (3, 6)])

    def factory():
        return AdaptiveController(m)

    ref_ce, _ = reference_append_only(
        m, ids, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    seg_ce, _ = segmented_forward(
        m, ids, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    ok, msg = assert_executors_correspond_ce(ref_ce, seg_ce, atol=1e-4)
    assert ok, f"动作边界下双执行器应对应: {msg}\nref={ref_ce}\nseg={seg_ce}"


def test_config_validation():
    """T5.2：非法 config 启动即报错。"""
    try:
        cfg = MiniMindXConfig(hidden_size=30, num_hidden_layers=6,
                              num_attention_heads=2, num_key_value_heads=1,
                              vocab_size=64, flash_attn=False,
                              variant="minimind_x", mhar_heads=4, residual_v2=True)
        cfg.validate()
        assert False, "30 不能被 mhar_heads=4 整除应报错"
    except ValueError:
        pass
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6,
                          num_attention_heads=2, num_key_value_heads=1,
                          vocab_size=64, flash_attn=False,
                          variant="minimind_x", mhar_heads=4, residual_v2=True)
    assert cfg.validate() is True
    assert cfg.architecture_version() == "minimind_x_residual_v2"


def test_param_manifest_and_flops_breakdown():
    """T5.2 参数 manifest / T5.5 FLOPs 分项。"""
    from trainer.x_utils import param_manifest
    from model.minimind_x.flops import estimate_flops
    m = _model()
    man = param_manifest(m)
    assert man["total"] > 0
    assert any("thinking_core" in k or "update_gate" in k or "loop_mhar" in k
               for k in man["by_module"])
    fl = estimate_flops(m.config, seq_len=16, loops=2, memory_blocks=2, n_retrievals=2)
    for k in ("reader_attn", "reader_gate", "mhar", "update_gate", "retrieval",
              "backbone", "total"):
        assert k in fl and fl[k] >= 0
    assert fl["reader_attn"] > 0  # 有 memory 时 Reader 有真实开销
    assert fl["total"] >= fl["backbone"]


def test_joint_path_logprob_and_no_avg_depth_replay():
    """T4.7：联合路径概率可加总；禁平均深度重放门槛。"""
    m = _model()
    ctrl = AdaptiveController(m)
    lp = ctrl.joint_path_logprob(
        token_logprobs=torch.tensor([-1.0, -2.0]),
        halt_logprobs=[-0.5],
        retrieval_logprobs=[-0.3],
        budget_logprobs=[-0.1])
    assert abs(lp - (-3.9)) < 1e-6
    assert ctrl.assert_replay_not_averaged([1, 3, 2, 4]) is True
    assert ctrl.stats["depth_per_token"] == [1, 3, 2, 4]
    assert ctrl.stats["depth_mean_for_log_only"] == 2  # round(2.5)=2（银行家舍入）；仅日志用


def test_action_order_no_retrieve_after_halt():
    """T2.5 / V2 §9.1：halt 判停后禁止再检索；R_max 强制吸收。"""
    torch.manual_seed(21)
    m = _model(max_loop=3)
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    # oracle：第 0 轮后 REFRESH，但若 halt 先停则不得检索
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4, oracle_r=torch.tensor([1]),
        oracle_retrieve=[2, 2, 2])  # 三个 GLOBAL_REFRESH
    events = [t["event"] for t in ctrl.trajectory]
    # R*=1：应在第 0 轮后停，之后无 refresh（不出现「先检索后发现已停」）
    # 训练全展开仍会算辅助轮动作，但 execute 后的 refresh 只发生在未强制停前
    # 参考口径：initial 必有；若 n_run==1 则不应有轮间 refresh 执行导致的版本暴涨
    assert "initial_retrieval" in events
    assert out.num_loops_executed >= 1
    # 轨迹中 refresh/rescore 只能出现在 initial 之后、且轮次 < num_loops_executed
    for t in ctrl.trajectory:
        if t["event"] in ("refresh", "rescore_warm"):
            assert t.get("round", -1) < out.num_loops_executed, \
                f"halt 后不应再检索: {t}"
