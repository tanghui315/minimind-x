"""FLOPs / 参数量解析测试（任务清单 T0.5）。"""
from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, count_params, estimate_flops


def _cfg(**over):
    kw = dict(hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
              num_key_value_heads=2, vocab_size=128, flash_attn=False,
              variant="middle_loop")
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_count_params_matches_actual():
    cfg = _cfg()
    model = MiniMindXForCausalLM(cfg)
    actual = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("lm_head", "model.halt_router", "model.retrieve_router", "model.budget_router", "model.memory_manager", "model.retriever")))
    analytic = count_params(cfg)
    assert abs(analytic - actual) / actual < 0.01


def test_flops_scaling():
    cfg = _cfg()
    f1 = estimate_flops(cfg, seq_len=512, loops=1)
    f2 = estimate_flops(cfg, seq_len=512, loops=2)
    f4 = estimate_flops(cfg, seq_len=512, loops=4)
    # 有效深度 = P + C + T*R = 3+2+3R；主干 FLOPs 线性于 R（增量 = 每轮 T 层）
    assert f1["effective_depth"] == 8 and f4["effective_depth"] == 17
    d1, d2 = f2["total"] - f1["total"], f4["total"] - f2["total"]
    assert abs(d1 * 2 - d2) / d2 < 1e-6  # R1→R2 增 1 轮，R2→R4 增 2 轮
    assert f1["memory_attn"] == 0.0 and f1["retrieval"] == 0.0
    fr = estimate_flops(cfg, seq_len=512, loops=1, memory_blocks=4, n_retrievals=2)
    assert fr["memory_attn"] > 0 and fr["retrieval"] > 0
    ft = estimate_flops(cfg, seq_len=512, loops=1, train=True)
    assert abs(ft["total"] - 3 * f1["total"]) < 1.0


def test_dense_variant_flops_equals_depth():
    cfg = _cfg(variant="dense")
    f = estimate_flops(cfg, seq_len=256, loops=4)  # dense 不循环：loops 被忽略于深度语义外
    assert f["effective_depth"] == cfg.num_hidden_layers
