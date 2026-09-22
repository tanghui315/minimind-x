"""Loop KV 语义测试（任务清单横切；D2）：decode/prefill 一致性、循环不追加 KV、三模式 generate。"""
import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def _tiny_model(**cfg_over):
    torch.manual_seed(3)
    cfg_kw = dict(hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=128, max_position_embeddings=512,
                  flash_attn=False, max_loop=4, variant="middle_loop")
    cfg_kw.update(cfg_over)
    cfg = MiniMindXConfig(**cfg_kw)
    return MiniMindXForCausalLM(cfg).eval()


def test_decode_prefill_consistency_fixed_loops():
    """逐 token decode（带 cache）与整段 forward 的 next-token logits 一致（D2 回归）。"""
    m = _tiny_model()
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        full = m(ids, num_loops=3).logits[:, -1]
        # 增量：先 prefill 前 6 个，再逐步喂
        out = m(ids[:, :6], use_cache=True, num_loops=3)
        past = out.past_key_values
        for t in range(6, 12):
            out = m(ids[:, t:t + 1], past_key_values=past, use_cache=True, num_loops=3)
            past = out.past_key_values
        inc = out.logits[:, -1]
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_cache_grows_one_per_token_regardless_of_loops():
    m = _tiny_model()
    ids = torch.randint(0, 128, (1, 4))
    with torch.no_grad():
        out = m(ids[:, :2], use_cache=True, num_loops=4)
        assert out.past_key_values[0][0].shape[1] == 2
        out = m(ids[:, 2:3], past_key_values=out.past_key_values, use_cache=True, num_loops=4)
        assert out.past_key_values[0][0].shape[1] == 3  # 4 轮循环仍只 +1


def test_exit_choice_selects_loop():
    """显式 exit_choice=r 时，logits 应等于「只跑 r 轮」的前向（采样 R 的实现基础）。"""
    m = _tiny_model()
    ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        r2 = m(ids, num_loops=2).logits
        r4_pick2 = m(ids, num_loops=4, exit_choice=torch.tensor([1, 1])).logits
    assert torch.allclose(r2, r4_pick2, atol=1e-4), (r2 - r4_pick2).abs().max()


def test_halt_early_stop_matches_fixed_exit():
    """推理早停：把 halt_router 输出钉在第 2 轮 stop，结果应等于 exit_choice=2。"""
    m = _tiny_model()
    # 钉住 halt router：权重置 0、bias 大 → 恒 P(stop)>threshold（第 1 轮即停）
    with torch.no_grad():
        torch.nn.init.zeros_(m.model.halt_router.proj.weight)
        m.model.halt_router.proj.bias.fill_(10.0)
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        halted = m(ids, num_loops=4, halt=True).logits
        fixed1 = m(ids, num_loops=4, exit_choice=torch.zeros(2, dtype=torch.long)).logits
        one = m(ids, num_loops=1).logits
    assert torch.allclose(halted, one, atol=1e-4)
    assert torch.allclose(halted, fixed1, atol=1e-4)


def test_generate_modes_and_stats():
    m = _tiny_model()
    ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        out = m.generate(inputs=ids, max_new_tokens=4, do_sample=False, eos_token_id=None,
                         mode="fast", return_stats=True)
        assert out["generated_ids"].shape[1] == 10
        assert out["stats"]["avg_loops"] <= 1.0 + 1e-6  # fast 封顶 1 轮
        out2 = m.generate(inputs=ids, max_new_tokens=4, do_sample=False, eos_token_id=None,
                          fixed_loops=3, return_stats=True)
        assert abs(out2["stats"]["avg_loops"] - 3.0) < 1e-6


def test_memory_changes_logits():
    """Working Memory 注入会改变输出（cross-attention 生效 sanity）。"""
    m = _tiny_model(variant="minimind_x", block_size=8)
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        base = m(ids, num_loops=2).logits
        mem_hidden = torch.randn(2, 3, m.config.hidden_size)
        mem_kv = m.model.thinking_core.project_memory(mem_hidden)
        with_mem = m(ids, num_loops=2, memory_kv_per_layer=mem_kv).logits
    assert not torch.allclose(base, with_mem, atol=1e-3)
    assert with_mem.shape == base.shape


def test_loop_embedding_forward():
    m = _tiny_model(loop_embedding=True)
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        out = m(ids, num_loops=2, collect_exits=True)
    assert len(out.exit_logits) == 2
