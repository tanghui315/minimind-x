"""Memory 管理测试（任务清单 T5.1/T5.2 验收）：驱逐不变量、打分、跨 loop KV 缓存。"""
import torch

from model.minimind_x import MiniMindXConfig
from model.minimind_x.memory import WorkingMemory, MemoryManager


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_manager_budget_invariant():
    cfg = _cfg()
    mgr = MemoryManager(cfg)
    B, d, bs = 2, cfg.hidden_size, cfg.block_size
    mem = WorkingMemory(torch.randn(B, 0, bs, d), torch.randn(B, 0, d))
    q = torch.randn(B, d)
    for step in range(5):
        new_h = torch.randn(B, 3, bs, d)
        new_k = torch.randn(B, 3, d)
        mem, _, _ = mgr.update(mem, new_h, new_k, q, budget_blocks=4)
        assert mem.size <= 4  # 驱逐不变量：容量永不超过预算
    assert mem.token_size == 4 * bs


def test_manager_evicts_low_relevance():
    """相关性主导时（β=γ=δ=0），与 query 无关的块应先被驱逐。"""
    cfg = _cfg(mm_beta=0.0, mm_gamma=0.0, mm_delta=0.0, mm_eta=0.0)
    mgr = MemoryManager(cfg)
    B, d, bs = 1, cfg.hidden_size, cfg.block_size
    q = torch.tensor([[1.0] + [0.0] * (d - 1)])
    keys = torch.tensor([[[1.0] + [0.0] * (d - 1),      # 与 query 同向，应保留
                          [0.0, 1.0] + [0.0] * (d - 2),  # 正交，应被驱逐
                          [-1.0] + [0.0] * (d - 1)]])    # 反向，应被驱逐
    hidden = torch.randn(B, 3, bs, d)
    mem = WorkingMemory(hidden, keys)
    mem, kept_new, u = mgr.update(mem, torch.randn(B, 2, bs, d),
                                  torch.randn(B, 2, d), q, budget_blocks=3)
    kept_keys = mem.keys[0]
    cos_q = torch.nn.functional.cosine_similarity(kept_keys, q.expand_as(kept_keys), dim=-1)
    assert cos_q.argmax().item() == 0  # 相关块仍在（必然保留在 top-3）
    assert mem.size == 3


def test_working_memory_kv_cache():
    from model.minimind_x import MiniMindXForCausalLM
    cfg = _cfg()
    model = MiniMindXForCausalLM(cfg).eval()
    B, K, bs, d = 2, 3, cfg.block_size, cfg.hidden_size
    mem = WorkingMemory(torch.randn(B, K, bs, d), torch.randn(B, K, d), block_size=bs)
    kv = mem.get_kv(model.model.thinking_core)
    assert len(kv) == cfg.thinking_layers
    for k, v in kv:
        assert k.shape == (B, K * bs, cfg.num_key_value_heads, cfg.head_dim)
        assert v.shape == k.shape
    assert mem.get_kv(model.model.thinking_core) is not None
    # 内容变更（add）后缓存失效重投影
    mem2 = mem.with_added(torch.randn(B, 1, bs, d), torch.randn(B, 1, d))
    assert mem2._kv_cache is None
    kv2 = mem2.get_kv(model.model.thinking_core)
    assert kv2[0][0].shape[1] == (K + 1) * bs


def test_usage_and_persistence_ema():
    cfg = _cfg()
    B, K, bs, d = 1, 2, cfg.block_size, cfg.hidden_size
    mem = WorkingMemory(torch.randn(B, K, bs, d), torch.randn(B, K, d))
    hit = torch.tensor([[1.0, 0.0]])
    mem = mem.with_usage_bumped(hit, decay=0.5)
    assert abs(mem.usage[0, 0].item() - 0.5) < 1e-6 and mem.usage[0, 1].item() == 0.0
    scores = torch.tensor([[1.0, 1.0]])
    mem = mem.with_persistence_ema(scores, decay=0.5)
    assert abs(mem.persistence[0, 0].item() - 0.5) < 1e-6
