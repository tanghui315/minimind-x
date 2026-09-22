"""R=1 等价性测试（任务清单 T0.5 / 横切）：同权重下 MiniMindX 与原 MiniMind 前向数值一致。

这是 D1「不改原模型、权重可互转」和 D2 KV 语义的回归基线。
注：transformers>=5 在 torch<2.4 下禁止实例化 PreTrainedModel 子类，
参照侧用未封装的 MiniMindModel + lm_head 手工组装（数值路径与 MiniMindForCausalLM 相同）。
"""
import torch
from torch import nn

from model.model_minimind import MiniMindConfig, MiniMindModel
from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, remap_minimind_state_dict


class ReferenceMiniMind(nn.Module):
    """原 MiniMindForCausalLM 的去封装等价物（权重键名一致）。"""

    def __init__(self, cfg: MiniMindConfig):
        super().__init__()
        self.model = MiniMindModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids):
        hidden, _, _ = self.model(input_ids)
        return self.lm_head(hidden)


def _tiny_cfg_pair(**x_over):
    base = dict(hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
                num_key_value_heads=2, vocab_size=128, max_position_embeddings=512,
                flash_attn=False)
    base.update(x_over)
    orig = MiniMindConfig(**base)
    xcfg = MiniMindXConfig(variant="dense", use_loop=False, **base)
    return orig, xcfg


def test_r1_logits_equivalence():
    torch.manual_seed(0)
    orig_cfg, x_cfg = _tiny_cfg_pair()
    orig = ReferenceMiniMind(orig_cfg).eval()
    x = MiniMindXForCausalLM(x_cfg).eval()
    # 把原模型权重 remap 进 X（router/retriever 等新模块 strict=False）
    mapped = remap_minimind_state_dict(orig.state_dict(), x_cfg)
    missing, unexpected = x.load_state_dict(mapped, strict=False)
    assert not unexpected
    assert all(k.startswith(("model.halt_router", "model.retrieve_router",
                             "model.budget_router", "model.memory_manager",
                             "model.retriever")) for k in missing)
    input_ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        lo = orig(input_ids)
        lx = x(input_ids, num_loops=1).logits
    assert torch.allclose(lo, lx, atol=1e-4), (lo - lx).abs().max()


def test_r1_with_use_loop_variant_equivalence():
    """use_loop=True 但 num_loops=1 时同样等价（zone 循环跑 1 轮 = 直通）。"""
    torch.manual_seed(1)
    orig_cfg, _ = _tiny_cfg_pair()
    x_cfg = MiniMindXConfig(variant="middle_loop", **dict(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128, max_position_embeddings=512, flash_attn=False))
    orig = ReferenceMiniMind(orig_cfg).eval()
    x = MiniMindXForCausalLM(x_cfg).eval()
    x.load_state_dict(remap_minimind_state_dict(orig.state_dict(), x_cfg), strict=False)
    input_ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        lo = orig(input_ids)
        lx = x(input_ids, num_loops=1).logits
    assert torch.allclose(lo, lx, atol=1e-4)


def test_collect_exits_shapes_and_kv_cache():
    """全展开 collect_exits 的形状 + use_cache 的 KV 长度（D2：循环不追加 KV）。"""
    torch.manual_seed(2)
    x_cfg = MiniMindXConfig(variant="middle_loop", max_loop=4, **dict(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128, max_position_embeddings=512, flash_attn=False))
    x = MiniMindXForCausalLM(x_cfg).eval()
    input_ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        out = x(input_ids, collect_exits=True, num_loops=3)
        assert len(out.exit_logits) == 3
        assert all(lg.shape == out.logits.shape for lg in out.exit_logits)
        out2 = x(input_ids, num_loops=4, use_cache=True)
        kv = out2.past_key_values
        assert len(kv) == 8  # 总层数
        for past in kv:
            if isinstance(past, list):
                # core 层：轮次轴 KV —— 4 轮 × 每轮长度 12（token 轴不随轮数膨胀）
                assert len(past) == 4 and all(p[0].shape[1] == 12 for p in past)
            else:
                assert past is not None and past[0].shape[1] == 12
