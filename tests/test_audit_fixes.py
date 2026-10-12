"""收口审查修复：init 冻结 key / refresh 失效 warm / 空库 sft_c。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              freeze_key=True, key_refresh_interval=1)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_freeze_key_at_model_init():
    """不依赖 optimizer：构造即 requires_grad=False，仍在 state_dict。"""
    torch.manual_seed(80)
    m = MiniMindXForCausalLM(_cfg())
    assert m.model.retriever.key_proj.weight.requires_grad is False
    assert any("key_proj" in k for k in m.state_dict())
    # freeze_key=False 时保持可训
    m2 = MiniMindXForCausalLM(_cfg(freeze_key=False))
    assert m2.model.retriever.key_proj.weight.requires_grad is True


def test_refresh_index_invalidates_warm_and_last_global():
    torch.manual_seed(81)
    m = MiniMindXForCausalLM(_cfg()).eval()
    ctrl = AdaptiveController(m)
    ctx = torch.randint(3, 60, (32,))
    ids = torch.randint(3, 60, (1, 6))
    ctrl.begin(ctx)
    q = m.model.prelude_hidden(ids[:, :3], None)
    ctrl.retrieve_into_memory(q)
    assert ctrl.has_warm_cache() is True
    with torch.no_grad():
        m.model.retriever.key_proj.weight.mul_(1.7)
    ctrl.refresh_index()
    assert ctrl.has_warm_cache() is False, "refresh 后 warm_candidates 必须失效"
    assert ctrl.stats.get("last_global_result") is None


def test_sft_c_empty_and_1d_context():
    """no_evidence：context_ids 为空 [0] 或 [1,0] 均可训练。"""
    import trainer.train_sft_x as tsx
    torch.manual_seed(82)
    m = MiniMindXForCausalLM(_cfg()).train()
    cfg = m.config
    tsx.args = type("A", (), {})(); a = tsx.args
    a.device = "cpu"; a.oracle_rate = 0.0; a.train_halt_router = 1
    a.train_budget_router = 1; a.lambda_distill = 0.05; a.lm_config = cfg
    a.step_global = 0
    tsx.lm_config = cfg; tsx.model = m
    for ctx in (torch.zeros(0, dtype=torch.long), torch.zeros(1, 0, dtype=torch.long)):
        batch = {"input_ids": torch.randint(3, 60, (1, 8)),
                 "labels": torch.randint(3, 60, (1, 8)),
                 "question_len": torch.tensor([4]),
                 "context_ids": ctx,
                 "gold_block": torch.tensor([-1]),
                 "oracle_r": torch.tensor([1]),
                 "oracle_retrieve": torch.tensor([0]),
                 "action_schema_version": 2, "answer_text": "x"}
        loss, info = tsx.sft_c_step(batch)
        assert torch.isfinite(loss)
        loss.backward()
        m.zero_grad(set_to_none=True)
