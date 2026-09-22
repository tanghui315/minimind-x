"""Frozen Index 范式声明（评审 8.5/10 收口）：key 冻结、query 可训。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def test_freeze_key_default_and_manifest():
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", block_size=8)
    assert cfg.freeze_key is True
    assert cfg.key_refresh_interval == 0
    assert cfg.validate() is True


def test_key_proj_frozen_query_trainable():
    """优化器视角：key_proj.requires_grad=False；query_proj 仍可训。"""
    from trainer.x_utils import build_optimizer, param_manifest
    torch.manual_seed(50)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", block_size=8, freeze_key=True)
    m = MiniMindXForCausalLM(cfg).train()
    opt = build_optimizer(m, lr=1e-3)
    assert m.model.retriever.key_proj.weight.requires_grad is False
    assert m.model.retriever.query_proj.weight.requires_grad is True
    man = param_manifest(m)
    assert man["freeze_key"] is True
    assert man["frozen_key_params"] > 0
    # key_proj 不进优化器参数组
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_all_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_opt == n_all_train


def test_retrieval_ce_only_query_grad():
    """CE 路径：query 有 grad，key 无 grad（Frozen Index 语义）。"""
    torch.manual_seed(51)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", block_size=8, freeze_key=True)
    m = MiniMindXForCausalLM(cfg).train()
    q = torch.randn(1, 4, 32, requires_grad=True)
    # 建库（no_grad，与生产一致）
    ctx = torch.randint(3, 60, (32,))
    store = __import__("model.minimind_x.retrieval", fromlist=["LongContextStore"]).LongContextStore(ctx, 8)
    store.build_keys(m.model.retriever, m.model.embed_tokens)
    qv = m.model.retriever.project_query(q)
    scores = m.model.retriever.coarse_scores(qv, store)
    loss = torch.nn.functional.cross_entropy(scores, torch.tensor([0]))
    loss.backward()
    assert m.model.retriever.query_proj.weight.grad is not None
    assert m.model.retriever.query_proj.weight.grad.abs().sum() > 0
    assert m.model.retriever.key_proj.weight.grad is None
