"""fine→coarse 检索蒸馏测试（v1.3 / ARCHITECTURE §10-A）：损失数值、梯度流向、SFT-C 接线。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.retrieval import LongContextStore


def _tiny(variant="minimind_x", **over):
    torch.manual_seed(11)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant=variant, block_size=8, topk_coarse=4, topk_fine=2)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_distill_loss_value_and_zero():
    m = _tiny()
    ret = m.model.retriever
    cand = torch.tensor([[1, 3, 5], [2, 4, 6]])
    fine = torch.randn(2, 3)
    # 构造 student：coarse 在候选位置上与 teacher 逐值相等 → 分布一致 → KL≈0
    coarse_aligned = torch.zeros(2, 10).scatter_(1, cand, fine)
    loss0 = ret.distill_loss(coarse_aligned, cand, fine)
    assert loss0.item() < 1e-5, loss0.item()
    # 反转分布（student = −teacher）→ KL 显著 > 0
    # （统一平移不变是 KL 的正确性质：蒸馏只学排序形状，不学绝对分值）
    coarse_mis = torch.zeros(2, 10).scatter_(1, cand, -fine)
    loss1 = ret.distill_loss(coarse_mis, cand, fine)
    assert loss1.item() > 0.1
    assert torch.isfinite(loss1)


def test_distill_gradients_flow_to_query_proj():
    """蒸馏的梯度必须真实进入 query_proj（评审 P0-1 的有效解法验证）。"""
    m = _tiny()
    # 建库
    ctx = torch.randint(3, 64, (m.config.block_size * 6,))
    store = LongContextStore(ctx, m.config.block_size)
    store.build_keys(m.model.retriever, m.model.embed_tokens)
    ids = torch.randint(3, 64, (1, 5))
    q_hidden = m.model.prelude_hidden(ids, no_grad=True)
    with torch.no_grad():  # teacher：no_grad 检索
        result = m.model.retriever.retrieve(q_hidden, store, m)
    # student：带梯度的 query 重打分
    q_grad = m.model.prelude_hidden(ids, no_grad=False)
    q_vec = m.model.retriever.project_query(q_grad)
    coarse_trainable = m.model.retriever.coarse_scores(q_vec, store)
    loss = m.model.retriever.distill_loss(coarse_trainable, result.cand_idx,
                                          result.fine_scores)
    loss.backward()
    g = m.model.retriever.query_proj.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "蒸馏梯度未进入 query_proj"
    # teacher 侧确实被 detach（fine_scores 无梯度）
    assert result.fine_scores.requires_grad is False


def test_controller_forward_produces_distill():
    from model.minimind_x.controller import AdaptiveController
    m = _tiny()
    ctrl = AdaptiveController(m)
    ids = torch.randint(3, 64, (2, 6))
    labels = torch.randint(3, 64, (2, 6))
    ctx = torch.randint(3, 64, (m.config.block_size * 4,))
    out, extras = ctrl.forward_with_oracles(ids, None, labels, ctx)  # 无 gold：纯蒸馏监督
    assert "distill_loss" in extras and torch.isfinite(extras["distill_loss"])
    assert extras["distill_loss"].requires_grad  # 训练侧拿到的是带梯度损失
    total = sum(l.mean() for l in out.exit_logits) * 0 + extras["distill_loss"]
    total.backward()
    assert m.model.retriever.query_proj.weight.grad is not None


def test_sft_c_batch_guard():
    """评审 Bug 1 回归：B>1 必须显式报错而非静默丢上下文。"""
    import pytest
    from trainer.train_sft_x import assert_sft_c_batch
    assert_sft_c_batch(torch.zeros(1, 8, dtype=torch.long))  # B=1 通过
    with pytest.raises(ValueError, match="batch>1"):
        assert_sft_c_batch(torch.zeros(2, 8, dtype=torch.long))
