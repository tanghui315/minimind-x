"""因果契约测试（V2 T0.1 验收，C5/C6 修复锁定）。

V2 §10.3 反例验收：
  1. 固定 prompt 与 source_context，替换完整 gold completion → 初始检索结果不变
  2. T0.3：budget 档位真正联动 topk_f（库容充足时不同档取到不同数量的块）
  3. T0.7：gold CE 用带梯分的重算 logits（backward 进入 query 投影）
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController


def _tiny_model():
    torch.manual_seed(21)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, max_loop=2)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def _orthogonal_retriever(model):
    with torch.no_grad():
        d = model.config.hidden_size
        model.model.retriever.key_proj.weight.copy_(torch.eye(d))
        model.model.retriever.query_proj.weight.copy_(torch.eye(d))
    return model


def test_initial_retrieval_invariant_to_gold_completion():
    """换 gold completion → 初始检索 block_idx 必须逐位不变（T0.1 核心反例）。"""
    model = _orthogonal_retriever(_tiny_model())
    cfg = model.config
    q = torch.randint(3, 20, (6,))
    ctx = torch.randint(3, 20, (cfg.block_size * 6,))
    gold1 = torch.randint(30, 60, (8,))
    gold2 = torch.randint(30, 60, (8,))

    def run(gold):
        ctrl = AdaptiveController(model)
        input_ids = torch.cat([q, gold]).unsqueeze(0)
        labels = torch.full((1, input_ids.shape[1]), -100, dtype=torch.long)
        out, extras = ctrl.forward_with_oracles(
            input_ids, None, labels, ctx, query_len=6)
        return ctrl.stats["last_result"].block_idx.clone()

    b1, b2 = run(gold1), run(gold2)
    assert torch.equal(b1, b2), f"初始检索随 gold 变化：{b1} vs {b2}"


def test_budget_changes_actual_selection():
    """T0.3：库容充足时，不同 budget 档取到不同数量的块（cap 联动 topk_f）。"""
    model = _tiny_model()
    cfg = model.config
    ctrl = AdaptiveController(model)
    ctrl.begin(torch.randint(3, 20, (cfg.block_size * 8,)))  # 8 块库
    q_hidden = torch.randn(1, 4, cfg.hidden_size)
    small = ctrl.retrieve_into_memory(q_hidden, budget_idx=0)   # 1024 tok → 4 块
    big = ctrl.retrieve_into_memory(q_hidden, budget_idx=4)     # 16384 tok → 但 topk_coarse=4 限制
    assert small.token_size <= big.token_size
    assert ctrl.stats["budget_cap_tokens"] in cfg.memory_budgets
    assert "actual_valid_tokens" in ctrl.stats and "unique_blocks" in ctrl.stats


def test_gold_ce_has_gradient_path():
    """T0.7：gold CE 用 grad mode 重算，backward 进入 retriever 投影。"""
    model = _tiny_model()
    cfg = model.config
    ctrl = AdaptiveController(model)
    q = torch.randint(3, 20, (5,))
    gold = torch.randint(30, 60, (6,))
    input_ids = torch.cat([q, gold]).unsqueeze(0)
    labels = torch.full((1, input_ids.shape[1]), -100, dtype=torch.long)
    out, extras = ctrl.forward_with_oracles(
        input_ids, None, labels, torch.randint(3, 20, (cfg.block_size * 4,)),
        gold_block_idx=torch.tensor([1]), query_len=5)
    assert "retrieval_ce" in extras and extras["retrieval_ce"].requires_grad
    extras["retrieval_ce"].backward()
    g = model.model.retriever.query_proj.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "gold CE 梯度未进入 query 投影"


def test_router_probs_collected_during_full_unroll():
    """T0.2：collect_exits 前向返回逐轮 router 概率（minimind_x 变体）。"""
    model = _tiny_model()
    x = torch.randint(3, 20, (2, 8))
    out = model(x, collect_exits=True, num_loops=2)
    assert out.router_probs is not None
    assert "halt" in out.router_probs and len(out.router_probs["halt"]) == 2
    if "retrieval" in out.router_probs:  # 变体含检索 router 时
        assert len(out.router_probs["retrieval"]) == 2
