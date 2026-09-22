"""可学习性测试（外部评审 P2-1）：不只验证接口，验证「能学」。

1. loop 路径可训练：固定小任务上 R=2 过拟合，loss 显著下降且梯度流经循环区
2. memory 路径可训练：注入固定 memory cross-attn 后 loss 显著下降（梯度流经 k/v 投影）
3. oracle 机制：sweep_budget 形状与取值、sweep_retrieval 的 per-sample 标签口径
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import optim

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def _cfg(variant="middle_loop", **over):
    kw = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant=variant, max_loop=2)
    kw.update(over)
    return MiniMindXConfig(**kw)


def _copy_task(n=4, seq=16, vocab=64, seed=0):
    """复制任务：label = input 右移（模型需把信息传到输出位）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(3, vocab, (n, seq), generator=g)
    y = torch.cat([torch.full((n, 1), 1, dtype=torch.long), x[:, :-1]], dim=1)
    return x, y


def _train(model, x, y, steps=60, lr=2e-3, num_loops=1, memory_factory=None):
    opt = optim.AdamW(model.parameters(), lr=lr)
    losses = []
    core_params_before = [p.detach().clone() for p in model.model.thinking_core.parameters()]
    for _ in range(steps):
        memory_kv = memory_factory() if memory_factory is not None else None
        res = model(x, labels=y, num_loops=num_loops, memory_kv_per_layer=memory_kv)
        opt.zero_grad()
        (res.loss + res.aux_loss).backward()
        opt.step()
        losses.append(res.loss.item())
    core_moved = any((p.detach() - b).abs().max().item() > 0
                     for p, b in zip(model.model.thinking_core.parameters(), core_params_before))
    return losses, core_moved


def test_loop_path_is_trainable():
    """R=2 循环区在固定任务上过拟合：loss 大幅下降，且循环区参数确实被更新。"""
    torch.manual_seed(0)
    x, y = _copy_task()
    model = MiniMindXForCausalLM(_cfg(variant="middle_loop"))
    losses, core_moved = _train(model, x, y, num_loops=2)
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])  # 过拟合发生
    assert core_moved, "循环区参数未更新——梯度没有流经 loop 路径"


def test_memory_path_is_trainable():
    """固定 memory cross-attn 注入后可学习（梯度经 project_memory 的 k/v 投影流动）。

    每步重投影（工厂）：SFT 真实路径是 no_grad 建记忆（D5），此处刻意保留梯度以验证
    memory cross-attn 路径本身可训练。
    """
    torch.manual_seed(1)
    x, y = _copy_task()
    model = MiniMindXForCausalLM(_cfg(variant="minimind_x", num_hidden_layers=6))
    mem_hidden = torch.randn(x.shape[0], 3, model.config.hidden_size)
    factory = lambda: model.model.thinking_core.project_memory(mem_hidden)
    losses, core_moved = _train(model, x, y, num_loops=2, memory_factory=factory)
    assert losses[-1] < losses[0] * 0.6, (losses[0], losses[-1])
    assert core_moved


def test_loop_embedding_ablation_runs():
    """D4 消融开关：加轮次嵌入后同样可训练（供加/不加对照实验）。"""
    torch.manual_seed(2)
    x, y = _copy_task()
    model = MiniMindXForCausalLM(_cfg(variant="middle_loop", loop_embedding=True))
    losses, _ = _train(model, x, y, num_loops=2, steps=30)
    assert losses[-1] < losses[0]


def test_sweep_budget_and_retrieval_labels():
    from model.minimind_x.controller import AdaptiveController
    from trainer.oracle_labels import sweep_budget, sweep_retrieval
    torch.manual_seed(3)
    cfg = _cfg(variant="minimind_x", num_hidden_layers=6, block_size=8)
    model = MiniMindXForCausalLM(cfg).eval()
    x = torch.randint(3, 64, (1, 10))
    y = torch.randint(3, 64, (1, 10))
    ctx = torch.randint(3, 64, (cfg.block_size * 3,))
    loss_matrix = sweep_budget(model, x, y, ctx,
                               controller_factory=lambda: AdaptiveController(model))
    assert loss_matrix.shape == (1, len(cfg.memory_budgets))
    assert torch.isfinite(loss_matrix).all()
    # V2 反事实口径（T0.1）：REUSE vs REFRESH，同可见前缀，不读 gold 后缀
    labels_r, gain = sweep_retrieval(model, x, y, ctx, query_len=6,
                                     controller_factory=lambda: AdaptiveController(model))
    assert labels_r.shape == gain.shape == (1, cfg.max_loop)
    assert set(labels_r.unique().tolist()) <= {0, 1}
    assert torch.isfinite(gain).all()
