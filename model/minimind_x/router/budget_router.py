"""Memory/Budget Router（任务清单 T5.4；Doc 4 §14）。

预测所需 Memory 预算档位（{1K,2K,4K,8K,16K} 的分类），监督信号为 oracle B*。
"""
import torch
from torch import nn

from .halt_router import pool_hidden


class BudgetRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_budgets = len(config.memory_budgets)
        self.proj = nn.Linear(config.hidden_size, self.n_budgets, bias=True)

    def forward(self, hidden):
        """hidden [B,S,d] -> logits [B, n_budgets]（训练 CE；推理 argmax 得预算档）。"""
        return self.proj(pool_hidden(hidden, "mean"))
