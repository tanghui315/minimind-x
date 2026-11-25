"""Thinking/Halting Router（任务清单 T2.1；Doc 2 §12、V2 §7.5）。

预测 continue/stop（二分类）；标签由 oracle R* 展开为逐轮 stop 标签。
T2.1：可选注入 detached 门控统计（ḡ, Δ^cand, Δ^commit）——经零初始化旁路加到 logit，
不改变主投影维度（旧 checkpoint 可直接加载，stats 旁路初始贡献 0）。
"""
import torch
from torch import nn


def pool_hidden(hidden, pool="mean", mask=None):
    """[B,S,d] -> [B,d]（decode 时 S=1，两种池化等价）。"""
    if hidden.dim() == 2:
        return hidden
    if mask is not None:
        if pool == "last":
            positions = torch.arange(hidden.size(1), device=hidden.device)
            last = positions.masked_fill(~mask.bool(), -1).amax(dim=1)
            row = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[row, last.clamp_min(0)] * (last >= 0).unsqueeze(-1)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    if pool == "last":
        return hidden[:, -1, :]
    return hidden.mean(dim=1)


class HaltRouter(nn.Module):
    """p_s = σ(W_s · pool(h_r) + W_stats · [ḡ,Δ^cand,Δ^commit])，P(stop)。"""

    def __init__(self, config):
        super().__init__()
        self.pool = config.halt_pool
        self.proj = nn.Linear(config.hidden_size, 1, bias=True)
        # T2.1 / V2 §7.5：detached 统计旁路（零初始化 → 与旧路径数值一致）
        self.stats_proj = nn.Linear(3, 1, bias=False)
        nn.init.zeros_(self.stats_proj.weight)

    def forward(self, hidden, residual_stats=None, mask=None):
        return torch.sigmoid(self.forward_logits(hidden, residual_stats, mask))

    def forward_logits(self, hidden, residual_stats=None, mask=None):
        """T2.4：返回 halt logit（非概率）。"""
        pooled = pool_hidden(hidden, self.pool, mask)
        logit = self.proj(pooled)
        if residual_stats is not None:
            stats = torch.stack([
                residual_stats["gate_mean"],
                residual_stats["delta_cand"],
                residual_stats["delta_commit"],
            ], dim=-1).to(logit.dtype).to(logit.device)
            logit = logit + self.stats_proj(stats)
        return logit.squeeze(-1)
