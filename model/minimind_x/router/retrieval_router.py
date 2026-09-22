"""Retrieval Router（任务清单 T6.1 / T2.5；Doc 2 §7、V2 §9.2）。

二值 p_r = σ(W_r · pool(h_r))：当前信息够不够（兼容旧路径）。
V2 三动作（T2.5）：a^ret ∈ {REUSE, RESCORE_WARM, GLOBAL_REFRESH}；
无 Warm Candidate Cache 时 RESCORE_WARM 在 logits 中 mask 掉（不可伪装成廉价全局搜索）。
oracle 标签 = 同前缀反事实（trainer/oracle_labels.py）。
"""
import torch
from torch import nn

from .halt_router import pool_hidden

ACTION_REUSE = 0
ACTION_RESCORE_WARM = 1
ACTION_GLOBAL_REFRESH = 2
ACTION_NAMES = ("REUSE", "RESCORE_WARM", "GLOBAL_REFRESH")
ACTION_COUNT = 3
# 唯一合法枚举（V2 §9.2）；旧二值标签只在数据边界转换一次
ACTION_SCHEMA_VERSION = 2


def normalize_oracle_actions(seq, schema_version: int = ACTION_SCHEMA_VERSION):
    """把 oracle 动作序列归一到 schema v2：0=REUSE, 1=RESCORE_WARM, 2=GLOBAL_REFRESH。

    schema v1（旧二值）：0=REUSE, 1=GLOBAL_REFRESH（无 RESCORE_WARM）。
    进入模型后禁止再转换（Oracle/loss/执行/trace 共用本枚举）。
    """
    out = []
    for a in seq or []:
        a = int(a)
        if schema_version <= 1:
            # v1: 0=REUSE, 1=REFRESH → v2: 0=REUSE, 2=GLOBAL_REFRESH
            out.append(ACTION_REUSE if a == 0 else ACTION_GLOBAL_REFRESH)
        else:
            if a not in (ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH):
                raise ValueError(f"非法动作 id {a}（schema v{schema_version}）")
            out.append(a)
    return out


def action_mask(warm_available: bool, device=None, batch: int = 1) -> torch.Tensor:
    """可用动作 mask [B,3]。无 warm cache 时 RESCORE_WARM 不可用（V2 §9.2）。"""
    m = torch.ones(batch, ACTION_COUNT, dtype=torch.bool, device=device)
    if not warm_available:
        m[:, ACTION_RESCORE_WARM] = False
    return m


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """把不可用动作 logits 置 -inf（softmax 前）。"""
    return logits.masked_fill(~mask, float("-inf"))


class RetrievalRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool = config.retrieve_query_pool
        self.proj = nn.Linear(config.hidden_size, 1, bias=True)
        # T2.5：三动作分类头（REUSE / RESCORE_WARM / GLOBAL_REFRESH）
        self.action_head = nn.Linear(config.hidden_size, ACTION_COUNT, bias=True)

    def forward(self, hidden):
        """二值 P(需要离开 REUSE)——旧 oracle/推理路径兼容。"""
        pooled = pool_hidden(hidden, self.pool)
        return torch.sigmoid(self.proj(pooled)).squeeze(-1)

    def action_logits(self, hidden, warm_available: bool = True):
        """三动作 logits [B,3]；无 warm cache 时 RESCORE_WARM 置 -inf。"""
        pooled = pool_hidden(hidden, self.pool)
        logits = self.action_head(pooled)
        mask = action_mask(warm_available, device=logits.device, batch=logits.shape[0])
        return apply_action_mask(logits, mask)

    def action_logprobs(self, hidden, actions, warm_available: bool = True):
        """所选动作的 log π(a|s)（轨迹联合路径概率用，T4.7）。actions: [B] long。"""
        logits = self.action_logits(hidden, warm_available=warm_available)
        return torch.log_softmax(logits, dim=-1).gather(1, actions.view(-1, 1)).squeeze(-1)
