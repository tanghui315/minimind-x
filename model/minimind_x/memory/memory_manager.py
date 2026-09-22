"""Memory Manager（任务清单 T5.2；Doc 1 §10-11、Doc 2 §9）—— adaptive heuristic memory
manager（措辞校准 v1.3：打分为固定/可学习权重的启发式组合，非端到端学习的记忆策略；
打分头的可学习化与 RL 化列为 V2，见 ARCHITECTURE §8）。

职责：ADD / KEEP / EVICT / REPLACE，依据综合分驱逐低价值块：
    U_i = α·R_i + β·A_i + γ·N_i − δ·D_i (+ η·P_i)
  R: Relevance（与当前 query 的余弦）  A: Attention/Retrieval Usage（EMA 统计）
  N: Novelty（新块对已有记忆的新颖度） D: Redundancy（与记忆内其它块的重复度）
  P: Persistence（历史 U 的 EMA；Doc 1 五因子，Doc 2 四因子 —— P 默认系数 0，可开）

系数默认为 config 固定值；learnable=True 时改为可学习标量（消融用）。
全部为张量操作、可 batch、不可微（驱逐是离散动作，SFT 走 oracle 监督、RL 走采样，见 D5）。
"""
import torch
import torch.nn.functional as F
from torch import nn

from .working_memory import WorkingMemory


def _cosine(a, b):
    """a: [B, d] 或 [B, m, d]，b: [B, n, d] -> [B, m, n]。"""
    return torch.einsum("bmd,bnd->bmn", F.normalize(a, dim=-1), F.normalize(b, dim=-1))


class MemoryManager(nn.Module):
    def __init__(self, config, learnable=False):
        super().__init__()
        defaults = dict(alpha=1.0, beta=0.5, gamma=0.3, delta=0.5, eta=0.0)
        vals = {k: float(getattr(config, f"mm_{k}", v)) for k, v in defaults.items()}
        for name, v in vals.items():
            t = torch.tensor(v)
            if learnable:
                self.register_parameter(name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def pairwise_sim(self, keys):
        """keys [B, M, d] -> [B, M, M]（含自身对角 1）。"""
        return _cosine(keys, keys)

    def score(self, query_vec, mem: WorkingMemory):
        """对记忆内所有块打分。query_vec: [B, d]。返回 U: [B, M]。"""
        if mem.is_empty():
            return query_vec.new_zeros(query_vec.shape[0], 0)
        rel = _cosine(query_vec.unsqueeze(1), mem.keys).squeeze(1)  # [B, M]
        sim = self.pairwise_sim(mem.keys)
        off_diag = sim.clone()
        off_diag.diagonal(dim1=1, dim2=2).fill_(float("-inf"))
        redundancy = off_diag.max(dim=-1).values                    # D: 与最相似邻居的重复度
        novelty = 1.0 - redundancy.clamp(min=0)                     # N: 邻居越不像越新颖
        u = (self.alpha * rel + self.beta * mem.usage + self.gamma * novelty
             - self.delta * redundancy + self.eta * mem.persistence)
        return u

    def novelty_of_new(self, new_keys, mem_keys):
        """新块新颖度：1 − 与现有记忆的最大相似度。new_keys [B, k, d], mem_keys [B, M, d]。"""
        if mem_keys.shape[1] == 0:
            return new_keys.new_ones(new_keys.shape[0], new_keys.shape[1])
        return 1.0 - _cosine(new_keys, mem_keys).max(dim=-1).values.clamp(min=0)

    def update(self, mem: WorkingMemory, new_hidden, new_keys, query_vec, budget_blocks,
               hit_new=None):
        """ADD + SCORE + EVICT：并入新块、按 U 保留 top-budget（Doc 2 §9）。

        new_hidden/new_keys: [B, k, d]；budget_blocks: 容量（block 数）；
        hit_new: [B, k] 新块是否本轮被选中使用（usage 更新）。
        返回 (新 WorkingMemory, 被保留的新块 mask, U 全量分数)。
        """
        mem = mem.with_added(new_hidden, new_keys)
        if hit_new is not None:  # usage EMA：新块初始 usage 由命中情况决定
            bsz, m_old = mem.usage.shape[0], mem.size - new_keys.shape[1]
            hit_full = torch.cat(
                [mem.usage.new_zeros(bsz, m_old), hit_new.float()], dim=1)
            mem = mem.with_usage_bumped(hit_full)
        u = self.score(query_vec, mem)
        if not mem.is_empty():
            mem = mem.with_persistence_ema(u)  # 先更新 persistence（形状对齐），再驱逐
        m = mem.size
        if m > budget_blocks:
            keep_idx = u.topk(budget_blocks, dim=-1).indices  # KEEP 高分，EVICT argmin(U)
            mem = mem.select(keep_idx.sort(dim=-1).values)  # 保持时序稳定
            kept_new_mask = keep_idx >= m - new_keys.shape[1]
        else:
            kept_new_mask = None
        return mem, kept_new_mask, u
