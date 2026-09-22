"""Hierarchical Sparse Retriever / QSA-style（任务清单 T4.2；Doc 1 §7-8、Doc 2 §6）。

两级检索：
  Coarse：query · block_key（mean-pool 索引）→ TopK_c 个候选块
  Fine：候选块内 token 级 max-sim 重排（late-interaction 式）→ TopK_f 个最终块

监督：合成 needle 数据自带 gold block 标注 → coarse 分数做 CE（retrieval CE）。
TopK 不可微（D5）。**Frozen Index 范式**：LongContextStore 建库阶段在 no_grad 下
用 key_proj 生成库 key；训练阶段梯度只走 **query 投影** 的 CE/蒸馏路径，
key_proj 不获得梯度（不是 bug，是可缓存索引 + 推理效率的设计选择）。
配置声明 config.freeze_key=True；若要 Q/K 联合训练，须周期 rebuild index
（key_refresh_interval），不能只去掉 no_grad。
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .block_store import LongContextStore


@dataclass
class RetrievalResult:
    block_idx: torch.Tensor            # [B, k] 最终选中块
    block_hidden: torch.Tensor         # [B, k, block, d] 懒编码的块内容
    coarse_scores: torch.Tensor        # [B, n_blocks]（gold CE / distill 的 student 侧）
    cand_idx: torch.Tensor = None      # [B, c] 粗检候选块号（distill 对齐用）
    fine_scores: torch.Tensor = None   # [B, c] 候选块 token 级 max-sim（distill 的 teacher 侧）
    n_retrievals: int = 1


class QSARetriever(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.query_proj = nn.Linear(d, d, bias=False)
        self.key_proj = nn.Linear(d, d, bias=False)
        self.query_pool = config.retrieve_query_pool

    def project_query(self, query_hidden):
        """query_hidden [B,S,d] 或 [B,d] -> [B,d]。"""
        if query_hidden.dim() == 3:
            pooled = (query_hidden[:, -1] if self.query_pool == "last"
                      else query_hidden.mean(dim=1))
        else:
            pooled = query_hidden
        return self.query_proj(pooled)

    @staticmethod
    def _cos(q, keys):
        """q [B,d], keys [n,d] 或 [B,n,d] -> [B,n]。"""
        if keys.dim() == 2:
            keys = keys.unsqueeze(0).expand(q.shape[0], -1, -1)
        return torch.einsum("bd,bnd->bn", F.normalize(q, dim=-1), F.normalize(keys, dim=-1))

    def coarse_scores(self, query_vec, store: LongContextStore):
        if store.block_keys is None:
            raise RuntimeError("先调用 store.build_keys(retriever, embed_tokens)")
        return self._cos(query_vec, store.block_keys)

    def retrieve(self, query_hidden, store: LongContextStore, model=None,
                 topk_coarse=None, topk_fine=None):
        """执行两级检索。query_hidden 来自 Prelude(问题) 或 loop exit hidden（D9）。"""
        model = model or getattr(self, "_model", None)
        if model is None:
            raise ValueError("retrieve 需要 model（用于懒编码块内容）")
        cfg = self.config
        topk_c = topk_coarse or cfg.topk_coarse
        topk_f = topk_fine or cfg.topk_fine
        q = self.project_query(query_hidden)                      # [B,d]
        coarse = self.coarse_scores(q, store)                     # [B,n]
        n = coarse.shape[1]
        topk_c = min(topk_c, n)
        cand_idx = coarse.topk(topk_c, dim=-1).indices            # [B,c]
        # ---- Fine：候选块 token 级 max-sim 重排 ----
        cand_hidden = store.get_block_hidden(model, cand_idx)     # [B,c,block,d]
        tok_sim = torch.einsum("bd,bckd->bck", F.normalize(q, dim=-1),
                               F.normalize(cand_hidden, dim=-1))
        fine_scores = tok_sim.max(dim=-1).values                   # [B,c]
        topk_f = min(topk_f, topk_c)
        k = fine_scores.topk(topk_f, dim=-1).indices               # [B,k]（cand 内下标）
        final_idx = cand_idx.gather(1, k)                          # [B,k] 全局块号
        final_hidden = cand_hidden.gather(
            1, k.view(*k.shape, 1, 1).expand(-1, -1, cfg.block_size, cand_hidden.shape[-1]))
        return RetrievalResult(block_idx=final_idx, block_hidden=final_hidden,
                               coarse_scores=coarse, cand_idx=cand_idx,
                               fine_scores=fine_scores)

    def retrieval_ce_loss(self, coarse_scores, gold_idx):
        """gold block 的 CE（合成 needle 数据的 gold 标注；Doc 4 §7.3）。gold_idx: [B]。"""
        return F.cross_entropy(coarse_scores, gold_idx)

    def distill_loss(self, coarse_scores, cand_idx, fine_scores):
        """late-interaction 蒸馏（v1.3，外部评审 Bug 3）：fine 排序知识 → coarse 打分。

        teacher = 候选块 token 级 max-sim（detach），student = 同批候选块的 coarse 分数。
        两侧都是 query/key 投影的可微函数，KL 不经过离散 TopK —— 在不放松选择离散性的
        前提下训练投影（弥补 gold CE 只在 needle 数据上可用的稀疏监督）。
        """
        student = coarse_scores.gather(1, cand_idx)                # [B,c]，梯度入 query_proj
        teacher = fine_scores.detach()
        return F.kl_div(F.log_softmax(student, dim=-1),
                        F.log_softmax(teacher, dim=-1),
                        log_target=True, reduction="batchmean")

    def budget_topk(self, budget_tokens, block_size=None):
        """预算（token 数）-> topk_f（块数），供 BudgetRouter 的档位联动。"""
        block_size = block_size or self.config.block_size
        return max(1, int(budget_tokens) // block_size)
