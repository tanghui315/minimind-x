"""Sparse Block Attention（Baseline Model C，任务清单 T4.3/T4.5；Doc 1 §8）。

QSA/NSA 思路的极简实现：每个 query 在 [top-k 块 ∪ 局部窗口] 上做注意力。
- 块键 = 当前层 key 在块内 mean（compressed-K），逐 query top-k 选块
- 注意力走手动路径（带块选择 mask + 窗口 mask + 因果 mask）
- 仅用于 Model C（sparse attention + fixed depth）与消融；
  终局 minimind_x 的长上下文走 store+检索（D9），不经过本层。

研究级实现：[B, H, S, S] 的 mask 会随 S 增长，适合 ≤32K 序列的 baseline 对比。
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from model.model_minimind import Attention, apply_rotary_pos_emb, repeat_kv


class SparseBlockAttention(Attention):
    def __init__(self, config):
        super().__init__(config)
        self.block = config.block_size
        self.n_selected = getattr(config, "sparse_topk_blocks", 64)
        self.window = getattr(config, "attn_window", 512)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        total_len = xk.shape[1]
        xq = xq.transpose(1, 2)
        xk_rep = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv_rep = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # ---- 块键（compressed K）：key 在块内 mean → 逐 query 选块 ----
        n_blk = math.ceil(total_len / self.block)
        pad = n_blk * self.block - total_len
        k_pad = F.pad(xk_rep, (0, 0, 0, pad)) if pad else xk_rep
        block_k = k_pad.view(bsz, self.n_local_heads, n_blk, self.block,
                             self.head_dim).mean(dim=3)
        # query 与块键的相似度：[B, H, S, n_blk]
        blk_score = torch.einsum("bhsd,bhnd->bhsn",
                                 F.normalize(xq, dim=-1), F.normalize(block_k, dim=-1))
        # ---- T0.4（C8 修复）：块选择因果化 + 最终掩码取交集 ----
        # query i（全局位）可见 key 满足 col ≤ i；块 b 的 key 起始于 b*block，
        # 故「块含至少一个可见 key」⟺ blk_start(b) ≤ i —— 逐行块可见性 [S, n_blk]
        blk_start = torch.arange(n_blk, device=x.device) * self.block
        row_g = torch.arange(total_len - seq_len, total_len, device=x.device)
        blk_vis_rows = blk_start.unsqueeze(0) <= row_g.unsqueeze(1)          # [S, n_blk]
        blk_score = blk_score.masked_fill(
            ~blk_vis_rows.unsqueeze(0).unsqueeze(0), float("-inf"))
        k_sel = min(self.n_selected, n_blk)
        sel = blk_score.topk(k_sel, dim=-1).indices                    # [B,H,S,k]
        # 逐 head 选块的并集（研究级近似：union over heads）
        block_mask = torch.zeros(bsz, self.n_local_heads, seq_len, n_blk,
                                 device=x.device, dtype=torch.bool).scatter_(-1, sel, True)
        col_block = torch.arange(total_len, device=x.device) // self.block  # key 的块号
        allow_blocks = block_mask.any(dim=1, keepdim=True)[..., col_block]  # [B,1,S,total]

        # ---- 窗口 + 因果（修复：选择并集后再与因果取交集，块选择不得越过因果线）----
        row = torch.arange(total_len - seq_len, total_len, device=x.device).unsqueeze(1)
        col = torch.arange(total_len, device=x.device).unsqueeze(0)
        band = (col <= row) & (col > row - self.window)
        causal = col <= row                                                 # [S, total]
        allow = (allow_blocks | band.unsqueeze(0).unsqueeze(0)) & causal.unsqueeze(0).unsqueeze(0)
        scores = (xq @ xk_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allow, float("-inf"))
        if attention_mask is not None:
            scores = scores + (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv_rep
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), past_kv
