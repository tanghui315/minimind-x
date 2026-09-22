"""Prelude zone（任务清单 T1.1）：token understanding / context encoding，无循环。

复用原 MiniMindBlock（保证 R=1、无 memory 时与原模型数值等价）；
可选 LocalWindowAttention（Doc 1 §19：Prelude/Coda 只看局部窗口）。
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from model.model_minimind import Attention, MiniMindBlock, apply_rotary_pos_emb, repeat_kv


class LocalWindowAttention(Attention):
    """滑动窗口因果注意力：query i 只看 [i-window, i] 的 key（T4.4）。"""

    def __init__(self, config):
        super().__init__(config)
        self.window = getattr(config, "attn_window", 512)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        window = self.window
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
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2),
                      repeat_kv(xv, self.n_rep).transpose(1, 2))
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 窗口带状 mask：query 全局位置 row 允许 key 全局位置 col 满足 row-window <= col <= row
        row = torch.arange(total_len - seq_len, total_len, device=x.device).unsqueeze(1)
        col = torch.arange(total_len, device=x.device).unsqueeze(0)
        band = (col <= row) & (col > row - window)
        scores = scores.masked_fill(~band.unsqueeze(0).unsqueeze(0), float("-inf"))
        if attention_mask is not None:
            scores = scores + (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), past_kv


class ZoneBlock(MiniMindBlock):
    """按 zone 注意力模式替换 attention 的 MiniMindBlock（无替换时行为与父类一致）。"""

    def __init__(self, layer_id, config, attn_mode="full"):
        super().__init__(layer_id, config)
        if attn_mode == "local":
            self.self_attn = LocalWindowAttention(config)
        elif attn_mode == "sparse":  # Baseline Model C（Doc 6 §6）
            from .retrieval.sparse_attn import SparseBlockAttention
            self.self_attn = SparseBlockAttention(config)
        self.attn_mode = attn_mode


class ZoneStack(nn.Module):
    """连续层堆叠容器：Prelude 与 Coda 共用（无循环）。"""

    def __init__(self, config, num_layers, layer_offset=0, attn_mode="full"):
        super().__init__()
        self.num_layers = num_layers
        self.gradient_checkpointing = False  # T1.4
        self.layers = nn.ModuleList(
            [ZoneBlock(layer_offset + i, config, attn_mode) for i in range(num_layers)])

    def forward(self, hidden_states, position_embeddings, past_key_values=None,
                use_cache=False, attention_mask=None):
        past_key_values = past_key_values or [None] * self.num_layers
        presents = []
        for layer, past in zip(self.layers, past_key_values):
            if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, position_embeddings, past, use_cache,
                    attention_mask, use_reentrant=False)
            else:
                hidden_states, present = layer(
                    hidden_states, position_embeddings,
                    past_key_value=past, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
        return hidden_states, presents


class Prelude(nn.Module):
    """输入理解区（Doc 1 §4.1）：No Loop / No Deep Reasoning。"""

    def __init__(self, config, layer_offset=0):
        super().__init__()
        self.stack = ZoneStack(config, config.prelude_layers, layer_offset, config.prelude_attn)

    @property
    def layers(self):
        return self.stack.layers

    def forward(self, hidden_states, position_embeddings, past_key_values=None,
                use_cache=False, attention_mask=None):
        return self.stack(hidden_states, position_embeddings, past_key_values, use_cache, attention_mask)
