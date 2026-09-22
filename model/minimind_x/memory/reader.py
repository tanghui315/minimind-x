"""GatedMemoryReader（任务清单 T1.3；V2 文档 §8 + 附录 A 参考实现移植）。

独立的外部证据读取支路，替代旧的「memory KV 拼进 self-attention softmax」（C2）：
  · 独立 Q/K/V/O 投影（GQA），Q/K 均不用 token RoPE
  · 默认只在 Core 首层 self-attention 之后、FFN 之前执行一次
  · token 级 Reader Gate（c_r:[B,T,1]）+ 有界有符号残差缩放 ρ=tanh(a_m)
  · 严格 no-op 语义：None/M=0/全 masked/混合 batch 空行 → 恒等且无 NaN
  · NULL KV 仅对「无任何可见真实 memory」的 query 启用，其注入 delta 严格置零

初始化纪律（V2 §8.5）：不能同时 scale=0 且 o_proj=0（双零锁死）。
本实现：o_proj 正常随机 + gate_fc2 零初始化（gate 初值 0.5）+ scale 初值 0.01
（严格迁移可设 scale_init=0 → 精确 no-op 且 scale 仍获梯度）。

project_memory 的 grad 契约（V2 §13.2）：训练时必须在 grad mode 调用；
缓存归属调用方（按 memory_version/投影版本失效）。
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.model_minimind import RMSNorm


@dataclass(frozen=True)
class MemoryKV:
    """独立 memory K/V。缓存方须跟踪 memory_version / 投影权重版本（V2 §11.1）。"""
    key: Tensor    # [B,Hkv,M,Dhead]
    value: Tensor  # [B,Hkv,M,Dhead]
    valid: Tensor  # [B,M] bool


class GatedMemoryReader(nn.Module):
    def __init__(self, dim: int, memory_dim: int, query_heads: int = 8,
                 kv_heads: int = 2, head_dim: int = 64, gate_rank: int = 32,
                 scale_init: float = 0.01, dropout: float = 0.0) -> None:
        super().__init__()
        if min(dim, memory_dim, query_heads, kv_heads, head_dim, gate_rank) <= 0:
            raise ValueError("维度与头数必须为正")
        if query_heads % kv_heads or not -1 < scale_init < 1:
            raise ValueError("query_heads 须被 kv_heads 整除；scale_init ∈ (-1,1)")
        if not 0 <= dropout < 1:
            raise ValueError("dropout ∈ [0,1)")
        self.dim, self.memory_dim = dim, memory_dim
        self.query_heads, self.kv_heads, self.head_dim = query_heads, kv_heads, head_dim
        self.dropout = dropout
        self.q_norm, self.m_norm = RMSNorm(dim), RMSNorm(memory_dim)
        self.q_proj = nn.Linear(dim, query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(query_heads * head_dim, dim, bias=False)
        self.read_norm = RMSNorm(dim)
        self.gate_fc1 = nn.Linear(2 * dim, gate_rank)
        self.gate_fc2 = nn.Linear(gate_rank, 1)
        nn.init.zeros_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)
        self.scale_logit = nn.Parameter(torch.tensor(math.atanh(scale_init)))

    def project_memory(self, hidden: Tensor, valid: Tensor) -> MemoryKV:
        if hidden.ndim != 3 or hidden.shape[-1] != self.memory_dim:
            raise ValueError("memory hidden 须为 [B,M,memory_dim]")
        if valid.dtype != torch.bool or valid.shape != hidden.shape[:2]:
            raise ValueError("valid 须为 bool [B,M]")
        b, m, _ = hidden.shape
        # mask 先于归一化/投影：padding 位不携带内容
        clean = torch.where(valid[..., None], hidden, torch.zeros_like(hidden))
        mh = self.m_norm(clean)
        k = self.k_proj(mh).reshape(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mh).reshape(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        return MemoryKV(k, v, valid)

    def forward(self, hidden: Tensor, kv: Optional[MemoryKV],
                query_valid: Optional[Tensor] = None,
                allowed: Optional[Tensor] = None) -> Tuple[Tensor, dict]:
        b, t, d = hidden.shape
        if d != self.dim:
            raise ValueError("hidden 宽度不匹配")
        if query_valid is None:
            query_valid = torch.ones(b, t, dtype=torch.bool, device=hidden.device)
        if query_valid.dtype != torch.bool or query_valid.shape != (b, t):
            raise ValueError("query_valid 须为 bool [B,T]")
        if kv is None or kv.key.shape[-2] == 0:
            return hidden, {"gate": hidden.new_zeros(b, t, 1),
                            "delta": torch.zeros_like(hidden)}
        if kv.key.shape[0] != b or kv.value.shape != kv.key.shape:
            raise ValueError("memory K/V batch 或形状不匹配")
        m = kv.key.shape[-2]
        visible = kv.valid[:, None, :].expand(b, t, m) & query_valid[..., None]
        if allowed is not None:
            if allowed.dtype != torch.bool or allowed.shape != (b, t, m):
                raise ValueError("allowed 须为 bool [B,T,M]")
            visible = visible & allowed
        any_real = visible.any(dim=-1, keepdim=True)          # [B,T,1]
        visible = torch.cat([visible, ~any_real], dim=-1)[:, None, :, :]
        null_k = kv.key.new_zeros(b, self.kv_heads, 1, self.head_dim)
        k = torch.cat([kv.key, null_k], dim=-2)
        v = torch.cat([kv.value, torch.zeros_like(null_k)], dim=-2)
        rep = self.query_heads // self.kv_heads
        # 参考路径（可读性）；等价性验证后可换 fused GQA kernel
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        q = self.q_proj(self.q_norm(hidden)).reshape(
            b, t, self.query_heads, self.head_dim).transpose(1, 2)
        read = F.scaled_dot_product_attention(
            q, k, v, attn_mask=visible, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0)
        read = read.transpose(1, 2).reshape(b, t, self.query_heads * self.head_dim)
        message = self.o_proj(read)
        gate_features = torch.cat([self.q_norm(hidden), self.read_norm(message)], dim=-1)
        gate = self.gate_fc2(F.silu(self.gate_fc1(gate_features))).float().sigmoid()
        gate = gate * any_real.float()
        delta = self.scale_logit.float().tanh() * gate * message.float()
        delta = torch.where(any_real, delta, torch.zeros_like(delta))
        out = (hidden.float() + delta).to(hidden.dtype)
        out = torch.where(query_valid[..., None], out, hidden)
        return out, {"gate": gate, "delta": delta.to(hidden.dtype)}
