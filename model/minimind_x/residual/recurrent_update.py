"""Gated Recurrent Update（任务清单 T1.2；V2 文档 §7 + 附录 A 参考实现移植）。

  ΔU = U_r − H_{r-1}
  f  = [RMSNorm(H_prev); RMSNorm(E); RMSNorm(ΔU)]
  g  = σ(W2·SiLU(W1·f)+b2)                # 低秩，默认 rank=64，FP32 门控计算
  H_r = H_{r-1} + g ⊙ (U_r − H_{r-1})

初始化（V2 §7.4）：W2=0、b2=+4 → g≈0.982（近开门；第一步 W1 可能暂无梯度——
初始化的正常结果，验收看后续步整条路径）。

禁止项（V2 §5）：U_r 已含 Core 内部残差，不能写成 H_prev + g*U_r；
bypass 必须是上一轮 committed state（不是 MHAR 混合后的 Z_r）。
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.model_minimind import RMSNorm


class RecurrentUpdateGate(nn.Module):
    def __init__(self, dim: int, rank: int = 64, bias: float = 4.0) -> None:
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError("dim 与 rank 必须为正")
        self.dim = dim
        self.h_norm, self.e_norm, self.delta_norm = (
            RMSNorm(dim), RMSNorm(dim), RMSNorm(dim))
        self.fc1 = nn.Linear(3 * dim, rank, bias=True)
        self.fc2 = nn.Linear(rank, dim, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, bias)

    def forward(self, previous: Tensor, anchor: Tensor, candidate: Tensor,
                active: Optional[Tensor] = None,
                force_gate: Optional[float] = None) -> Tuple[Tensor, Tensor]:
        if previous.shape != candidate.shape or previous.shape != anchor.shape:
            raise ValueError("previous/anchor/candidate 须同形 [B,T,D]")
        if previous.ndim != 3 or previous.shape[-1] != self.dim:
            raise ValueError("hidden 形状非法")
        if active is not None and (active.dtype != torch.bool
                                    or active.shape != previous.shape[:2]):
            raise ValueError("active 须为 bool [B,T]")
        if force_gate is None:
            features = torch.cat([self.h_norm(previous), self.e_norm(anchor),
                                  self.delta_norm(candidate - previous)], dim=-1)
            gate = self.fc2(F.silu(self.fc1(features))).float().sigmoid()
            updated = (previous.float()
                       + gate * (candidate.float() - previous.float())).to(previous.dtype)
        else:
            if not 0 <= force_gate <= 1:
                raise ValueError("force_gate ∈ [0,1]")
            gate = torch.full_like(previous, force_gate, dtype=torch.float32)
            if force_gate == 0:
                updated = previous
            elif force_gate == 1:
                updated = candidate
            else:
                updated = (previous.float()
                           + gate * (candidate.float() - previous.float())).to(previous.dtype)
        if active is not None:
            updated = torch.where(active[..., None], updated, previous)
        return updated, gate
