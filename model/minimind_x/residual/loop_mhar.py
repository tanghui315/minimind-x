"""Loop-MHAR（任务清单 T1.1；V2 文档 §6 + 附录 A 参考实现移植）。

在 loop 边界按 feature 头读取「同一 token 的历轮 committed 状态」：
  history = [anchor(E), committed_1, ..., committed_{r-1}]，每项 [B,T,D]
  · key = 全宽 RMSNorm（仅用于打分），value = 原始状态
  · 共享 pseudo-query（零初始化）按头内积 + 来源类型偏置（latest=+4 起始）
  · softmax 仅沿来源轴（不跨 token/head/batch）
  · 输出 = latest + η·(mixed − latest)；η=0 严格恒等（迁移 warm-up 用）

纪律（V2 §6）：不在主训练路径 detach history；不原地修改已存入的 Tensor。
"""
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn

from model.model_minimind import RMSNorm


class LoopMHAR(nn.Module):
    def __init__(self, dim: int, heads: int = 4, latest_bias: float = 4.0) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim 必须为正且被 heads 整除")
        self.dim, self.heads = dim, heads
        self.norm = RMSNorm(dim)
        self.query = nn.Parameter(torch.zeros(heads, dim // heads))
        # 行语义：0=anchor，1=较早 committed，2=最新 committed
        self.type_bias = nn.Parameter(torch.zeros(3, heads))
        with torch.no_grad():
            self.type_bias[2].fill_(latest_bias)

    def forward(self, history: Sequence[Tensor], strength: float = 1.0
                ) -> Tuple[Tensor, Tensor]:
        if not history or not 0.0 <= strength <= 1.0:
            raise ValueError("history 非空且 strength∈[0,1]")
        current = history[-1]
        if current.ndim != 3 or current.shape[-1] != self.dim:
            raise ValueError("history 张量形状须为 [B,T,dim]")
        if any(x.shape != current.shape for x in history):
            raise ValueError("所有 history 须同形状")
        if any(x.device != current.device or x.dtype != current.dtype for x in history):
            raise ValueError("所有 history 须同 device/dtype")
        values = torch.stack(list(history), dim=0)      # [N,B,T,D]
        n, b, t, _ = values.shape
        keys = self.norm(values).float().reshape(n, b, t, self.heads, -1)
        # 无 1/sqrt(d_h) 缩放（V2 §6.2：改动需同步初始化与实验口径）
        logits = torch.einsum("he,nbthe->nbth", self.query.float(), keys)
        roles = torch.ones(n, dtype=torch.long, device=current.device)
        roles[0], roles[-1] = 0, 2
        logits = logits + self.type_bias[roles].float()[:, None, None, :]
        weights = logits.softmax(dim=0)                 # 仅来源轴
        vh = values.float().reshape(n, b, t, self.heads, -1)
        mixed = torch.einsum("nbth,nbthe->bthe", weights, vh).reshape(b, t, self.dim)
        out = current.float() + strength * (mixed - current.float())
        return out.to(current.dtype), weights
