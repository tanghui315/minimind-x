"""Coda zone（任务清单 T1.1）：output formation / token prediction，无循环、不参与 loop。"""
from torch import nn

from .prelude import ZoneStack


class Coda(nn.Module):
    """输出生成区（Doc 1 §4.3）。"""

    def __init__(self, config, layer_offset=0):
        super().__init__()
        self.stack = ZoneStack(config, config.coda_layers, layer_offset, config.coda_attn)

    @property
    def layers(self):
        return self.stack.layers

    def forward(self, hidden_states, position_embeddings, past_key_values=None,
                use_cache=False, attention_mask=None):
        return self.stack(hidden_states, position_embeddings, past_key_values, use_cache, attention_mask)
