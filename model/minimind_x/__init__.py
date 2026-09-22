# MiniMind-X: Adaptive Compute Foundation Model
# 包结构对应《MiniMind-X 编码任务清单》D1 / Doc 2 §23。
#
# 命名映射（Doc 2 §23 → 本实现）：
#   retrieval/indexer.py + selector.py -> 合并进 retrieval/qsa_retriever.py
#   memory/memory_cache.py             -> 并入 memory/working_memory.py（跨 loop K/V 缓存）
from .config import MiniMindXConfig, PRESETS, resolve_variant
from .minimind_x import (MiniMindXModel, MiniMindXForCausalLM, MiniMindXOutput,
                         remap_minimind_state_dict)
from .losses import joint_loss, expected_loops, halt_labels_from_oracle
from .flops import estimate_flops, count_params

__all__ = [
    "MiniMindXConfig", "PRESETS", "resolve_variant",
    "MiniMindXModel", "MiniMindXForCausalLM", "MiniMindXOutput", "remap_minimind_state_dict",
    "joint_loss", "expected_loops", "halt_labels_from_oracle",
    "estimate_flops", "count_params",
]
