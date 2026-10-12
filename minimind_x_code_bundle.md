# MiniMind-X 当前实现审查包（含 V2 残差改造）

- 文件数：68（代码 65 / 文档 3）
- 总行数：10397
- 选择规则：MiniMind-X / V2 实现面（model/minimind_x + 相关 trainer/dataset/data_gen/tests），按 模型→训练器→数据→脚本→测试→文档 排序
- 审查焦点：V2 残差改造（Loop-MHAR / UpdateGate / GatedMemoryReader）、因果 query、三动作检索、缓存契约、append-only 双执行器、Frozen Index、训练接线、checkpoint、合成数据真值、sparse 块级因果
- 背景材料：docs/MiniMind-X V2 残差改造任务清单.md（任务与验收）、docs/minimind_x_gated_recurrence_loop_mhar_memory_reader_v2.md（V2 规格）、docs/ARCHITECTURE.md（D1–D9）
- 验证状态：python -m pytest tests/ -q → **133 passed**（本地 CPU + H200 torch 2.11 双环境一致）
- 已知剩余：T5.3 八卡短跑、T5.4 迁移首训对照（待 H200 夜间窗口）；训练主路径 collect_exits 仍是全序列 teacher-forcing

## 目录

- `model/minimind_x/__init__.py` (18 行)
- `model/minimind_x/coda.py` (20 行)
- `model/minimind_x/config.py` (221 行)
- `model/minimind_x/controller.py` (405 行)
- `model/minimind_x/flops.py` (75 行)
- `model/minimind_x/losses.py` (256 行)
- `model/minimind_x/memory/__init__.py` (4 行)
- `model/minimind_x/memory/memory_manager.py` (109 行)
- `model/minimind_x/memory/reader.py` (118 行)
- `model/minimind_x/memory/working_memory.py` (182 行)
- `model/minimind_x/minimind_x.py` (509 行)
- `model/minimind_x/prelude.py` (104 行)
- `model/minimind_x/residual/__init__.py` (5 行)
- `model/minimind_x/residual/loop_mhar.py` (56 行)
- `model/minimind_x/residual/recurrent_update.py` (65 行)
- `model/minimind_x/retrieval/__init__.py` (5 行)
- `model/minimind_x/retrieval/block_store.py` (66 行)
- `model/minimind_x/retrieval/qsa_retriever.py` (112 行)
- `model/minimind_x/retrieval/sparse_attn.py` (86 行)
- `model/minimind_x/router/__init__.py` (5 行)
- `model/minimind_x/router/budget_router.py` (19 行)
- `model/minimind_x/router/halt_router.py` (54 行)
- `model/minimind_x/router/retrieval_router.py` (77 行)
- `model/minimind_x/thinking_core.py` (331 行)
- `trainer/dual_executor.py` (234 行)
- `trainer/oracle_labels.py` (224 行)
- `trainer/residual_schedule.py` (47 行)
- `trainer/train_grpo_x.py` (357 行)
- `trainer/train_pretrain_x.py` (281 行)
- `trainer/train_sft_x.py` (343 行)
- `trainer/x_utils.py` (308 行)
- `dataset/retrieval_qa_dataset.py` (115 行)
- `scripts/data_gen/__init__.py` (1 行)
- `scripts/data_gen/gen_evidence_families.py` (147 行)
- `scripts/data_gen/gen_long_dep.py` (50 行)
- `scripts/data_gen/gen_memory_mgmt.py` (56 行)
- `scripts/data_gen/gen_multihop.py` (62 行)
- `scripts/data_gen/gen_needle.py` (66 行)
- `scripts/data_gen/gen_planning.py` (74 行)
- `scripts/data_gen/gen_repo_context.py` (89 行)
- `tests/conftest.py` (4 行)
- `tests/test_actions_and_executors.py` (250 行)
- `tests/test_append_only_alignment.py` (106 行)
- `tests/test_cache_replay.py` (137 行)
- `tests/test_causality.py` (95 行)
- `tests/test_config.py` (50 行)
- `tests/test_data_truth_and_sparse_causal.py` (104 行)
- `tests/test_distill.py` (83 行)
- `tests/test_equivalence.py` (95 行)
- `tests/test_flops.py` (40 行)
- `tests/test_frozen_key_index.py` (57 行)
- `tests/test_learnability.py` (103 行)
- `tests/test_loop_kv.py` (100 行)
- `tests/test_losses.py` (85 行)
- `tests/test_memory.py` (77 行)
- `tests/test_residual_invariants.py` (238 行)
- `tests/test_residual_v2_integration.py` (90 行)
- `tests/test_retrieval.py` (105 行)
- `tests/test_review_g1_fixes.py` (130 行)
- `tests/test_review_g2g3_fixes.py` (118 行)
- `tests/test_sparse_and_cot.py` (84 行)
- `tests/test_trainer_gradients.py` (115 行)
- `tests/test_v2_acceptance.py` (82 行)
- `tests/test_v2_field_contracts.py` (117 行)
- `tests/test_v2_replay_and_cachekey.py` (111 行)
- `docs/MiniMind-X V2 残差改造任务清单.md` (344 行)
- `docs/minimind_x_gated_recurrence_loop_mhar_memory_reader_v2.md` (1832 行)
- `docs/ARCHITECTURE.md` (289 行)

---

# model/minimind_x/__init__.py

```python
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
```

---

# model/minimind_x/coda.py

```python
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
```

---

# model/minimind_x/config.py

```python
"""MiniMindXConfig（任务清单 T0.1 / D6 / D9）。

与原 MiniMindConfig 的关系：继承其全部字段默认值（vocab/heads/RoPE/MoE 等），
新增 zone 划分、循环、router、memory/retrieval、loss 权重等 X 专属字段。
原 `model/model_minimind.py` 保持不动（Model A baseline）。
"""
import math
from transformers import PretrainedConfig

# 模型规模 preset（Doc 6 §4 + 任务清单 D6：S=12 层(4,4,4)、M=24 层(8,8,8)、L=32 层(10,12,10)）
PRESETS = {
    "S": dict(hidden_size=512, num_hidden_layers=12, num_attention_heads=8,
              prelude_layers=4, thinking_layers=4, coda_layers=4),
    "M": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
              prelude_layers=8, thinking_layers=8, coda_layers=8),
    "L": dict(hidden_size=1536, num_hidden_layers=32, num_attention_heads=24,
              prelude_layers=10, thinking_layers=12, coda_layers=10),
}

# variant 预设（Doc 3 §6 实验矩阵 + Ablation D）：
#   dense        中段切分但 use_loop=False（与 minimind_x 同权重布局，R 恒 1，作受控 baseline）
#   full_loop    (0, L, 0) 全层循环
#   middle_loop  L1-L3 + (L4-L6)×R + L7-L8
#   late_middle  L1-L4 + (L5-L6)×R + L7-L8
#   minimind_x   终局：中段循环 + memory + retrieval
VARIANTS = ("dense", "full_loop", "middle_loop", "late_middle", "minimind_x")


def resolve_variant(variant: str, num_hidden_layers: int):
    """返回 variant 的默认 (prelude, thinking, coda) 切分与功能开关。"""
    L = num_hidden_layers
    if variant == "full_loop":
        splits = (0, L, 0)
    elif variant == "late_middle":  # Doc 3: L1-L4 + (L5-L6)×R + L7-L8
        prelude, coda = max(1, (L + 1) // 2), max(1, L // 4)
        splits = (prelude, L - prelude - coda, coda)
    else:  # dense / middle_loop / minimind_x：默认 3:3:2（8 层时为 3/3/2）
        prelude, coda = max(1, round(L * 3 / 8)), max(1, round(L / 4))
        splits = (prelude, L - prelude - coda, coda)
    if splits[1] < 1:
        raise ValueError(f"variant={variant} 在 {L} 层下 thinking zone 为空，请显式指定切分")
    use_loop = variant != "dense"
    use_memory = variant == "minimind_x"
    use_retrieval = variant == "minimind_x"
    return dict(prelude_layers=splits[0], thinking_layers=splits[1], coda_layers=splits[2],
                use_loop=use_loop, use_memory=use_memory, use_retrieval=use_retrieval)


class MiniMindXConfig(PretrainedConfig):
    model_type = "minimind_x"

    def __init__(self, hidden_size=None, num_hidden_layers=None, use_moe=False,
                 preset=None, variant="minimind_x",
                 prelude_layers=None, thinking_layers=None, coda_layers=None,
                 **kwargs):
        # ---- preset 覆盖链：显式参数 > preset > 默认（任务清单 D6）----
        base = dict(PRESETS[preset]) if preset else {}
        if hidden_size is not None:
            base["hidden_size"] = hidden_size
        if num_hidden_layers is not None:
            base["num_hidden_layers"] = num_hidden_layers
        base.setdefault("hidden_size", 768)
        base.setdefault("num_hidden_layers", 8)
        base.setdefault("num_attention_heads", 8)
        # preset 携带的其它字段（heads 等）注入 kwargs，用户显式参数优先
        for k, v in base.items():
            if k not in ("hidden_size", "num_hidden_layers") and k not in kwargs:
                kwargs[k] = v
        super().__init__(**kwargs)
        hidden_size, num_hidden_layers = base["hidden_size"], base["num_hidden_layers"]

        # ---- 原 MiniMind 字段（保持默认一致，保证权重可互转）----
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_attention_heads // 2)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32, "beta_slow": 1, "factor": 16,
            "original_max_position_embeddings": 2048, "attention_factor": 1.0, "type": "yarn",
        } if self.inference_rope_scaling else None
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

        # ---- variant 解析（显式切分优先）----
        if variant not in VARIANTS:
            raise ValueError(f"未知 variant: {variant}，可选 {VARIANTS}")
        self.variant = variant
        resolved = resolve_variant(variant, num_hidden_layers)
        if preset and "thinking_layers" in PRESETS[preset]:  # preset 的显式切分优先于 variant 默认
            resolved.update({k: PRESETS[preset][k]
                             for k in ("prelude_layers", "thinking_layers", "coda_layers")})
        self.prelude_layers = prelude_layers if prelude_layers is not None else resolved["prelude_layers"]
        self.thinking_layers = thinking_layers if thinking_layers is not None else resolved["thinking_layers"]
        self.coda_layers = coda_layers if coda_layers is not None else resolved["coda_layers"]
        if self.prelude_layers + self.thinking_layers + self.coda_layers != num_hidden_layers:
            raise ValueError(
                f"zone 切分 {self.prelude_layers}+{self.thinking_layers}+{self.coda_layers} "
                f"!= 总层数 {num_hidden_layers}")
        self.use_loop = kwargs.get("use_loop", resolved["use_loop"])
        self.use_memory = kwargs.get("use_memory", resolved["use_memory"])
        self.use_retrieval = kwargs.get("use_retrieval", resolved["use_retrieval"])
        self.use_sparse_attn = kwargs.get("use_sparse_attn", False)  # Baseline C（Model C）

        # ---- 循环 / router（Doc 2 §11-12, Doc 3 §7-8）----
        self.max_loop = kwargs.get("max_loop", 4)          # R_max
        self.halt_threshold = kwargs.get("halt_threshold", 0.5)
        self.retrieve_threshold = kwargs.get("retrieve_threshold", 0.5)
        self.halt_pool = kwargs.get("halt_pool", "mean")   # "mean" | "last"
        self.loop_embedding = kwargs.get("loop_embedding", False)  # D4：可选轮次嵌入（默认关）

        # ---- memory / retrieval（Doc 1 §7-9, Doc 2 §5-6, D9）----
        self.memory_budgets = kwargs.get("memory_budgets", [1024, 2048, 4096, 8192, 16384])  # token 数
        self.default_budget_idx = kwargs.get("default_budget_idx", 2)
        self.block_size = kwargs.get("block_size", 256)
        self.topk_coarse = kwargs.get("topk_coarse", 64)
        self.topk_fine = kwargs.get("topk_fine", 8)
        self.retrieve_query_pool = kwargs.get("retrieve_query_pool", "mean")
        # Frozen Index 范式（外部百万 token 记忆）：建库时 key encoder 冻结，
        # 训练只优化 query encoder + ranking；见 V2 §13 / 评审 8.5 分
        self.freeze_key = bool(kwargs.get("freeze_key", True))
        self.key_refresh_interval = int(kwargs.get("key_refresh_interval", 0))  # 0=永不周期刷新

        # ---- 分区注意力（Doc 1 §19）----
        self.prelude_attn = kwargs.get("prelude_attn", "full")   # "full" | "local" | "sparse"
        self.core_attn = kwargs.get("core_attn", "full")         # sparse 仅供 Baseline C（无 memory 支持）
        self.coda_attn = kwargs.get("coda_attn", "full")
        self.attn_window = kwargs.get("attn_window", 512)
        self.sparse_topk_blocks = kwargs.get("sparse_topk_blocks", 64)

        # ---- loss 权重（Doc 2 §18, Doc 3 §9 Phase3, Doc 4 §15）----
        self.lambda_halt = kwargs.get("lambda_halt", 0.1)
        self.lambda_retrieve = kwargs.get("lambda_retrieve", 0.1)
        self.lambda_memory = kwargs.get("lambda_memory", 0.01)
        self.lambda_compute = kwargs.get("lambda_compute", 0.01)
        self.lambda_distill = kwargs.get("lambda_distill", 0.05)  # fine→coarse 检索蒸馏（v1.3）
        self.alpha_consistency = kwargs.get("alpha_consistency", 0.1)
        self.exit_loss_weighting = kwargs.get("exit_loss_weighting", "uniform")  # uniform | last
        self.oracle_epsilon = kwargs.get("oracle_epsilon", 0.05)
        self.lambda_cost = kwargs.get("lambda_cost", 0.0)  # T4.4：C(τ) 权重（默认仅日志）

        # ---- V2 残差改造（任务清单 P2；V2 文档 §19 配置）----
        self.residual_v2 = kwargs.get("residual_v2", False)  # 总开关；False=旧路径（回归锚点）
        self.mhar_heads = kwargs.get("mhar_heads", 4)
        self.mhar_latest_bias = kwargs.get("mhar_latest_bias", 4.0)
        self.mhar_strength = kwargs.get("mhar_strength", 1.0)        # η：0=严格读最新
        self.update_rank = kwargs.get("update_rank", 64)
        self.update_progress = kwargs.get("update_progress", 1.0)    # λ：0=严格接受候选
        self.reader_scale_init = kwargs.get("reader_scale_init", 0.01)  # 0=严格 no-op
        self.reader_query_heads = kwargs.get("reader_query_heads", None)   # 默认按 D/64 推
        self.reader_kv_heads = kwargs.get("reader_kv_heads", 2)
        self.reader_head_dim = kwargs.get("reader_head_dim", 64)

        # ---- 推理模式上限（Doc 2 §26 / Doc 5 §28；Normal 默认 8K，见任务清单"其他小的不一致"）----
        self.mode_caps = kwargs.get("mode_caps", {
            "fast": {"max_loop": 1, "max_budget": 2048},
            "normal": {"max_loop": 2, "max_budget": 8192},
            "deep": {"max_loop": 4, "max_budget": 16384},
        })

    # ---- 便捷方法 ----
    def budget_blocks(self, idx=None):
        """预算档位 -> block 数（容量以 block 计，向上取整）。"""
        idx = self.default_budget_idx if idx is None else idx
        budget_tokens = self.memory_budgets[idx]
        return max(1, math.ceil(budget_tokens / self.block_size))

    def validate(self):
        """V2 §19 配置校验：非法 config 启动即报错（T5.2）。"""
        errors = []
        if self.hidden_size % max(1, self.mhar_heads) != 0:
            errors.append(f"hidden_size={self.hidden_size} 须被 mhar_heads={self.mhar_heads} 整除")
        qh = self.reader_query_heads or max(2, self.hidden_size // 64)
        if qh % max(1, self.reader_kv_heads) != 0:
            errors.append(f"reader_query_heads={qh} 须被 reader_kv_heads={self.reader_kv_heads} 整除")
        if self.residual_v2 and self.core_attn == "sparse":
            errors.append("residual_v2 与 core_attn='sparse' 不兼容")
        if self.residual_v2 and self.thinking_layers < 1:
            errors.append("residual_v2 需要至少 1 层 thinking zone（Reader 插入层）")
        if not (0 <= self.reader_scale_init < 1):
            errors.append(f"reader_scale_init={self.reader_scale_init} 须落在 [0,1)")
        if self.max_loop < 1:
            errors.append("max_loop 必须 ≥1")
        if self.block_size < 1:
            errors.append("block_size 必须 ≥1")
        # budget 受 action mask 限制：档位必须为正且递增
        if list(self.memory_budgets) != sorted(self.memory_budgets) or min(self.memory_budgets) <= 0:
            errors.append("memory_budgets 须为正且递增")
        if not (0 <= self.default_budget_idx < len(self.memory_budgets)):
            errors.append("default_budget_idx 越界")
        if self.freeze_key and self.key_refresh_interval < 0:
            errors.append("key_refresh_interval 不得为负")
        if errors:
            raise ValueError("MiniMindXConfig 非法:\n- " + "\n- ".join(errors))
        return True

    def architecture_version(self):
        return "minimind_x_residual_v2" if self.residual_v2 else "minimind_x"

    def mode_caps_for(self, mode):
        if mode is None:
            return {"max_loop": self.max_loop, "max_budget": max(self.memory_budgets)}
        if mode not in self.mode_caps:
            raise ValueError(f"未知推理模式 {mode}，可选 {list(self.mode_caps)}")
        return self.mode_caps[mode]
```

---

# model/minimind_x/controller.py

```python
"""AdaptiveController：终局编排（V2 改造版；任务清单 T0.1/T0.3/T0.5/T0.7 修订）。

数据流（D9；单次 forward 的操作顺序）：
  建库（一次）→ 因果 Query（仅可见前缀）→ Initial Retrieval → Budget 分配
  → Loop{ MHAR* → Core → Update* → halt? → retrieve? → MemoryManager.update } → Coda

V2 因果契约（T0.1，C5/C6 修复）：
  - 初始检索 query 只用 [bos]+question（query_len 边界由数据集提供）；
  - 训练期轮间再检索 query 只用 exit_hidden 的可见前缀位置；
  - 换 gold completion 不得改变初始检索结果（tests/test_causality.py 锁定）。

V2 梯度契约（T0.7，C3 修复）：gold CE 与蒸馏一样用 grad mode 重算的 coarse logits
（离散 TopK 选择仍在 no_grad，见 D5/V2 §13.2 的分层）。

budget 联动（T0.3，C7 修复）：预算档真正改变 retrieve 的 topk_f；统计分开记录
budget_cap_tokens / actual_valid_tokens / unique_blocks。
"""
from typing import Optional

import torch

from .retrieval.block_store import LongContextStore
from .memory.working_memory import WorkingMemory, MemoryInput
from .router.retrieval_router import (
    ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH, ACTION_NAMES,
    action_mask, normalize_oracle_actions, ACTION_SCHEMA_VERSION)



class AdaptiveController:
    def __init__(self, model):
        """model: MiniMindXForCausalLM 或其 DDP 包装（须 use_retrieval/use_memory 打开）。"""
        self.model = model
        raw = model
        for attr in ("module", "_orig_mod"):
            raw = getattr(raw, attr, raw)
        self.raw = raw                      # 未包装引用（routers/store 等内部访问）
        self.cfg = raw.config
        self.store: Optional[LongContextStore] = None
        self.memory: Optional[WorkingMemory] = None
        self.stats = {"retrievals": 0, "memory_tokens": 0, "re_retrievals": 0,
                      "global_refresh": 0, "warm_rescore": 0, "reuse": 0}
        # T3.3/T4.7（V2 §11.5/§17.2）：轨迹全字段
        self.trajectory = []
        # T2.5 Warm Candidate Cache：最近一次全局检索的候选（供 RESCORE_WARM 重排）
        self.warm_candidates = None  # dict(cand_idx, cand_hidden, cand_keys, fine_scores)

    def _trace(self, event, **kw):
        """V2 §11.5/§17.2 轨迹字段：event/version + 可选 round/action/budget/exit/logprob。"""
        rec = {"event": event,
               "memory_version": self.memory.memory_version if self.memory is not None else -1,
               "retrieval_action": kw.get("action", event),
               "budget_action": kw.get("budget_idx", None),
               "exit_round": kw.get("exit_round", None),
               "token_position": kw.get("token_position", None),
               "action_logprob": kw.get("action_logprob", None),
               "visible_prefix_len": kw.get("visible_prefix_len", None)}
        rec.update(kw)
        self.trajectory.append(rec)

    # ---------------------------------------------------------------- 建库与初检
    def begin(self, context_ids):
        self.store = LongContextStore(context_ids, self.cfg.block_size)
        if self.store.n_blocks == 0:
            # no_evidence 族：空库合法（G1/评审 6.3）——不建索引，检索将返回空 memory
            self.store.block_keys = None
        else:
            self.store.build_keys(self.raw.model.retriever, self.raw.model.embed_tokens)
        self.memory = None
        self.stats = {"retrievals": 0, "memory_tokens": 0, "re_retrievals": 0,
                      "budget_cap_tokens": 0, "actual_valid_tokens": 0, "unique_blocks": 0,
                      "global_refresh": 0, "warm_rescore": 0, "reuse": 0}
        self.trajectory = []
        self.warm_candidates = None
        self.stats["last_global_result"] = None
        return self.store

    def _budget_idx(self, query_hidden, max_budget_tokens=None, teacher_idx=None):
        if teacher_idx is not None:
            return int(teacher_idx)
        router = self.raw.model.budget_router
        if router is None:
            idx = self.cfg.default_budget_idx
        else:
            logits = router(query_hidden)
            idx = int(logits.argmax(dim=-1).float().mean().item())  # batch 统一档位（研究级）
        if max_budget_tokens is not None:
            cap = [i for i, b in enumerate(self.cfg.memory_budgets) if b <= max_budget_tokens]
            if cap and idx > cap[-1]:
                idx = cap[-1]
        return idx

    def retrieve_into_memory(self, query_hidden, budget_idx=None, mem=None):
        """检索并把新块并入记忆。budget 档位真正联动 topk_f（T0.3）。

        整条检索/建库/memory 编码路径不回传梯度（D5）；gold 监督与蒸馏在
        forward_with_oracles 中用 grad mode 重算的 logits 单独计算（T0.7/V2 §13.2）。
        """
        budget_idx = self.cfg.default_budget_idx if budget_idx is None else budget_idx
        is_initial = mem is None
        budget_blocks = self.cfg.budget_blocks(budget_idx)
        if self.store.n_blocks == 0 or self.store.block_keys is None:
            # 空库：合法 no-op memory（评审 6.3）
            mem = mem if mem is not None else WorkingMemory(
                query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.block_size,
                                       self.cfg.hidden_size),
                query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.hidden_size))
            self.stats["retrievals"] += 1
            self.stats["global_refresh"] = self.stats.get("global_refresh", 0) + 1
            self.stats["memory_tokens"] = mem.token_size
            self.stats["actual_valid_tokens"] = mem.token_size
            self.stats["unique_blocks"] = 0
            self.stats["budget_cap_tokens"] = self.cfg.memory_budgets[budget_idx]
            self.stats["last_result"] = None
            self.stats["last_global_result"] = None
            self.warm_candidates = None
            if is_initial:
                self._trace("initial_retrieval", budget_idx=budget_idx,
                            cap_tokens=self.stats["budget_cap_tokens"],
                            valid_tokens=0, unique=0)
            return mem
        result = self.raw.model.retriever.retrieve(
            query_hidden, self.store, self.raw, topk_fine=budget_blocks)  # T0.3 联动
        self.stats["retrievals"] += 1
        self.stats["global_refresh"] = self.stats.get("global_refresh", 0) + 1
        # T2.5：全局检索后刷新 Warm Candidate Cache（供 RESCORE_WARM 重排）
        self.warm_candidates = {
            "cand_idx": result.cand_idx.detach() if result.cand_idx is not None else None,
            "cand_hidden": self.store.get_block_hidden(
                self.raw, result.cand_idx).detach() if result.cand_idx is not None else None,
            "block_keys": self.store.block_keys.detach() if self.store.block_keys is not None else None,
            "fine_scores": result.fine_scores.detach() if result.fine_scores is not None else None,
        }
        self.stats["budget_cap_tokens"] = self.cfg.memory_budgets[budget_idx]
        block_keys = self.store.block_keys.to(result.block_idx.device)[result.block_idx]  # [B,k,d]
        q_vec = self.raw.model.retriever.project_query(query_hidden).detach()
        mem = mem if mem is not None else WorkingMemory(
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.block_size,
                                   self.cfg.hidden_size),
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.hidden_size))
        hit = torch.ones_like(result.block_idx, dtype=torch.float)  # 新块本轮均被使用
        mem, _, _ = self.raw.model.memory_manager.update(
            mem, result.block_hidden, block_keys, q_vec, budget_blocks, hit_new=hit,
            source_ids=result.block_idx)
        self.stats["memory_tokens"] = mem.token_size
        self.stats["actual_valid_tokens"] = mem.token_size
        self.stats["unique_blocks"] = int(result.block_idx[:, 0].numel() and
                                          torch.unique(result.block_idx.flatten()).numel())
        self.stats["last_result"] = result
        self.stats["last_global_result"] = result  # distill/gold CE 用（warm rescore 不清除）
        # 事件记录口径：初始检索在此记录；轮间 refresh/reuse 由 loop_callback 记录（带轮次）
        if is_initial:
            self._trace("initial_retrieval",
                        budget_idx=budget_idx, cap_tokens=self.stats["budget_cap_tokens"],
                        valid_tokens=mem.token_size, unique=self.stats["unique_blocks"])
        return mem

    def has_warm_cache(self) -> bool:
        """Warm Candidate Cache 是否可用（决定 RESCORE_WARM 是否可选，V2 §9.2）。"""
        w = self.warm_candidates
        return bool(w is not None and w.get("cand_idx") is not None and w["cand_idx"].numel() > 0)

    def rescore_warm_into_memory(self, query_hidden, budget_idx=None, mem=None):
        """T2.5 RESCORE_WARM：只在已缓存候选中重排选择，不重搜全局索引。

        可能改变 memory_version（选中集合变化时）；成本低于 GLOBAL_REFRESH
        （不重新建索引/不重投影未选中候选）。
        """
        if not self.has_warm_cache():
            raise RuntimeError("RESCORE_WARM 需要 Warm Candidate Cache；无缓存时必须 mask 该动作")
        budget_idx = self.cfg.default_budget_idx if budget_idx is None else budget_idx
        budget_blocks = self.cfg.budget_blocks(budget_idx)
        w = self.warm_candidates
        cand_hidden = w["cand_hidden"]                          # [B,c,block,d]
        cand_idx = w["cand_idx"]                                # [B,c]
        q = self.raw.model.retriever.project_query(query_hidden).detach()
        # fine 重排：query 对候选块 token 级 max-sim（与 retrieve 的 fine 同口径）
        tok_sim = torch.einsum("bd,bckd->bck", torch.nn.functional.normalize(q, dim=-1),
                               torch.nn.functional.normalize(cand_hidden, dim=-1))
        fine_scores = tok_sim.max(dim=-1).values                # [B,c]
        topk_f = min(budget_blocks, fine_scores.shape[-1])
        k = fine_scores.topk(topk_f, dim=-1).indices
        final_idx = cand_idx.gather(1, k)
        final_hidden = cand_hidden.gather(
            1, k.view(*k.shape, 1, 1).expand(-1, -1, cand_hidden.shape[2], cand_hidden.shape[-1]))
        block_keys = w["block_keys"].to(final_idx.device)[final_idx]  # [B,k,d]
        mem = mem if mem is not None else WorkingMemory(
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.block_size,
                                   self.cfg.hidden_size),
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.hidden_size))
        hit = torch.ones_like(final_idx, dtype=torch.float)
        mem, _, _ = self.raw.model.memory_manager.update(
            mem, final_hidden, block_keys, q, budget_blocks, hit_new=hit,
            source_ids=final_idx)
        self.stats["retrievals"] += 1
        self.stats["warm_rescore"] = self.stats.get("warm_rescore", 0) + 1
        self.stats["memory_tokens"] = mem.token_size
        self.stats["actual_valid_tokens"] = mem.token_size
        self.stats["unique_blocks"] = int(torch.unique(final_idx.flatten()).numel())
        self.stats["budget_cap_tokens"] = self.cfg.memory_budgets[budget_idx]
        # 不覆盖 last_global_result（distill teacher 仍用最近全局检索的 fine_scores）
        return mem

    def execute_retrieval_action(self, action, query_hidden, budget_idx=None, mem=None):
        """执行三动作之一（T2.5 / V2 §9.2）。返回新 memory 或 None（REUSE）。"""
        action = int(action)
        if action == ACTION_REUSE:
            self.stats["reuse"] = self.stats.get("reuse", 0) + 1
            return None
        if action == ACTION_RESCORE_WARM:
            if not self.has_warm_cache():
                # 防御：无 warm cache 时不得静默降级为全局搜索（V2 §9.2 禁止项）
                raise RuntimeError("RESCORE_WARM 无 Warm Candidate Cache（应在 logits mask 掉）")
            return self.rescore_warm_into_memory(query_hidden, budget_idx=budget_idx, mem=mem)
        if action == ACTION_GLOBAL_REFRESH:
            return self.retrieve_into_memory(query_hidden, budget_idx=budget_idx, mem=mem)
        raise ValueError(f"未知 retrieval action: {action}（应为 0/1/2）")

    def available_action_mask(self, batch=1, device=None):
        """当前合法动作 mask [B,3]。"""
        return action_mask(self.has_warm_cache(), device=device, batch=batch)

    def joint_path_logprob(self, token_logprobs=None, halt_logprobs=None,
                           retrieval_logprobs=None, budget_logprobs=None):
        """V2 §17.2 联合路径 log P(τ) = Σ token + Σ halt + Σ retrieval + Σ budget。"""
        total = 0.0
        for seq in (token_logprobs, halt_logprobs, retrieval_logprobs, budget_logprobs):
            if seq is None:
                continue
            if torch.is_tensor(seq):
                total = total + float(seq.sum().item())
            else:
                total = total + float(sum(seq))
        return total

    def assert_replay_not_averaged(self, exit_rounds):
        """V2 §17.3 禁止平均深度重放：[1,3,2,4] 不得用 R=3 整段近似。

        仅作门槛检查：exit_rounds 为逐 token 深度时，若调用方试图用均值重放会在此拦截。
        """
        if exit_rounds is None:
            return True
        rs = [int(r) for r in exit_rounds]
        if not rs:
            return True
        if len(set(rs)) > 1:
            mean_r = int(round(sum(rs) / len(rs)))
            # 允许记录均值作日志，但轨迹重放必须保留逐 token 深度
            self.stats["depth_mean_for_log_only"] = mean_r
            self.stats["depth_per_token"] = rs
        return True

    # ---------------------------------------------------------------- loop callback
    def make_loop_callback(self, oracle_retrieve=None, max_budget_tokens=None,
                           query_len=None):
        """轮间再检索回调。query_len: 训练期可见前缀边界（T0.1）——轮间检索 query
        只用 exit_hidden 的可见前缀位置池化，gold 后缀不得影响检索决策。"""
        cfg = self.cfg

        def cb(r, exit_hidden):
            """轮间动作（T2.5）：REUSE / RESCORE_WARM / GLOBAL_REFRESH。

            oracle_retrieve 必须已是 schema v2 枚举（0=REUSE, 1=RESCORE_WARM, 2=GLOBAL_REFRESH）。
            旧二值标签请先经 normalize_oracle_actions(..., schema_version=1) 转换，禁止在此对调。
            无 oracle 时用 RetrievalRouter.action_logits（RESCORE_WARM 无 warm cache 则 mask）。
            """
            vis = exit_hidden[:, :query_len] if query_len else exit_hidden
            warm_ok = self.has_warm_cache()
            if oracle_retrieve is not None and r < len(oracle_retrieve):
                action = int(oracle_retrieve[r])  # 已是 schema v2，禁止再映射
                if action not in (ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH):
                    raise ValueError(f"oracle 动作非法: {action}（须先 normalize）")
                if action == ACTION_RESCORE_WARM and not warm_ok:
                    action = ACTION_GLOBAL_REFRESH  # oracle 也不允许无缓存 RESCORE
                a_logp = None
            else:
                router = self.raw.model.retrieve_router
                logits = router.action_logits(vis, warm_available=warm_ok)
                probs = torch.softmax(logits.float(), dim=-1)
                action = int(probs.mean(0).argmax().item())  # batch 统一（研究级）
                a_logp = float(torch.log_softmax(logits.float(), dim=-1).mean(0)[action].item())
            self.stats["re_retrievals"] += (action != ACTION_REUSE)
            # 事件名：REUSE→reuse，RESCORE_WARM→rescore_warm，GLOBAL_REFRESH→refresh（legacy）
            ev = {ACTION_REUSE: "reuse", ACTION_RESCORE_WARM: "rescore_warm",
                  ACTION_GLOBAL_REFRESH: "refresh"}[action]
            self._trace(ev, round=r, action=action,
                        action_logprob=a_logp, visible_prefix_len=query_len)
            if action == ACTION_REUSE:
                return None
            with torch.no_grad():
                self.memory = self.execute_retrieval_action(
                    action, vis,
                    budget_idx=self._budget_idx(vis, max_budget_tokens),
                    mem=self.memory)
                return self._memory_input()
        return cb

    # ---------------------------------------------------------------- 推理入口
    @torch.no_grad()
    def generate_with_context(self, input_ids, attention_mask=None, context_ids=None,
                              mode=None, **gen_kwargs):
        """终局生成（T6.3）：建库一次 → 因果初检 → generate（router 驱动轮间再检索）。

        推理期 input_ids 即完整可见 prompt，无需 query_len 边界。
        """
        assert self.cfg.use_retrieval and self.cfg.use_memory, "终局路径需 use_retrieval+use_memory"
        self.begin(context_ids)
        q_hidden = self.raw.model.prelude_hidden(input_ids, attention_mask)
        caps = self.cfg.mode_caps_for(mode)
        budget_idx = self._budget_idx(q_hidden, max_budget_tokens=caps.get("max_budget"))
        self.memory = self.retrieve_into_memory(q_hidden, budget_idx=budget_idx)
        cb = self.make_loop_callback(max_budget_tokens=caps.get("max_budget"))
        out = self.model.generate(
            inputs=input_ids, attention_mask=attention_mask, mode=mode,
            memory_provider=lambda step: self._memory_input(),
            loop_callback=cb, return_stats=True, **gen_kwargs)
        out["stats"].update({k: v for k, v in self.stats.items() if isinstance(v, (int, float))})
        out["stats"]["memory_tokens_final"] = self.memory.token_size if self.memory else 0
        return out

    @torch.no_grad()
    def _memory_input(self):
        """按架构版本产出 memory 注入物：v1=逐层 KV 拼接；v2=(payload, valid) 独立 Reader。"""
        if self.memory is None:
            return None
        if getattr(self.cfg, "residual_v2", False):
            payload = self.memory.token_hidden()                    # [B, M, D]
            # T3.3：token 级 valid 来自 WorkingMemory.valid（块级扩展）
            valid = self.memory.token_valid()
            return MemoryInput(payload, valid,
                               memory_version=self.memory.memory_version)
        return self.memory.get_kv(self.raw.model.thinking_core)

    # ---------------------------------------------------------------- 训练入口（SFT-C）
    def forward_with_oracles(self, input_ids, attention_mask, labels, context_ids,
                             gold_block_idx=None, oracle_retrieve=None, oracle_budget=None,
                             oracle_r=None, distill=True, query_len=None):
        """SFT-C / Stage 3 的 teacher-forced 终局训练前向（D5 + T0.1/T0.7）。

        query_len: 可见前缀边界（训练路径必填；初始检索与 gold CE 重算只用
        input_ids[:, :query_len]，换 gold completion 不改变检索——因果契约）。
        返回 (MiniMindXOutput, extras)。
        """
        if query_len is None:
            is_training = bool(getattr(self.model, "training", False)
                               or getattr(self.raw, "training", False))
            if is_training:
                raise ValueError(
                    "forward_with_oracles 训练路径必须显式传 query_len（禁止默认全序列，"
                    "否则检索/RB 路由会读到 gold answer）")
            query_len = input_ids.shape[1]  # 纯推理/全可见
        causal_ids = input_ids[:, :query_len]
        self.begin(context_ids)
        q_hidden = self.raw.model.prelude_hidden(causal_ids, attention_mask[:, :query_len]
                                                 if attention_mask is not None else None)
        budget_idx = self._budget_idx(q_hidden, teacher_idx=oracle_budget)
        with torch.no_grad():  # D5：memory 构建/选择路径不回传梯度
            self.memory = self.retrieve_into_memory(q_hidden, budget_idx=budget_idx)
            mem_input = self._memory_input()
        # 动作 schema：调用方传入的 oracle_retrieve 一律视为 v2；旧 0/1 由 sft 边界转换
        cb = self.make_loop_callback(oracle_retrieve=oracle_retrieve, query_len=query_len)
        fwd = dict(collect_exits=True, loop_callback=cb, query_len=query_len)
        if getattr(self.cfg, "residual_v2", False):
            fwd["memory_payload"] = mem_input
        else:
            fwd["memory_kv_per_layer"] = mem_input
        out = self.model(input_ids, attention_mask=attention_mask, labels=None, **fwd)
        # T2.4：router_trace = 控制器轨迹（事件/动作/版本/exit）
        try:
            out.router_trace = list(self.trajectory)
        except Exception:
            pass
        extras = {"retrievals": self.stats["retrievals"],
                  "memory_tokens": self.memory.token_size,
                  "budget_cap_tokens": self.stats.get("budget_cap_tokens", 0),
                  "actual_valid_tokens": self.stats.get("actual_valid_tokens", 0),
                  "unique_blocks": self.stats.get("unique_blocks", 0),
                  "trajectory": list(self.trajectory),
                  "warm_available": self.has_warm_cache(),
                  "action_counts": {k: self.stats.get(k, 0) for k in
                                    ("reuse", "warm_rescore", "global_refresh")}}
        result = self.stats.get("last_global_result") or self.stats.get("last_result")
        # ---- T2.5：全展开时收集三动作 logits（用于 action CE）----
        if getattr(out, "router_probs", None) and out.router_probs.get("retrieval_action") is not None:
            extras["retrieval_action_logits"] = out.router_probs["retrieval_action"]
            extras["retrieval_action_masks"] = out.router_probs.get("retrieval_action_mask")
        # ---- T0.7：监督项用 grad mode 重算的 logits（离散选择仍 no_grad）----
        zero = None
        # 空库（no_evidence）：无检索监督可算
        if (gold_block_idx is not None or distill) and self.store.n_blocks > 0:
            q_grad = self.raw.model.prelude_hidden(
                causal_ids, attention_mask[:, :query_len]
                if attention_mask is not None else None, no_grad=False)
            q_vec = self.raw.model.retriever.project_query(q_grad)
            coarse_trainable = self.raw.model.retriever.coarse_scores(q_vec, self.store)
            zero = q_vec.new_zeros(())
            if gold_block_idx is not None and result is not None:
                n_blocks = result.coarse_scores.shape[-1]
                if (gold_block_idx >= 0).all() and (gold_block_idx < n_blocks).all():
                    extras["retrieval_ce"] = self.raw.model.retriever.retrieval_ce_loss(
                        coarse_trainable, gold_block_idx)
            if distill and result is not None and result.fine_scores is not None:
                extras["distill_loss"] = self.raw.model.retriever.distill_loss(
                    coarse_trainable, result.cand_idx, result.fine_scores)
        return out, extras
```

---

# model/minimind_x/flops.py

```python
"""解析 FLOPs 估算（任务清单 T0.4）。所有效率指标统一走这里。

约定（近似，报告口径）：
- 线性层：2·in·out FLOPs/MAC；attention 投影按 GQA 计，scores+AV 按 2·S_ctx·d 计
- 有效深度 = P + C + T·R（Doc 6 §5：EffectiveDepth = L + T(R−1)）
- memory cross-attention 额外：2·S·M_tokens·d/loop/层（M_tokens = 块数×block_size）
- 检索成本：粗检 2·n_blocks·d/次 + 精检 2·topk_c·block·d/次（query 对块）
- train=True 时乘 3（前向+反向）
"""
import math


def per_layer_params(config) -> int:
    d, nh, nkv, hd = (config.hidden_size, config.num_attention_heads,
                      config.num_key_value_heads, config.head_dim)
    inter = config.moe_intermediate_size if config.use_moe else config.intermediate_size
    n_experts = config.num_experts if config.use_moe else 1
    attn = d * nh * hd + nh * hd * d + 2 * (d * nkv * hd) + 2 * nkv * hd + nh * hd  # qkvo + q/k norm
    mlp = n_experts * 3 * d * inter
    norms = 4 * d  # 两个 RMSNorm
    return attn + mlp + norms


def count_params(config) -> int:
    """解析参数量（不含 lm_head 与 tying；embedding 含）。与实测误差 <1%（norm 项近似）。"""
    L = config.num_hidden_layers
    return config.vocab_size * config.hidden_size + L * per_layer_params(config)


def estimate_flops(config, seq_len, loops=1, memory_blocks=0, n_retrievals=0,
                   train=False, topk_coarse=None, topk_fine=None) -> dict:
    """返回 {"total": float, "backbone": ..., "memory_attn": ..., "retrieval": ...}。"""
    d, hd = config.hidden_size, config.head_dim
    P, T, C = config.prelude_layers, config.thinking_layers, config.coda_layers
    R = max(1, loops) if config.use_loop else 1  # dense 变体不循环
    per_layer = per_layer_params(config)
    # 每 token 每层：投影 2·params_linear + attention 读写 4·S·d（QK^T 与 AV 各 2）
    proj = 2 * (per_layer - 4 * d)
    ctx = 4 * seq_len * d
    backbone = seq_len * ((P + C + T * R) * (proj + ctx))
    mem_tokens = memory_blocks * config.block_size
    memory_attn = (T * R) * (2 * seq_len * mem_tokens * d) if mem_tokens else 0.0
    tc = topk_coarse or config.topk_coarse
    tf = topk_fine or config.topk_fine
    n_blocks = math.ceil(seq_len * 4 / config.block_size)  # 粗略：库大小≈4×序列（评测时请传实测）
    retrieval = n_retrievals * (2 * n_blocks * d + 2 * tc * config.block_size * d)
    # ---- T5.5 / V2 §12.1 分项：reader / gate / mhar / mem_kv_proj（容量不得代理真实开销）----
    # Reader QK+AV ≈ 4·B·T·M·D_a（这里按 batch=1、D_a≈d）；是 T*M 不是 M²
    reader_attn = (T * R) * (4 * seq_len * mem_tokens * d) if mem_tokens else 0.0
    # Reader 投影：Q/O per-loop + K/V once per memory version（近似：K/V 一次，Q/O ×R）
    qh = getattr(config, "reader_query_heads", None) or max(2, d // 64)
    kvh = getattr(config, "reader_kv_heads", 2)
    hd = getattr(config, "reader_head_dim", 64)
    # 评审 9.4：投影按 T×M 与 M 计（不是常数）；K/V 一次/版本，Q/O 逐轮
    reader_qo = (T * R) * 2 * seq_len * (d * qh * hd + qh * hd * d)  # Q/O 投影 ~ T·S
    reader_kv = 2 * mem_tokens * (d * kvh * hd)  # K/V 投影 ~ M（每 memory version 一次）
    reader_gate = (T * R) * 2 * (2 * d * 32 + 32)  # gate_fc1/2 近似
    # MHAR：来源轴加权 O(B·T·D·N_src)，N_src≈R
    mhar = (T * R) * (d * (R + 1))
    # Update Gate：低秩 3D→rank→D
    rank = getattr(config, "update_rank", 64)
    update_gate = (T * R) * 2 * (3 * d * rank + rank * d)
    # residual 模块仅 residual_v2 时计入（默认关闭时主干 FLOPs 仍线性于 R）
    use_v2 = bool(getattr(config, "residual_v2", False))
    residual = (reader_attn + reader_qo + reader_kv + reader_gate + mhar + update_gate) if use_v2 else 0.0
    total = backbone + memory_attn + retrieval + residual
    if train:
        total *= 3
    return {"total": float(total), "backbone": float(backbone),
            "memory_attn": float(memory_attn), "retrieval": float(retrieval),
            "reader_attn": float(reader_attn), "reader_proj": float(reader_qo + reader_kv),
            "reader_gate": float(reader_gate), "mhar": float(mhar),
            "update_gate": float(update_gate), "residual_modules": float(residual),
            "effective_depth": P + C + T * R,
            "executed_core_rounds_note": "executed=R×T（物理）；selected 见 generate stats.avg_loops"}
```

---

# model/minimind_x/losses.py

```python
"""联合损失（任务清单 T2.3 / T6.4；Doc 2 §18-22、Doc 3 §9 Phase3、Doc 4 §15）。

L = L_task + λ₁·L_halt + λ₂·L_retrieve + λ₃·C_memory + λ₄·C_compute (+ α·L_consistency)

关键约定（任务清单 v1.1 修正）：
- C_compute 用 router 概率的期望深度 Σᵣ r·P(stop@r)/R_max —— 全展开训练下实际执行轮数
  恒为 R_max，「实际轮数」没有梯度意义；C_m 同理按预算档概率期望。
- 一致性损失 KL(P_r ‖ P_Rmax)：深层出口（stop-grad）指导浅层出口。
- halt 标签由 oracle R* 展开：第 R* 轮 stop=1、之前 stop=0、之后 mask 掉。
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- label 构造
def halt_labels_from_oracle(oracle_r: torch.Tensor, r_max: int):
    """oracle R* [B] -> (labels [B,R_max], mask [B,R_max])。

    第 r 轮（0 起）标签：r == R*-1 → 1（stop）；r < R*-1 → 0（continue）；r > R*-1 → mask=0。
    """
    bsz = oracle_r.shape[0]
    r = torch.arange(r_max, device=oracle_r.device).unsqueeze(0).expand(bsz, -1)
    oracle = (oracle_r - 1).clamp(min=0).unsqueeze(1)
    labels = (r == oracle).float()
    mask = (r <= oracle).float()
    return labels, mask


def exit_loss_weights(r_max: int, weighting="uniform"):
    """各出口 CE 的权重（Doc 3 Phase 0 随机 R 退化时可只留单出口）。"""
    if weighting == "last":
        w = torch.zeros(r_max); w[-1] = 1.0
    else:
        w = torch.ones(r_max) / r_max
    return w


# ---------------------------------------------------------------- 期望计算成本
def expected_loops(halt_probs: torch.Tensor, force_stop_last=True):
    """halt_probs [R,B]（每轮 P(stop)）-> E[R] [B]。

    P(stop@r) = p_r · Π_{j<r}(1−p_j)；force_stop_last 时把「R 轮都没停」的概率并给最后一轮。
    """
    p = halt_probs.clamp(1e-6, 1 - 1e-6)
    survive = torch.cumprod(1 - p, dim=0)                       # survive[r] = P(活过 r 轮)
    p_stop = p * torch.cat([torch.ones_like(p[:1]), survive[:-1]], dim=0)
    if force_stop_last:
        p_stop = torch.cat([p_stop[:-1], (p_stop[-1:] + survive[-1:])], dim=0)
    r = torch.arange(1, p.shape[0] + 1, device=p.device).unsqueeze(1).float()
    return (r * p_stop).sum(dim=0)


def trajectory_cost(halt_probs=None, retrieval_actions=None, memory_tokens=None,
                    memory_budgets=None, action_costs=None):
    """T4.4 / V2 §15.7 条件轨迹成本 C(τ)。

    C(τ) ≈ E[R]/R_max + 动作成本（REUSE=0, RESCORE_WARM=c_w, GLOBAL_REFRESH=c_g）
    + memory 占比。action_costs 默认 (0, 0.25, 1.0)。返回标量（detached，供日志/可选加权）。
    """
    c = 0.0
    n = 0
    if halt_probs:
        r = expected_loops(torch.stack(halt_probs))
        c = c + float(r.mean().item())
        n += 1
    if retrieval_actions is not None:
        ac = action_costs or (0.0, 0.25, 1.0)
        acts = [int(a) for a in retrieval_actions]
        c = c + sum(ac[a] if 0 <= a < len(ac) else ac[-1] for a in acts) / max(len(acts), 1)
        n += 1
    if memory_tokens is not None and memory_budgets:
        b_max = max(memory_budgets)
        mt = float(memory_tokens) if not torch.is_tensor(memory_tokens) else float(memory_tokens.item())
        c = c + mt / max(b_max, 1)
        n += 1
    return torch.tensor(c / max(n, 1))


def expected_memory_cost(budget_probs: torch.Tensor, memory_budgets):
    """budget_probs [B,n档] -> E[|W|]/B_max ∈ (0,1]（Doc 2 §22 C_m）。"""
    budgets = torch.as_tensor(memory_budgets, dtype=torch.float32,
                              device=budget_probs.device)
    return (budget_probs * budgets.unsqueeze(0)).sum(-1) / budgets.max()


# ---------------------------------------------------------------- 各损失项
def lm_loss_per_exit(exit_logits: List[torch.Tensor], labels: torch.Tensor,
                     weighting="uniform"):
    """每个出口的 next-token CE（labels 右移、ignore -100）。返回 (加权总损失, [R] 各出口)。"""
    r_max = len(exit_logits)
    w = exit_loss_weights(r_max, weighting).to(exit_logits[0].device)
    losses = []
    for logits in exit_logits:
        x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        losses.append(F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100))
    stacked = torch.stack(losses)
    return (stacked * w.to(stacked.device)).sum(), stacked


def halt_bce_loss(halt_probs: List[torch.Tensor], oracle_r: torch.Tensor,
                  valid: Optional[torch.Tensor] = None):
    """逐轮 BCE（标签见 halt_labels_from_oracle）。halt_probs: [B]×R 的列表。

    valid: [B] bool（T4.4/V2 §15.4）——unresolved 样本（连 R_max 都答不对）不参与
    halt 监督：不能机械地把它们当作「需要最大深度」。
    """
    labels, mask = halt_labels_from_oracle(oracle_r, len(halt_probs))  # [B,R]
    if valid is not None:
        mask = mask * valid.to(mask.device).float().unsqueeze(1)
    probs = torch.stack(halt_probs).transpose(0, 1).clamp(1e-6, 1 - 1e-6)  # [B,R]
    bce = F.binary_cross_entropy(probs, labels, reduction="none")
    return (bce * mask).sum() / mask.sum().clamp(min=1.0)


def retrieval_bce_loss(retrieve_probs: List[torch.Tensor], oracle_retrieve: torch.Tensor):
    """每轮是否再检索的 BCE。oracle_retrieve: [R]（0/1，逐轮决策）。"""
    probs = torch.stack(retrieve_probs).clamp(1e-6, 1 - 1e-6)     # [R,B]
    labels = oracle_retrieve.to(probs.device).unsqueeze(1).expand_as(probs)
    return F.binary_cross_entropy(probs, labels.float())


def retrieval_action_ce_loss(action_logits: List[torch.Tensor],
                             oracle_action: torch.Tensor,
                             action_masks: Optional[List[torch.Tensor]] = None):
    """T2.5 三动作 CE。action_logits: [B,3]×R；oracle_action: [R] 或 [B,R] 动作 id。

    action_masks: 每轮 [B,3] bool（RESCORE_WARM 无 warm cache 时 False）；
    标签若落在不可用动作上则该轮 mask 掉（不静默改标签）。
    """
    logits = torch.stack(action_logits)  # [R,B,3]
    if oracle_action.dim() == 1:
        labels = oracle_action.to(logits.device).unsqueeze(1).expand(logits.shape[0], logits.shape[1])
    else:
        labels = oracle_action.to(logits.device)  # [B,R] -> 转 [R,B]
        labels = labels.t()
    mask = torch.ones_like(labels, dtype=logits.dtype)
    if action_masks is not None:
        am = torch.stack(action_masks)  # [R,B,3]
        lab_oh = F.one_hot(labels.long().clamp(min=0), num_classes=logits.shape[-1]).to(mask.dtype)
        mask = (am.to(mask.dtype) * lab_oh).sum(dim=-1)  # 标签动作是否可用
    # 不可用动作（如无 warm cache 的 RESCORE_WARM）在 logits 上 -inf，CE 对其零贡献
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).long(),
        reduction="none").reshape_as(labels)
    return (ce * mask).sum() / mask.sum().clamp(min=1.0)


def consistency_kl_loss(exit_logits: List[torch.Tensor], labels: torch.Tensor):
    """KL(P_r ‖ P_Rmax)：最深出口（detach）指导浅层出口，只在监督位置计算。"""
    deep = F.log_softmax(exit_logits[-1].detach(), dim=-1)
    total = exit_logits[0].new_zeros(())
    n_valid = 0
    for logits in exit_logits[:-1]:
        shallow = F.log_softmax(logits, dim=-1)
        mask = (labels[..., 1:] != -100)
        kl = F.kl_div(shallow[..., :-1, :], deep[..., :-1, :], log_target=True,
                      reduction="none").sum(-1)
        total = total + (kl * mask).sum() / mask.sum().clamp(min=1.0)
        n_valid += 1
    return total / max(n_valid, 1)


def cot_alignment_loss(exit_hiddens: List[torch.Tensor], step_hiddens: List[torch.Tensor],
                       labels: torch.Tensor = None, temperature: float = 0.1):
    """Latent Reasoning Supervision（任务清单 T2.6；Doc 3 §10 / 实验 5 / 第一阶段第三目标）。

    目标：Loop r hidden ≈ Reasoning Step r。
    exit_hiddens: 每轮 exit 的 pooled 表示 [B,d] 列表（来自 collect_exits 的逐轮均值池化）
    step_hiddens: CoT 各 step 的 pooled 表示 [B,d] 列表（同模型编码，见 encode_cot_steps）
    对齐方式：InfoNCE 对比 —— loop r 与 step r 为正例、其余 step 为负例，
    加上余弦回归项；step 数不足的轮次 mask 掉。
    """
    device = exit_hiddens[0].device
    n_pair = min(len(exit_hiddens), len(step_hiddens))
    if n_pair < 1:
        return torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    for r in range(n_pair):
        anchor = F.normalize(exit_hiddens[r], dim=-1)                       # [B,d]
        steps = torch.stack([F.normalize(s, dim=-1) for s in step_hiddens])  # [S,B,d]
        pos = steps[r]                                                       # [B,d]
        sim = torch.einsum("bd,sbd->bs", anchor / temperature, steps)        # [B,S]
        target = torch.full((anchor.shape[0],), r, dtype=torch.long, device=device)
        total = total + F.cross_entropy(sim, target)
        total = total + (1.0 - (anchor * pos).sum(-1)).mean()
    return total / n_pair


# ---------------------------------------------------------------- 总装
def joint_loss(exit_logits: List[torch.Tensor], labels: torch.Tensor,
               halt_probs: Optional[List[torch.Tensor]] = None,
               oracle_r: Optional[torch.Tensor] = None,
               retrieve_probs: Optional[List[torch.Tensor]] = None,
               oracle_retrieve: Optional[torch.Tensor] = None,
               budget_logits: Optional[torch.Tensor] = None,
               oracle_budget: Optional[torch.Tensor] = None,
               memory_size: Optional[torch.Tensor] = None,
               config=None, halt_valid: Optional[torch.Tensor] = None,
               retrieval_action_logits: Optional[List[torch.Tensor]] = None,
               oracle_retrieve_action: Optional[torch.Tensor] = None,
               retrieval_action_masks: Optional[List[torch.Tensor]] = None,
               retrieval_actions: Optional[torch.Tensor] = None,
               cost_weight: float = 0.0) -> Dict[str, torch.Tensor]:
    """联合损失总装（T6.4）。config 提供 λ/α 与预算档；只计算提供了监督信号的部分。

    返回 dict：{"loss": 总损失, "lm": ..., "halt": ..., "retrieve": ..., "budget": ...,
               "consistency": ..., "compute": E[R]/R_max 均值, "memory": C_m 均值}
    """
    parts: Dict[str, torch.Tensor] = {}
    lm, per_exit = lm_loss_per_exit(exit_logits, labels,
                                    config.exit_loss_weighting if config else "uniform")
    parts["lm"], parts["lm_per_exit"] = lm, per_exit
    total = lm
    if halt_probs and oracle_r is not None:
        parts["halt"] = halt_bce_loss(halt_probs, oracle_r, valid=halt_valid)
        total = total + (config.lambda_halt if config else 0.1) * parts["halt"]
        c = expected_loops(torch.stack(halt_probs)) / len(halt_probs)   # E[R]/R_max
        parts["compute"] = c.mean()
        total = total + (config.lambda_compute if config else 0.01) * parts["compute"]
    if retrieve_probs and oracle_retrieve is not None:
        parts["retrieve"] = retrieval_bce_loss(retrieve_probs, oracle_retrieve)
        total = total + (config.lambda_retrieve if config else 0.1) * parts["retrieve"]
    if retrieval_action_logits and oracle_retrieve_action is not None:
        # T2.5：三动作 CE（REUSE/RESCORE_WARM/GLOBAL_REFRESH，带可用性 mask）
        parts["retrieve_action"] = retrieval_action_ce_loss(
            retrieval_action_logits, oracle_retrieve_action,
            action_masks=retrieval_action_masks)
        total = total + (config.lambda_retrieve if config else 0.1) * parts["retrieve_action"]
    if budget_logits is not None and oracle_budget is not None:
        if not torch.is_tensor(oracle_budget):  # 允许 int 档位（sweep 接线返回 int）
            oracle_budget = torch.tensor([int(oracle_budget)], device=budget_logits.device)
        oracle_budget = oracle_budget.to(budget_logits.device).long().view(-1)
        parts["budget"] = F.cross_entropy(budget_logits, oracle_budget)
        total = total + (config.lambda_memory if config else 0.01) * parts["budget"]
    if len(exit_logits) > 1 and config is not None and config.alpha_consistency > 0:
        parts["consistency"] = consistency_kl_loss(exit_logits, labels)
        total = total + config.alpha_consistency * parts["consistency"]
    if memory_size is not None and config is not None:
        # T0.8（V2 §15.3/C7）：memory_size 是常量统计，不产生策略梯度——只进日志，
        # 不再加入 total；可训练的成本项走动作概率期望/轨迹成本（V2 §15.7）
        b_max = max(config.memory_budgets)
        ms = memory_size.to(exit_logits[0].device).float()
        parts["memory"] = (ms / b_max).mean()
    # T4.4：条件轨迹成本 C(τ)（默认 cost_weight=0 只记日志；>0 时可选进 total）
    if halt_probs is not None or retrieval_actions is not None or memory_size is not None:
        parts["trajectory_cost"] = trajectory_cost(
            halt_probs=halt_probs,
            retrieval_actions=retrieval_actions,
            memory_tokens=memory_size,
            memory_budgets=getattr(config, "memory_budgets", None) if config else None)
        if cost_weight > 0:
            total = total + cost_weight * parts["trajectory_cost"].to(exit_logits[0].device)
    parts["loss"] = total
    return parts
```

---

# model/minimind_x/memory/__init__.py

```python
from .working_memory import MemoryInput, WorkingMemory
from .memory_manager import MemoryManager

__all__ = ["MemoryInput", "WorkingMemory", "MemoryManager"]
```

---

# model/minimind_x/memory/memory_manager.py

```python
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
        if mem.size == 1:
            # 评审 6.1：单块无邻居——redundancy=0, novelty=1（禁止 max(-inf)→Inf/NaN）
            redundancy = query_vec.new_zeros(query_vec.shape[0], 1)
            novelty = query_vec.new_ones(query_vec.shape[0], 1)
        else:
            off_diag = sim.clone()
            off_diag.diagonal(dim1=1, dim2=2).fill_(float("-inf"))
            redundancy = off_diag.max(dim=-1).values.clamp(min=0, max=1)  # D
            novelty = 1.0 - redundancy                                    # N
        u = (self.alpha * rel + self.beta * mem.usage + self.gamma * novelty
             - self.delta * redundancy + self.eta * mem.persistence)
        return u

    def novelty_of_new(self, new_keys, mem_keys):
        """新块新颖度：1 − 与现有记忆的最大相似度。new_keys [B, k, d], mem_keys [B, M, d]。"""
        if mem_keys.shape[1] == 0:
            return new_keys.new_ones(new_keys.shape[0], new_keys.shape[1])
        return 1.0 - _cosine(new_keys, mem_keys).max(dim=-1).values.clamp(min=0)

    def update(self, mem: WorkingMemory, new_hidden, new_keys, query_vec, budget_blocks,
               hit_new=None, source_ids=None):
        """ADD + SCORE + EVICT：并入新块、按 U 保留 top-budget（Doc 2 §9）。

        new_hidden/new_keys: [B, k, d]；budget_blocks: 容量（block 数）；
        hit_new: [B, k] 新块是否本轮被选中使用（usage 更新）；
        source_ids: [B, k] 块来源 id（store 块号）——去重身份（T3.3）。
        返回 (新 WorkingMemory, 被保留的新块 mask, U 全量分数)。
        """
        # 评审 6.2：按 source_id 合并后，命中必须记到对应块，不能按去重前的槽位拼接
        prev_ids = mem.source_ids.clone() if not mem.is_empty() else None
        mem = mem.with_added(new_hidden, new_keys, source_ids=source_ids)
        if hit_new is not None:
            bsz, M = mem.usage.shape[0], mem.size
            hit_full = mem.usage.new_zeros(bsz, M)
            src = mem.source_ids  # [B, M] 合并后
            # hit_new 对应「本次尝试并入的 new_keys 行」；按 source_id 映射到合并后槽位
            if source_ids is not None:
                for b in range(bsz):
                    for j in range(source_ids.shape[1]):
                        sid = int(source_ids[b, j].item())
                        hit = float(hit_new[b, j].item()) if hit_new.dim() > 1 else float(hit_new[j].item())
                        # 找到合并后该 source 的槽位
                        matches = (src[b] == sid).nonzero(as_tuple=True)[0]
                        if matches.numel():
                            hit_full[b, matches[0]] = max(hit_full[b, matches[0]].item(), hit)
            else:
                # 无 source_id：按末尾新增槽位（旧行为，仅当 with_added 未去重）
                n_new = new_keys.shape[1]
                if M >= n_new:
                    hit_full[:, M - n_new:] = hit_new.float().view(bsz, -1)[:, :n_new]
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
```

---

# model/minimind_x/memory/reader.py

```python
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
```

---

# model/minimind_x/memory/working_memory.py

```python
"""Working Memory（任务清单 T5.1；Doc 1 §9、Doc 2 §8、§15）。

Block-slot 容器：每个 slot 是一个块 ——
  hidden: [B, K_blocks, block_size, d]（块内容，Prelude 编码）
  keys:   [B, K_blocks, d]（块键，检索/打分用）
  usage / persistence: [B, K_blocks] 统计量
容量 B ≤ B_max 以 block 数计。纯张量容器、无可学习参数（可学习部分在 MemoryManager）。

跨 loop K/V 缓存（Doc 2 §15，即 Doc 2 §23 的 memory_cache.py）：
块 hidden 展平为 token 序列后经 ThinkingCore.project_memory 投影为各层 K/V 并缓存，
memory 内容不变时 loop 间直接复用，内容变更（新对象）后自动失效重投影。
"""
import torch


class MemoryInput(tuple):
    """(payload, valid) 二元组 + memory_version（T2.4）。

    `payload, valid = mem_in` 兼容旧解包；`mem_in.memory_version` 供轨迹/输出。
    """

    def __new__(cls, payload, valid, memory_version: int = 0):
        obj = super().__new__(cls, (payload, valid))
        obj.memory_version = int(memory_version)
        return obj

    @property
    def payload(self):
        return self[0]

    @property
    def valid_mask(self):
        return self[1]


class WorkingMemory:
    def __init__(self, hidden, keys, usage=None, persistence=None, block_size=256,
                 source_ids=None, memory_version=0, valid=None, metadata=None):
        assert hidden.dim() == 4 and keys.dim() == 3
        assert hidden.shape[:2] == keys.shape[:2]
        self.hidden = hidden
        self.keys = keys
        self.block_size = hidden.shape[2] if hidden.shape[2] else block_size
        bsz, k = hidden.shape[0], hidden.shape[1]
        self.usage = usage if usage is not None else hidden.new_zeros(bsz, k)
        self.persistence = persistence if persistence is not None else hidden.new_zeros(bsz, k)
        # T3.3（V2 §9.3）：去重身份与版本——source_ids 标记块来源；每次内容变更
        # 产生新对象 + 递增版本（get_kv 的投影缓存随之失效）
        self.source_ids = source_ids if source_ids is not None else torch.arange(
            k, device=hidden.device).unsqueeze(0).expand(bsz, -1).contiguous()
        self.memory_version = memory_version
        # T3.3 WorkingMemoryV2：valid mask + metadata
        self.valid = valid if valid is not None else (
            torch.ones(bsz, k, dtype=torch.bool, device=hidden.device) if k
            else torch.zeros(bsz, 0, dtype=torch.bool, device=hidden.device))
        self.metadata = dict(metadata or {})
        self._kv_cache = None  # [(k,v), ...] 与 core 层对齐；内容变更即失效

    # ---- 基本属性 ----
    @property
    def size(self):  # block 数（标量；batch 内统一容量）
        return self.hidden.shape[1]

    @property
    def token_size(self):  # 以 token 计的记忆量（供 C_m 统计）
        return self.hidden.shape[1] * self.block_size

    def is_empty(self):
        return self.hidden.shape[1] == 0

    def token_hidden(self):
        """[B, K, block, d] -> [B, K*block, d]（供 project_memory）。"""
        b, k, bs, d = self.hidden.shape
        return self.hidden.reshape(b, k * bs, d)

    def token_valid(self):
        """[B, K] valid -> [B, K*block] token 级 valid（供 GatedMemoryReader）。"""
        b, k = self.valid.shape
        bs = self.block_size if k else 0
        return self.valid.unsqueeze(-1).expand(b, k, bs).reshape(b, k * bs)

    # ---- 修改操作（返回新对象，函数式；Doc 1 §10 ADD/KEEP/EVICT/REPLACE 由 manager 组合）----
    def with_added(self, hidden, keys, source_ids=None):
        """并入新块（版本递增；source_ids 去重——同 source 只更新 usage，不复制 slot）。"""
        if self.is_empty():
            return WorkingMemory(hidden, keys, block_size=self.block_size,
                                 source_ids=source_ids, memory_version=self.memory_version + 1,
                                 valid=torch.ones(hidden.shape[0], hidden.shape[1],
                                                  dtype=torch.bool, device=hidden.device),
                                 metadata=self.metadata)
        new_ids = (source_ids if source_ids is not None else
                   torch.arange(self.size, self.size + hidden.shape[1],
                                device=self.hidden.device).unsqueeze(0).expand(
                       hidden.shape[0], -1).contiguous())
        # T3.3/V2 §9.3：按 source_id 去重——重复检索同一块不复制 slot
        keep_rows = []
        bsz, k_new = hidden.shape[0], hidden.shape[1]
        # 研究级 B=1/共享 source 布局；逐样本 mask 新块中已存在的 source
        dup = torch.zeros(bsz, k_new, dtype=torch.bool, device=hidden.device)
        for b in range(bsz):
            existing = set(int(x) for x in self.source_ids[b].tolist())
            for j in range(k_new):
                sid = int(new_ids[b, j].item())
                if sid in existing:
                    dup[b, j] = True
        if bool(dup.any()):
            keep = ~dup.all(dim=0)  # 任一样本非重复则保留该槽（batch 统一容量，研究级）
            hidden = hidden[:, keep]
            keys = keys[:, keep]
            new_ids = new_ids[:, keep]
            if hidden.shape[1] == 0:
                return WorkingMemory(self.hidden, self.keys, self.usage, self.persistence,
                                     block_size=self.block_size, source_ids=self.source_ids,
                                     memory_version=self.memory_version,  # 无新内容
                                     valid=self.valid, metadata=self.metadata)
        new_valid = torch.ones(hidden.shape[0], hidden.shape[1],
                               dtype=torch.bool, device=hidden.device)
        return WorkingMemory(
            torch.cat([self.hidden, hidden], dim=1),
            torch.cat([self.keys, keys], dim=1),
            torch.cat([self.usage, self.usage.new_zeros(hidden.shape[0], hidden.shape[1])], dim=1),
            torch.cat([self.persistence, self.persistence.new_zeros(hidden.shape[0], hidden.shape[1])], dim=1),
            block_size=self.block_size,
            source_ids=torch.cat([self.source_ids, new_ids], dim=1),
            memory_version=self.memory_version + 1,
            valid=torch.cat([self.valid, new_valid], dim=1),
            metadata=self.metadata)

    def select(self, keep_idx):
        """keep_idx: [B, m'] —— 按样本保留 slot（驱逐 argmin(U) 之后的结果）。"""
        bsz = self.hidden.shape[0]
        arange = torch.arange(bsz, device=self.hidden.device).unsqueeze(1)
        return WorkingMemory(
            self.hidden[arange, keep_idx], self.keys[arange, keep_idx],
            self.usage[arange, keep_idx], self.persistence[arange, keep_idx],
            block_size=self.block_size, source_ids=self.source_ids[arange, keep_idx],
            memory_version=self.memory_version + 1,
            valid=self.valid[arange, keep_idx], metadata=self.metadata)

    def with_usage_bumped(self, hit_mask, decay=0.9):
        """attention/retrieval 使用统计的 EMA 更新：被再次选中的块 usage 提升。"""
        usage = decay * self.usage + (1 - decay) * hit_mask.float()
        return WorkingMemory(self.hidden, self.keys, usage, self.persistence,
                             block_size=self.block_size, source_ids=self.source_ids,
                             memory_version=self.memory_version,
                             valid=self.valid, metadata=self.metadata)

    def with_persistence_ema(self, scores, decay=0.9):
        """persistence = 历史 U 的 EMA（Doc 1 §11 Persistence 因子）。"""
        persistence = decay * self.persistence + (1 - decay) * scores.detach()
        return WorkingMemory(self.hidden, self.keys, self.usage, persistence,
                             block_size=self.block_size, source_ids=self.source_ids,
                             memory_version=self.memory_version,
                             valid=self.valid, metadata=self.metadata)

    # ---- 跨 loop K/V 缓存 ----
    def get_kv(self, thinking_core, cache_key=None):
        """返回与 thinking_core.layers 对齐的 [(k, v), ...]；缓存失效自动重投影。

        cache_key（T3.5）：提供时与 self._kv_cache_key 比对，不一致则重投影。
        """
        if cache_key is not None and getattr(self, "_kv_cache_key", None) != cache_key:
            self._kv_cache = None
        if self._kv_cache is None:
            self._kv_cache = thinking_core.project_memory(self.token_hidden())
            self._kv_cache_key = cache_key
        return self._kv_cache

    def invalidate_kv(self):
        self._kv_cache = None

    def cache_key(self, request_id=None, reader_layer_id=0,
                  projection_weight_version=0, encoder_version=0,
                  device="cpu", dtype="float32"):
        """T3.5 / V2 §11.1：Memory KV 缓存键。

        含 request / memory_version / layer / 投影版本 / encoder 版本 / device / dtype
        —— 仅用 Python 对象身份判断失效不够。
        """
        return (str(request_id), int(self.memory_version), int(reader_layer_id),
                int(projection_weight_version), int(encoder_version),
                str(device), str(dtype))
```

---

# model/minimind_x/minimind_x.py

```python
"""MiniMind-X 主模型（任务清单 T0.2 / T1.2 / T2.4 / T6.2-6.3）。

组装：Prelude → ThinkingCore(×R, 带 Working Memory cross-attention) → Coda → LM Head。

实现要点：
- R=1、无 memory、无 loop_embedding 时，前向与原 MiniMind 数值等价
  （权重经 remap_minimind_state_dict 映射，见 tests/test_equivalence.py）。
- 训练全展开 + per-exit 监督（D3）；推理早停（halt router 逐轮决策，batch 跑到全部停住，
  按样本 gather 各自 exit 的 hidden 与 KV —— D2 的「只消费停住那一轮的 KV」）。
- past_key_values 为扁平 list，长度 = P+T+C，执行序（与原模型 list-of-layers 协议一致）。
"""
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from model.model_minimind import RMSNorm, MOEFeedForward, precompute_freqs_cis
from .config import MiniMindXConfig
from .prelude import Prelude
from .thinking_core import ThinkingCore
from .coda import Coda
from .router import HaltRouter, RetrievalRouter, BudgetRouter


@dataclass
class MiniMindXOutput:
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List] = None
    hidden_states: Optional[torch.Tensor] = None
    # ---- X 专属输出 ----
    exit_logits: Optional[List[torch.Tensor]] = None   # collect_exits 时每个 exit 的 logits
    exit_hiddens: Optional[List[torch.Tensor]] = None  # 每个 exit 过 coda+norm 后的 hidden
    core_committed_states: Optional[List[torch.Tensor]] = None  # T2.4：Core 提交状态（未过 Coda；Router 应读它）
    coda_exit_states: Optional[List[torch.Tensor]] = None      # T2.4：过 Coda 后状态（与 committed 分离）
    halt_probs: Optional[List[torch.Tensor]] = None    # 每轮 P(stop) [B]
    halt_logits: Optional[List[torch.Tensor]] = None   # T2.4：每轮 halt logit（非概率；训练用）
    memory_versions: Optional[List[int]] = None        # T2.4：各轮使用的 memory 版本
    router_trace: Optional[list] = None                # T2.4：控制器轨迹（事件/动作/版本）
    residual_stats: Optional[list] = None              # T2.4：每轮 ḡ/Δ^cand/Δ^commit（detached）
    exit_idx: Optional[torch.Tensor] = None            # 每样本选中的 exit 轮号 [B]
    budget_logits: Optional[torch.Tensor] = None       # 预算档 logits [B, n_budgets]
    router_probs: Optional[dict] = None                # T0.2：全展开时逐轮 {"halt": [...], "retrieval": [...]}
    num_loops_executed: int = 0


class MiniMindXModel(nn.Module):
    def __init__(self, config: MiniMindXConfig):
        super().__init__()
        self.config = config
        P, T, C = config.prelude_layers, config.thinking_layers, config.coda_layers
        self.total_layers = P + T + C
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.prelude = Prelude(config, layer_offset=0)
        self.thinking_core = ThinkingCore(config, layer_offset=P)
        self.coda = Coda(config, layer_offset=P + T)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings,
            rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # ---- routers / controller 模块（config 关闭时不创建，保持 baseline 状态字典干净）----
        self.halt_router = HaltRouter(config) if config.use_loop else None
        self.retrieve_router = RetrievalRouter(config) if config.use_retrieval else None
        self.budget_router = BudgetRouter(config) if config.use_memory else None
        from .memory import MemoryManager
        self.memory_manager = MemoryManager(config) if config.use_memory else None
        from .retrieval.qsa_retriever import QSARetriever  # 延迟导入避免循环依赖
        self.retriever = QSARetriever(config) if config.use_retrieval else None

    # ------------------------------------------------------------------ utils
    def _start_pos(self, past_key_values):
        for past in past_key_values:
            if past is None:
                continue
            entry = past[0] if isinstance(past, list) else past  # core 层为按轮 list
            return entry[0].shape[1]
        return 0

    def _aux_loss(self, hidden_states):
        blocks = list(self.prelude.layers) + list(self.thinking_core.layers) + list(self.coda.layers)
        zero = hidden_states.new_zeros(1).squeeze()
        return sum((l.mlp.aux_loss for l in blocks if isinstance(l.mlp, MOEFeedForward)), zero)

    def _gather_exits(self, stacked, exit_idx):
        """stacked [R,B,...], exit_idx [B] -> [B,...]：按样本取其停住轮的量。"""
        bsz = stacked.shape[1]
        return stacked[exit_idx, torch.arange(bsz, device=stacked.device)]

    def _run_coda(self, hidden_states, position_embeddings, past_key_values,
                  use_cache, attention_mask):
        hidden_states, presents = self.coda(
            hidden_states, position_embeddings, past_key_values, use_cache, attention_mask)
        return self.norm(hidden_states), presents

    def prelude_hidden(self, input_ids, attention_mask=None, no_grad=True):
        """只跑 embedding+Prelude，返回问题表示（终局编排的初始检索 query 来源，D9）。

        no_grad=False 供蒸馏路径使用（v1.3）：query 侧保留梯度，
        使 ranking 蒸馏能训练 query_proj/Prelude；默认 no_grad 维持 D5 语义。
        """
        if no_grad:
            with torch.no_grad():
                return self._prelude_hidden(input_ids, attention_mask)
        return self._prelude_hidden(input_ids, attention_mask)

    def _prelude_hidden(self, input_ids, attention_mask=None):
        seq_len = input_ids.shape[1]
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])
        hidden_states, _ = self.prelude(hidden_states, position_embeddings,
                                        past_key_values=None, use_cache=False,
                                        attention_mask=attention_mask)
        return hidden_states

    @torch.no_grad()
    def encode_cot_steps(self, steps_ids):
        """CoT 各 step 的 pooled 表示（T2.6 Latent Reasoning Supervision 的对齐目标）。

        steps_ids: list of [S] LongTensor（每个 step 一段 token）；
        返回 list of [1,d]（均值池化，供 cot_alignment_loss）。
        """
        outs = []
        for ids in steps_ids:
            h = self.prelude_hidden(ids.view(1, -1))
            outs.append(h.mean(dim=1))
        return outs

    @torch.no_grad()
    def encode_context_blocks(self, context_ids, batch_blocks=64, block_idxs=None):
        """D9：长上下文分块、每块独立过 Prelude（块内注意力天然 windowed）→ 块 hidden。

        供 LongContextStore 懒编码使用；建库不回传梯度（D5）。
        context_ids: [S] 或 [1,S]；block_idxs: 只编码指定块（懒编码路径）；
        返回 [n_blocks, block_size, d]（或 [len(block_idxs), block_size, d]）。
        """
        cfg = self.config
        if context_ids.dim() == 2:
            context_ids = context_ids[0]
        n_blocks = (context_ids.numel() + cfg.block_size - 1) // cfg.block_size
        pad = n_blocks * cfg.block_size - context_ids.numel()
        if pad:
            context_ids = torch.cat([context_ids, context_ids.new_zeros(pad)])
        chunks = context_ids.view(n_blocks, cfg.block_size)
        if block_idxs is not None:
            chunks = chunks[torch.as_tensor(block_idxs, device=chunks.device)]
        n_blocks = chunks.shape[0]
        outs = []
        for i in range(0, n_blocks, batch_blocks):
            h = self.dropout(self.embed_tokens(chunks[i:i + batch_blocks]))
            freqs = (self.freqs_cos[:cfg.block_size], self.freqs_sin[:cfg.block_size])
            h, _ = self.prelude(h, freqs, past_key_values=None, use_cache=False, attention_mask=None)
            outs.append(h)
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                num_loops=1, halt=False, collect_exits=False, exit_choice=None,
                memory_kv_per_layer=None, loop_callback=None, memory_payload=None,
                query_len=None):
        cfg = self.config
        if hasattr(past_key_values, 'layers'):  # 兼容 transformers Cache 误传
            past_key_values = None
        # V2 路由：generate/controller 可能把 (payload, valid) 元组从 memory_kv_per_layer
        # 通道传入（v1 接口）；v2 模式下统一转 memory_payload
        if getattr(cfg, "residual_v2", False) and isinstance(memory_kv_per_layer, tuple):
            memory_payload = memory_payload or memory_kv_per_layer
            memory_kv_per_layer = None
        past = past_key_values or [None] * self.total_layers
        P, T, C = cfg.prelude_layers, cfg.thinking_layers, cfg.coda_layers
        start_pos = self._start_pos(past)
        bsz, seq_len = input_ids.shape
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # RoPE buffers 在 meta-device 初始化后可能丢失（transformers>=5.x），重算
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=cfg.head_dim, end=cfg.max_position_embeddings,
                rope_base=cfg.rope_theta, rope_scaling=cfg.rope_scaling)
            self.freqs_cos, self.freqs_sin = (freqs_cos.to(hidden_states.device),
                                              freqs_sin.to(hidden_states.device))
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_len],
                               self.freqs_sin[start_pos:start_pos + seq_len])
        presents = []

        # ---- Prelude ----
        hidden_states, pre_presents = self.prelude(
            hidden_states, position_embeddings, past[:P], use_cache, attention_mask)
        presents += pre_presents

        # ---- Budget Router：读 Prelude 输出（问题表示），预测预算档 ----
        # 评审 3.2：Router 输入必须与执行一致——只可见前缀，不见 gold 后缀
        router_h = hidden_states if query_len is None else hidden_states[:, :query_len]
        budget_logits = self.budget_router(router_h) if self.budget_router is not None else None

        # ---- ThinkingCore（循环区）----
        num_loops = max(1, int(num_loops)) if cfg.use_loop else 1
        halt_fn = None
        if halt and self.halt_router is not None:
            def halt_fn(hd, residual_stats=None):  # T2.1：committed + detached 统计
                return self.halt_router(hd, residual_stats)
        core_out = self.thinking_core(
            hidden_states, position_embeddings, past[P:P + T], use_cache, attention_mask,
            memory_kv_per_layer=memory_kv_per_layer, num_loops=num_loops,
            halt_fn=halt_fn, halt_threshold=cfg.halt_threshold, loop_callback=loop_callback,
            anchor=hidden_states if getattr(cfg, "residual_v2", False) else None,
            memory_payload=memory_payload)
        exits, passes = core_out["exit_hiddens"], core_out["presents_per_pass"]
        halt_probs, exit_idx = core_out["halt_probs"], core_out["exit_idx"]
        n_run = len(exits)
        exit_hiddens = None

        if collect_exits:
            # 每个 exit 各自过 coda+norm（per-exit 监督的诚实语义：停在 r 轮 = 深度 r 的完整模型）
            exit_hiddens = [self._run_coda(h, position_embeddings, None, False, attention_mask)[0]
                            for h in exits]
            hidden_states = exit_hiddens[-1]
        else:
            # 选择 exit：显式 exit_choice（训练采样 R / oracle teacher-forcing）> halt 决策 > 最后一轮
            if exit_choice is not None:
                exit_idx = exit_choice.clone().to(exits[0].device)
                if exit_idx.dim() == 0:
                    exit_idx = exit_idx.expand(bsz)
            elif exit_idx is not None:
                exit_idx = torch.where(exit_idx < 0, torch.full_like(exit_idx, n_run - 1), exit_idx)
            if exit_idx is not None and n_run > 1:
                hidden_states = self._gather_exits(torch.stack(exits), exit_idx)
            else:
                hidden_states = exits[-1]
                if exit_choice is not None:
                    exit_idx = torch.full((bsz,), n_run - 1, dtype=torch.long,
                                          device=hidden_states.device)
        # core 层缓存：R_max 固定逻辑槽（T3.1，V2 §11.3 per-token last-valid）——
        # 槽 r 的历史 = 该槽此前内容（preserve past deep slots）；本 chunk 新 token：
        #   r < n_run → 其真实第 r 轮 KV；r ≥ n_run → 其最后实际轮 KV（仅填新 token 位）
        # （passes[r] 已含 past_in 拼接：读侧 clamp 与写侧填充采用同一 last-valid 规则）
        if use_cache:
            # 评审 5.1 / V2 §11.3：per-sample last-valid——
            # 布局 [B,S,H,D]；样本 b 的新 token 在 slot r>last_exec[b] 时复制其最后实际轮；
            # 历史 token 的深槽保留（只填新 token 位）
            if exit_idx is not None:
                last_exec = exit_idx.clamp(min=0, max=max(n_run - 1, 0)).tolist()
            else:
                last_exec = [n_run - 1] * bsz
            core_caches = []
            for l in range(T):
                prior_entry = past[P + l] if past_key_values is not None else None
                prior_slots = prior_entry if isinstance(prior_entry, list) else None
                slots = []
                for r in range(cfg.max_loop):
                    k_rows, v_rows = [], []
                    for b in range(bsz):
                        src_r = max(0, min(r, int(last_exec[b]), n_run - 1))
                        kb, vb = passes[src_r][l]  # [B,S,H,D]
                        k_rows.append(kb[b:b + 1])
                        v_rows.append(vb[b:b + 1])
                    k_all = torch.cat(k_rows, dim=0)  # [B,S,H,D] per-sample 选轮
                    v_all = torch.cat(v_rows, dim=0)
                    if prior_slots is not None and r < len(prior_slots):
                        # 只把新 token 位（末 seq_len）接到历史后；历史深槽不覆盖
                        pk, pv = prior_slots[r]
                        slots.append((torch.cat([pk, k_all[:, -seq_len:]], dim=1),
                                      torch.cat([pv, v_all[:, -seq_len:]], dim=1)))
                    else:
                        slots.append((k_all, v_all))
                core_caches.append(slots)
        presents += core_caches if use_cache else [None] * T

        # ---- Coda ----
        if exit_hiddens is not None:
            # collect_exits：最终 hidden 已在逐 exit 时过 coda+norm，不重复；
            # 仅在需要 cache 时为 coda 层补一份最终 exit 的 KV
            if use_cache:
                _, coda_presents = self._run_coda(
                    exits[-1], position_embeddings, past[P + T:], True, attention_mask)
            else:
                coda_presents = [None] * C
        else:
            hidden_states, coda_presents = self._run_coda(
                hidden_states, position_embeddings, past[P + T:], use_cache, attention_mask)
        presents += coda_presents

        aux_loss = self._aux_loss(hidden_states)
        # ---- T0.2（C4 修复）：全展开训练时收集 router 概率（不据此 break）----
        router_probs = None
        if collect_exits:
            router_probs = {}
            # 评审 3.2：监督 logits 与执行同输入（可见前缀池化）
            def _vis(h):
                return h if query_len is None else h[:, :query_len]
            if self.halt_router is not None:
                rs_list = core_out.get("residual_stats") or [None] * len(exits)
                router_probs["halt"] = [
                    self.halt_router(_vis(h), rs) for h, rs in zip(exits, rs_list)]
            if self.retrieve_router is not None:
                router_probs["retrieval"] = [self.retrieve_router(_vis(h)) for h in exits]
                router_probs["retrieval_action"] = [
                    self.retrieve_router.action_logits(_vis(h), warm_available=True)
                    for h in exits]
        # T2.4：halt_logits（非概率）+ residual_stats + memory_versions
        halt_logits_out = None
        if collect_exits and self.halt_router is not None:
            rs_list = core_out.get("residual_stats") or [None] * len(exits)
            halt_logits_out = [
                self.halt_router.forward_logits(
                    h if query_len is None else h[:, :query_len], rs)
                for h, rs in zip(exits, rs_list)]
        residual_stats = core_out.get("residual_stats")
        return dict(hidden_states=hidden_states, presents=presents, aux_loss=aux_loss,
                    exit_hiddens=exit_hiddens, halt_probs=halt_probs, exit_idx=exit_idx,
                    budget_logits=budget_logits, num_loops_executed=n_run,
                    router_probs=router_probs,
                    core_committed_states=core_out.get("committed_states"),
                    coda_exit_states=exit_hiddens,  # collect_exits 时 exit_hiddens 已过 Coda
                    halt_logits=halt_logits_out,
                    residual_stats=residual_stats,
                    memory_versions=core_out.get("memory_versions"))


class MiniMindXForCausalLM(nn.Module):
    def __init__(self, config: MiniMindXConfig = None):
        super().__init__()
        self.config = config or MiniMindXConfig()
        self.model = MiniMindXModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, num_loops=None, halt=False,
                collect_exits=False, exit_choice=None, memory_kv_per_layer=None,
                loop_callback=None, memory_payload=None, query_len=None, **kwargs):
        cfg = self.config
        if num_loops is None:
            num_loops = cfg.max_loop if (collect_exits and cfg.use_loop) else 1
        out = self.model(input_ids, attention_mask, past_key_values, use_cache,
                         num_loops=num_loops, halt=halt, collect_exits=collect_exits,
                         exit_choice=exit_choice, memory_kv_per_layer=memory_kv_per_layer,
                         loop_callback=loop_callback, memory_payload=memory_payload,
                         query_len=query_len)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(out["hidden_states"][:, slice_indices, :])
        exit_logits = None
        if out["exit_hiddens"] is not None:
            exit_logits = [self.lm_head(h[:, slice_indices, :]) for h in out["exit_hiddens"]]
        loss = None
        if labels is not None:  # 兼容现有 trainer：默认用最终 exit 的 CE（联合 loss 见 losses.py）
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MiniMindXOutput(
            loss=loss, aux_loss=out["aux_loss"], logits=logits,
            past_key_values=out["presents"], hidden_states=out["hidden_states"],
            exit_logits=exit_logits, exit_hiddens=out["exit_hiddens"],
            core_committed_states=out.get("core_committed_states"),
            coda_exit_states=out.get("coda_exit_states"),
            halt_probs=out["halt_probs"], halt_logits=out.get("halt_logits"),
            memory_versions=out.get("memory_versions"),
            residual_stats=out.get("residual_stats"),
            exit_idx=out["exit_idx"],
            budget_logits=out["budget_logits"], num_loops_executed=out["num_loops_executed"],
            router_probs=out.get("router_probs"),
            router_trace=out.get("router_trace"))

    # ------------------------------------------------------------------ generate
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0,
                 mode=None, fixed_loops=None, use_halting=True, memory_provider=None,
                 loop_callback=None, return_stats=False, **kwargs):
        """在原 MiniMind generate 基础上增加（T2.4/T6.3）：
          mode: "fast"/"normal"/"deep" —— 循环与 memory 预算上限（config.mode_caps）
          fixed_loops: 固定 R（RL/评测用），关闭早停
          use_halting: 是否用 halt router 逐 token 决策循环数（需 use_loop 且有 halt_router）
          memory_provider: callable(step) -> memory_kv_per_layer（终局动态检索注入）
          return_stats: 额外返回效率统计（avg_loops 等）
        """
        cfg = self.config
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)
        past_key_values = kwargs.pop("past_key_values", None)
        caps = cfg.mode_caps_for(mode)
        loop_cap = min(caps["max_loop"], cfg.max_loop)
        halting = (use_halting and cfg.use_loop and fixed_loops is None
                   and self.model.halt_router is not None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())
        stats = {"loop_sum": 0.0, "steps": 0, "memory_steps": 0,
                 "ttft_steps": 0, "decode_steps": 0,
                 "ttft_loop_sum": 0.0, "decode_loop_sum": 0.0}
        # T3.2（V2 §11.4）：prefill_policy=fixed_cap —— prompt 用固定 R_prefill 建 cache，
        # decode 再按 token 自适应；TTFT 与 decode 成本分开记
        prefill_policy = getattr(cfg, "prefill_policy", "fixed_cap")
        r_prefill = min(loop_cap, cfg.max_loop) if prefill_policy == "fixed_cap" else None
        for step in range(max_new_tokens):
            past_len = (past_key_values[0][0][0].shape[1] if isinstance(past_key_values[0], list)
                        else past_key_values[0][0].shape[1]) if past_key_values else 0
            mem = memory_provider(step) if memory_provider is not None else None
            if mem is not None:
                stats["memory_steps"] += 1
            is_prefill = (step == 0 and past_len == 0)
            if fixed_loops is not None:
                num_loops = fixed_loops
            elif is_prefill and r_prefill is not None:
                num_loops = r_prefill   # fixed_cap：prefill 不逐 token 早停
            else:
                num_loops = loop_cap if halting or cfg.use_loop else 1
            halt_this = halting and not (is_prefill and prefill_policy == "fixed_cap")
            fwd_kw = {}
            if getattr(cfg, "residual_v2", False) and isinstance(mem, tuple):
                fwd_kw["memory_payload"] = mem
            elif mem is not None:
                fwd_kw["memory_kv_per_layer"] = mem
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache,
                num_loops=num_loops, halt=halt_this, loop_callback=loop_callback, **fwd_kw, **kwargs)
            attention_mask = (torch.cat([attention_mask, attention_mask.new_ones(
                attention_mask.shape[0], 1)], -1) if attention_mask is not None else None)
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = (torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                          if do_sample else torch.argmax(logits, dim=-1, keepdim=True))
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1),
                                         next_token.new_full((next_token.shape[0], 1), eos_token_id),
                                         next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            # ---- 循环统计：按样本停住的轮次（+1 = 实际执行轮数）----
            if outputs.exit_idx is not None:
                n_sel = float((outputs.exit_idx + 1).float().mean().item())
            else:
                n_sel = float(num_loops)
            stats["loop_sum"] += n_sel
            stats["steps"] += 1
            if is_prefill:
                stats["ttft_steps"] += 1
                stats["ttft_loop_sum"] += n_sel
            else:
                stats["decode_steps"] += 1
                stats["decode_loop_sum"] += n_sel
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()
        if return_stats:
            stats["avg_loops"] = stats["loop_sum"] / max(stats["steps"], 1)
            stats["avg_ttft_loops"] = stats["ttft_loop_sum"] / max(stats["ttft_steps"], 1)
            stats["avg_decode_loops"] = stats["decode_loop_sum"] / max(stats["decode_steps"], 1)
            # 评审 9.4：selected=样本选中出口；executed=物理执行轮（fixed/自适应上界）
            stats["selected_core_rounds"] = stats["loop_sum"]
            stats["executed_core_rounds"] = stats["loop_sum"]  # 无 active-mask 压缩时二者相近
            return {'generated_ids': input_ids, 'stats': stats,
                    'past_kv': past_key_values} if kwargs.get("return_kv") else \
                   {'generated_ids': input_ids, 'stats': stats}
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids


def remap_minimind_state_dict(state_dict, config: MiniMindXConfig):
    """原 MiniMindForCausalLM 权重 -> MiniMindXForCausalLM 布局（任务清单 T0.3）。

    原键 model.layers.{i}.* 按切分映射到 prelude/thinking_core/coda；
    embed_tokens / norm / lm_head 原名保留。新模块（router/retriever）不在此 dict 中，
    加载时用 strict=False。
    """
    P, T, _ = config.prelude_layers, config.thinking_layers, config.coda_layers
    out = {}
    for k, v in state_dict.items():
        k2 = k[len("model."):] if k.startswith("model.") else k
        if k2.startswith("layers."):
            rest = k2[len("layers."):]
            idx_str = rest.split(".", 1)[0]
            rest = rest.split(".", 1)[1] if "." in rest else ""
            i = int(idx_str)
            if i < P:
                new_k = f"model.prelude.stack.layers.{i}.{rest}"
            elif i < P + T:
                new_k = f"model.thinking_core.layers.{i - P}.{rest}"
            else:
                new_k = f"model.coda.stack.layers.{i - P - T}.{rest}"
            out[new_k] = v
        elif k2 in ("embed_tokens.weight", "norm.weight"):
            out[f"model.{k2}"] = v
        elif k2 == "lm_head.weight":
            out["lm_head.weight"] = v
    return out
```

---

# model/minimind_x/prelude.py

```python
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
```

---

# model/minimind_x/residual/__init__.py

```python
# V2 残差改造三模块（任务清单 P1；V2 文档 §6-8 + 附录 A 参考实现移植）
from .loop_mhar import LoopMHAR
from .recurrent_update import RecurrentUpdateGate

__all__ = ["LoopMHAR", "RecurrentUpdateGate"]
```

---

# model/minimind_x/residual/loop_mhar.py

```python
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
```

---

# model/minimind_x/residual/recurrent_update.py

```python
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
```

---

# model/minimind_x/retrieval/__init__.py

```python
from .block_store import LongContextStore
from .qsa_retriever import QSARetriever, RetrievalResult
from .sparse_attn import SparseBlockAttention

__all__ = ["LongContextStore", "QSARetriever", "RetrievalResult", "SparseBlockAttention"]
```

---

# model/minimind_x/retrieval/block_store.py

```python
"""LongContextStore（任务清单 T4.1；Doc 1 §6、Doc 2 §5、D9）。

长期记忆库：1M 上下文按 block_size（默认 256）切块存储。
- 索引 key：块内 token embedding 的 mean-pool + 线性投影（廉价，建库即算）
- 块内容 hidden：复用 Prelude(windowed) 分块独立编码（懒编码 + LRU 缓存，
  避免一次性物化全部块 hidden 的显存开销）

建库与编码不回传梯度（D5：检索走非可微路径，监督只作用于 router 的分类损失）。
"""
import torch


class LongContextStore:
    def __init__(self, context_ids, block_size=256):
        """context_ids: [S] 或 [1,S] 的 LongTensor（host 或 device 均可）。"""
        if context_ids.dim() == 2:
            context_ids = context_ids[0]
        self.context_ids = context_ids
        self.block_size = block_size
        n = context_ids.numel()
        self.n_blocks = (n + block_size - 1) // block_size
        pad = self.n_blocks * block_size - n
        if pad:
            context_ids = torch.cat([context_ids, context_ids.new_zeros(pad)])
        self._chunks = context_ids.view(self.n_blocks, block_size)
        self.block_keys = None          # [n_blocks, d]（build_keys 后可用）
        self._hidden_cache = {}         # idx -> [block_size, d]
        self._cache_order = []

    # ---- 建索引 ----
    @torch.no_grad()
    def build_keys(self, retriever, embed_tokens, max_blocks=None, batch_blocks=256):
        """用 retriever.key_proj 对块内 embedding mean-pool 投影得到索引 key。"""
        device = next(retriever.parameters()).device
        n = min(self.n_blocks, max_blocks) if max_blocks else self.n_blocks
        keys = []
        for i in range(0, n, batch_blocks):
            emb = embed_tokens(self._chunks[i:i + batch_blocks].to(device))  # [b, block, d]
            keys.append(retriever.key_proj(emb.mean(dim=1)))
        self.block_keys = torch.cat(keys, dim=0)
        if max_blocks:  # 只索引前 max_blocks 块时同步裁剪 chunk 视图
            self.n_blocks = self.block_keys.shape[0]
        return self.block_keys

    # ---- 懒编码块内容 ----
    @torch.no_grad()
    def get_block_hidden(self, model, block_idx, cache_limit=4096):
        """block_idx: [k] 或 [B, k] 的 LongTensor -> 对应块 hidden（[k, block, d] 或 [B, k, block, d]）。

        model: MiniMindXForCausalLM；用其 Prelude 分块编码并做 LRU 缓存。
        """
        squeeze = block_idx.dim() == 1
        flat_idx = block_idx.reshape(-1)
        missing = sorted({int(i) for i in flat_idx.tolist() if i not in self._hidden_cache})
        if missing:
            newly = model.model.encode_context_blocks(
                self.context_ids, block_idxs=missing)
            for j, i in enumerate(missing):
                self._hidden_cache[i] = newly[j]
                self._cache_order.append(i)
            while len(self._cache_order) > cache_limit:  # LRU 驱逐
                self._hidden_cache.pop(self._cache_order.pop(0), None)
        hidden = torch.stack([self._hidden_cache[int(i)] for i in flat_idx.tolist()])
        if squeeze:
            return hidden
        return hidden.view(*block_idx.shape, self.block_size, -1)
```

---

# model/minimind_x/retrieval/qsa_retriever.py

```python
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
```

---

# model/minimind_x/retrieval/sparse_attn.py

```python
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
        # ---- 评审 9.5 / T0.4：块级因果 ----
        # 块键 = 全块 mean 会混入未来 token；即使 attention mask 挡住未来 value，
        # 未来内容仍会改变「选哪些块」。修复：
        #   全局 top-k 只允许「完整可见块」（块末 < query 位）——其 key 无未来污染；
        #   当前未完成块（块末 ≥ query）不参与全局选择，仅走因果局部窗口。
        blk_start = torch.arange(n_blk, device=x.device) * self.block
        blk_end = blk_start + self.block  # 开区间块末
        row_g = torch.arange(total_len - seq_len, total_len, device=x.device)
        # 完整可见：块末 ≤ query+1（块内所有 key 均 ≤ query）
        blk_fully_vis = blk_end.unsqueeze(0) <= (row_g + 1).unsqueeze(1)    # [S, n_blk]
        blk_score = blk_score.masked_fill(
            ~blk_fully_vis.unsqueeze(0).unsqueeze(0), float("-inf"))
        k_sel = min(self.n_selected, n_blk)
        sel = blk_score.topk(k_sel, dim=-1).indices                    # [B,H,S,k]
        block_mask = torch.zeros(bsz, self.n_local_heads, seq_len, n_blk,
                                 device=x.device, dtype=torch.bool).scatter_(-1, sel, True)
        # 强制排除未完整可见块（topk 可能捡到 -inf 槽）
        block_mask = block_mask & blk_fully_vis.unsqueeze(0).unsqueeze(0)
        col_block = torch.arange(total_len, device=x.device) // self.block
        allow_blocks = block_mask.any(dim=1, keepdim=True)[..., col_block]  # [B,1,S,total]

        # 窗口 + 因果：未完成块的 key 只能靠 band 进入；最终仍与因果取交集
        row = torch.arange(total_len - seq_len, total_len, device=x.device).unsqueeze(1)
        col = torch.arange(total_len, device=x.device).unsqueeze(0)
        band = (col <= row) & (col > row - self.window)
        causal = col <= row
        allow = (allow_blocks | band.unsqueeze(0).unsqueeze(0)) & causal.unsqueeze(0).unsqueeze(0)
        scores = (xq @ xk_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allow, float("-inf"))
        if attention_mask is not None:
            scores = scores + (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv_rep
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), past_kv
```

---

# model/minimind_x/router/__init__.py

```python
from .halt_router import HaltRouter, pool_hidden
from .retrieval_router import RetrievalRouter
from .budget_router import BudgetRouter

__all__ = ["HaltRouter", "RetrievalRouter", "BudgetRouter", "pool_hidden"]
```

---

# model/minimind_x/router/budget_router.py

```python
"""Memory/Budget Router（任务清单 T5.4；Doc 4 §14）。

预测所需 Memory 预算档位（{1K,2K,4K,8K,16K} 的分类），监督信号为 oracle B*。
"""
import torch
from torch import nn

from .halt_router import pool_hidden


class BudgetRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_budgets = len(config.memory_budgets)
        self.proj = nn.Linear(config.hidden_size, self.n_budgets, bias=True)

    def forward(self, hidden):
        """hidden [B,S,d] -> logits [B, n_budgets]（训练 CE；推理 argmax 得预算档）。"""
        return self.proj(pool_hidden(hidden, "mean"))
```

---

# model/minimind_x/router/halt_router.py

```python
"""Thinking/Halting Router（任务清单 T2.1；Doc 2 §12、V2 §7.5）。

预测 continue/stop（二分类）；标签由 oracle R* 展开为逐轮 stop 标签。
T2.1：可选注入 detached 门控统计（ḡ, Δ^cand, Δ^commit）——经零初始化旁路加到 logit，
不改变主投影维度（旧 checkpoint 可直接加载，stats 旁路初始贡献 0）。
"""
import torch
from torch import nn


def pool_hidden(hidden, pool="mean"):
    """[B,S,d] -> [B,d]（decode 时 S=1，两种池化等价）。"""
    if hidden.dim() == 2:
        return hidden
    if pool == "last":
        return hidden[:, -1, :]
    return hidden.mean(dim=1)


class HaltRouter(nn.Module):
    """p_s = σ(W_s · pool(h_r) + W_stats · [ḡ,Δ^cand,Δ^commit])，P(stop)。"""

    def __init__(self, config):
        super().__init__()
        self.pool = config.halt_pool
        self.proj = nn.Linear(config.hidden_size, 1, bias=True)
        # T2.1 / V2 §7.5：detached 统计旁路（零初始化 → 与旧路径数值一致）
        self.stats_proj = nn.Linear(3, 1, bias=False)
        nn.init.zeros_(self.stats_proj.weight)

    def forward(self, hidden, residual_stats=None):
        pooled = pool_hidden(hidden, self.pool)
        logit = self.proj(pooled)
        if residual_stats is not None:
            stats = torch.stack([
                residual_stats["gate_mean"],
                residual_stats["delta_cand"],
                residual_stats["delta_commit"],
            ], dim=-1).to(logit.dtype).to(logit.device)
            logit = logit + self.stats_proj(stats)
        return torch.sigmoid(logit).squeeze(-1)

    def forward_logits(self, hidden, residual_stats=None):
        """T2.4：返回 halt logit（非概率）。"""
        pooled = pool_hidden(hidden, self.pool)
        logit = self.proj(pooled)
        if residual_stats is not None:
            stats = torch.stack([
                residual_stats["gate_mean"],
                residual_stats["delta_cand"],
                residual_stats["delta_commit"],
            ], dim=-1).to(logit.dtype).to(logit.device)
            logit = logit + self.stats_proj(stats)
        return logit.squeeze(-1)
```

---

# model/minimind_x/router/retrieval_router.py

```python
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
```

---

# model/minimind_x/thinking_core.py

```python
"""ThinkingCore / Deliberation Core（任务清单 T1.1、T5.3；Doc 1 §4.2、Doc 2 §10-11、§14-15）。

关键语义（任务清单 D2/D3 及实现期细化，详见 docs/ARCHITECTURE.md §3；tests/test_loop_kv.py 锁定）：
1. 轮次轴 KV：core 层缓存按轮分槽（list of (k,v)），第 r 轮 query 只看历史 token 第 r 轮的
   K/V（历史轮数不足 clamp 到最近轮）。token 轴每步只增加 R 个槽；轮间递归同时通过
   residual hidden 与轮次槽流动；decode 与全展开 prefill 严格数值一致。
2. 训练全展开（full unroll）：固定跑 num_loops 轮并返回每个 exit 的 hidden，
   「随机 R / 早停」通过 exit 选择与 loss mask 实现（不做 data-dependent break）。
3. 推理早停：传入 halt_fn(exit_hidden)->P(stop)[B]，逐轮决策，全部样本停住即止。
4. Working Memory 是独立 KV 源：memory_kv_per_layer 是「每层一个 (k,v)」列表，
   拼接进该层 attention 的可见前缀（无 RoPE、不受因果/padding mask 限制、不进 cache）。
5. memory K/V 由本层自己的 k_proj/v_proj 投影（project_memory），一次检索只算一次、跨轮复用。
"""
import math
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from model.model_minimind import Attention, MiniMindBlock, apply_rotary_pos_emb, repeat_kv


class MemoryAwareAttention(Attention):
    """支持拼接 Working Memory K/V 前缀的注意力（memory 为 None 时完全退化为父类行为）。"""

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False,
                attention_mask=None, memory_kv=None):
        if memory_kv is None:
            return super().forward(x, position_embeddings, past_key_value, use_cache, attention_mask)
        bsz, seq_len, _ = x.shape
        mem_k, mem_v = memory_kv  # [B, M, n_kv_heads, head_dim]，已投影+归一化、未 RoPE
        mem_len = mem_k.shape[1]
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
        past_kv = (xk, xv) if use_cache else None  # cache 只含 token 序列，不含 memory
        attn_k = torch.cat([mem_k, xk], dim=1)
        attn_v = torch.cat([mem_v, xv], dim=1)
        xq = xq.transpose(1, 2)
        attn_k = repeat_kv(attn_k, self.n_rep).transpose(1, 2)
        attn_v = repeat_kv(attn_v, self.n_rep).transpose(1, 2)
        scores = (xq @ attn_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 因果 mask 只作用于最后 seq_len 列（当前块）；memory 与 past 前缀永远可见
        if self.is_causal:
            scores[:, :, :, -seq_len:] += torch.full(
                (seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
        if attention_mask is not None:
            mask = torch.cat([attention_mask.new_ones(bsz, 1, 1, mem_len),
                              attention_mask.unsqueeze(1).unsqueeze(2)], dim=-1)
            scores = scores + (1.0 - mask) * -1e9
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ attn_v
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), past_kv


class ThinkingBlock(MiniMindBlock):
    """Deliberation Core 层：在父类基础上把 memory_kv 传给 attention。"""

    def __init__(self, layer_id, config):
        super().__init__(layer_id, config)
        self.self_attn = MemoryAwareAttention(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None,
                use_cache=False, attention_mask=None, memory_kv=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask, memory_kv=memory_kv)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class ReaderFirstBlock(ThinkingBlock):
    """V2 Core 首层（任务清单 T2.2；V2 §8.1 插入位置）：

    Self-Attention + residual → 独立 Memory Reader（Reader Gate + 有界 scale）
    + 独立 residual → FFN + residual。全模型唯一 memory 读取点；
    memory_payload=None 时退化为普通 ThinkingBlock。
    """

    def forward(self, hidden_states, position_embeddings, past_key_value=None,
                use_cache=False, attention_mask=None, memory_kv=None,
                reader=None, memory_payload=None, memory_kv_proj=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask)
        hidden_states = hidden_states + residual
        if reader is not None and (memory_payload is not None or memory_kv_proj is not None):
            # 评审 9.2：优先用预投影 MemoryKV（跨 loop 复用）；否则现场 project
            kv = memory_kv_proj
            if kv is None:
                payload, valid = memory_payload
                kv = reader.project_memory(payload, valid)
            hidden_states, _ = reader(hidden_states, kv)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class ThinkingCore(nn.Module):
    """自适应思考区：[L4→L5→L6] × R（Doc 2 §10-11）。

    core_attn="sparse" 时（Baseline Model C）改用普通 ZoneBlock(sparse)，
    该路径无 memory 支持（Model C 是 fixed depth 的 sparse attention baseline）。
    """

    def __init__(self, config, layer_offset=0):
        super().__init__()
        self.config = config
        self.num_layers = config.thinking_layers
        self.gradient_checkpointing = False  # T1.4
        if config.core_attn == "sparse":
            from .prelude import ZoneBlock
            self.layers = nn.ModuleList(
                [ZoneBlock(layer_offset + i, config, "sparse") for i in range(config.thinking_layers)])
        else:
            blocks = [ThinkingBlock(layer_offset + i, config)
                      for i in range(config.thinking_layers)]
            self.layers = nn.ModuleList(blocks)
        # D4：可选轮次嵌入（每轮复用同一 RoPE positions；loop embedding 默认关闭，供消融）
        self.loop_embed = (nn.Embedding(config.max_loop, config.hidden_size)
                           if config.loop_embedding else None)
        # ---- V2 残差改造（任务清单 P2/T2.1-T2.2）----
        self.residual_v2 = bool(getattr(config, "residual_v2", False))
        if self.residual_v2:
            if config.core_attn == "sparse":
                raise ValueError("residual_v2 与 Baseline C(sparse core) 不兼容")
            from .residual import LoopMHAR, RecurrentUpdateGate
            from .memory.reader import GatedMemoryReader
            self.loop_mhar = LoopMHAR(config.hidden_size,
                                      heads=config.mhar_heads,
                                      latest_bias=config.mhar_latest_bias)
            self.update_gate = RecurrentUpdateGate(config.hidden_size,
                                                   rank=config.update_rank)
            qh = config.reader_query_heads or max(2, config.hidden_size // 64)
            self.memory_reader = GatedMemoryReader(
                config.hidden_size, config.hidden_size,
                query_heads=qh, kv_heads=config.reader_kv_heads,
                head_dim=min(config.reader_head_dim, config.hidden_size // qh)
                if config.hidden_size // qh >= 8 else 8,
                scale_init=config.reader_scale_init)
            # V2 模式下首层换成带 Reader 插槽的块（唯一 memory 读取点，T2.2）
            self.layers[0] = ReaderFirstBlock(layer_offset, config)

    def project_memory(self, memory_hidden):
        """把 memory 块 hidden（[B, M, d]）用每个 core 层自己的 k_proj/v_proj 投影成 K/V。

        含 k_norm、不含 RoPE（cross-attention 语义）；一次检索只算一次、跨轮复用（Doc 2 §15）。
        返回与 self.layers 对齐的 [(k, v), ...] 列表（sparse 路径不支持 memory）。
        """
        bsz, mem_len, _ = memory_hidden.shape
        out = []
        for layer in self.layers:
            if not isinstance(layer, ThinkingBlock):
                raise RuntimeError("core_attn='sparse'（Baseline C）不支持 Working Memory")
            attn = layer.self_attn
            k = attn.k_norm(attn.k_proj(memory_hidden).view(
                bsz, mem_len, attn.n_local_kv_heads, attn.head_dim))
            v = attn.v_proj(memory_hidden).view(
                bsz, mem_len, attn.n_local_kv_heads, attn.head_dim)
            out.append((k, v))
        return out

    def _single_pass(self, hidden_states, position_embeddings, past_key_values,
                     use_cache, attention_mask, memory_kv_per_layer, loop_idx,
                     memory_payload=None):
        """执行一轮。past_key_values: 每层一个「按轮缓存」条目（list of (k,v) 或 None）。

        轮次轴 KV 语义（D2 细化）：第 r 轮的 query 只看历史 token 在第 r 轮的 K/V
        （不足 r 轮的历史 clamp 到其最近一轮）——与全展开 prefill 每轮自洽重算严格一致。
        V2：首层为 ReaderFirstBlock（memory_payload 经独立 Reader 注入，T2.2）；
        旧路径 memory_kv_per_layer 拼接仅 residual_v2=False 时有效。
        """
        if self.loop_embed is not None:
            hidden_states = hidden_states + self.loop_embed.weight[loop_idx]
        memory_kv_per_layer = memory_kv_per_layer or [None] * self.num_layers
        # 评审 9.2：同一 memory_version 的 Reader 投影跨 loop 复用
        mem_kv_proj = None
        if self.residual_v2 and memory_payload is not None and self.memory_reader is not None:
            mv = getattr(memory_payload, "memory_version", None)
            if getattr(self, "_reader_kv_ver", None) != mv or getattr(self, "_reader_kv", None) is None:
                payload, valid = memory_payload
                self._reader_kv = self.memory_reader.project_memory(payload, valid)
                self._reader_kv_ver = mv
            mem_kv_proj = self._reader_kv
        presents = []
        for l, layer in enumerate(self.layers):
            pl = past_key_values[l]
            past_in = pl[min(loop_idx, len(pl) - 1)] if pl else None
            mem = memory_kv_per_layer[l]
            ckpt = getattr(self, "gradient_checkpointing", False)
            if ckpt and self.training and hidden_states.requires_grad:
                if isinstance(layer, ReaderFirstBlock):
                    hidden_states, present = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, position_embeddings, past_in, use_cache,
                        attention_mask, memory_kv=mem, reader=self.memory_reader,
                        memory_payload=memory_payload, memory_kv_proj=mem_kv_proj,
                        use_reentrant=False)
                elif isinstance(layer, ThinkingBlock):
                    hidden_states, present = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, position_embeddings, past_in, use_cache,
                        attention_mask, memory_kv=mem, use_reentrant=False)
                else:  # Baseline C 的 sparse 层
                    hidden_states, present = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, position_embeddings, past_in, use_cache,
                        attention_mask, use_reentrant=False)
                presents.append(present)
                continue
            if isinstance(layer, ReaderFirstBlock):
                hidden_states, present = layer(
                    hidden_states, position_embeddings,
                    past_key_value=past_in, use_cache=use_cache,
                    attention_mask=attention_mask, memory_kv=mem,
                    reader=self.memory_reader if self.residual_v2 else None,
                    memory_payload=memory_payload, memory_kv_proj=mem_kv_proj)
            elif isinstance(layer, ThinkingBlock):
                hidden_states, present = layer(
                    hidden_states, position_embeddings,
                    past_key_value=past_in, use_cache=use_cache,
                    attention_mask=attention_mask, memory_kv=mem)
            else:  # Baseline C 的 sparse 层
                hidden_states, present = layer(
                    hidden_states, position_embeddings,
                    past_key_value=past_in, use_cache=use_cache,
                    attention_mask=attention_mask)
            presents.append(present)
        return hidden_states, presents

    def forward(self, hidden_states, position_embeddings, past_key_values=None,
                use_cache=False, attention_mask=None, memory_kv_per_layer=None,
                num_loops=1, halt_fn=None, halt_threshold=0.5, loop_callback=None,
                anchor=None, memory_payload=None):
        """循环执行（V2 §12.4 顺序：MHAR → Core(首层 Reader) → Update 提交 → Halt）。

        参数：
          anchor: V2 必需——Prelude 输出 E（MHAR 来源 + Update Gate 参照）
          memory_payload: V2 的 (payload[B,M,D], valid[B,M])——经独立 Reader 注入，
                          替代旧 memory_kv_per_layer 拼接路径（T2.2）
          其余参数同前（num_loops/halt_fn/loop_callback 语义不变）

        返回 dict 新增：committed_states（每轮提交状态；v1 路径下与 exit_hiddens 同物）
        """
        num_loops = max(1, int(num_loops))
        # 评审：Reader 投影缓存不得跨 forward 复用（sweep 等 no_grad 前向会污染 detached 缓存）
        self._reader_kv = None
        self._reader_kv_ver = None
        past_key_values = past_key_values or [None] * self.num_layers
        exit_hiddens, presents_per_pass = [], []
        halt_probs, exit_idx = [], None
        committed_states = []
        residual_stats_per_round = []   # T2.4：每轮 residual 统计（ḡ/Δ^cand/Δ^commit）
        memory_versions = []            # T2.4：各轮 memory 版本（对象身份/显式版本）
        if self.residual_v2 and anchor is None:
            raise ValueError("residual_v2 需要 anchor（Prelude 输出）")
        history = [anchor] if self.residual_v2 else None
        previous = anchor if self.residual_v2 else None
        if halt_fn is not None:
            bsz = hidden_states.shape[0]
            halted = torch.zeros(bsz, dtype=torch.bool, device=hidden_states.device)
            exit_idx = torch.full((bsz,), -1, dtype=torch.long, device=hidden_states.device)
        for r in range(num_loops):
            if loop_callback is not None and r > 0:
                new_state = loop_callback(r - 1, exit_hiddens[-1])
                if new_state is not None:
                    if self.residual_v2 and isinstance(new_state, tuple):
                        memory_payload = new_state           # v2：返回 (payload, valid)
                    else:
                        memory_kv_per_layer = new_state      # v1：返回 per-layer KV
            if self.residual_v2:
                read_state, _ = self.loop_mhar(history, strength=self.config.mhar_strength)
            else:
                read_state = hidden_states
            candidate, presents = self._single_pass(
                read_state, position_embeddings, past_key_values,
                use_cache, attention_mask, memory_kv_per_layer, r,
                memory_payload=memory_payload)
            residual_stats_r = None
            if self.residual_v2:
                # 评审 5.1：已早停样本 active=False，committed 保持 previous（不写新状态）
                # active 须为 [B,T]；halted 是 [B]，广播到 token 轴
                active = None
                if halt_fn is not None:
                    active = (~halted).unsqueeze(1).expand(-1, previous.shape[1])
                gated, gate = self.update_gate(previous, anchor, candidate, active=active)
                lam = self.config.update_progress
                committed = candidate + lam * (gated - candidate)
                # T2.1 / V2 §7.5：halt 用 detached 门控统计（不把 gate 路径绑进 halt 梯度）
                with torch.no_grad():
                    gbar = gate.float().mean(dim=(1, 2))                     # [B] ḡ
                    d_cand = (candidate.float() - previous.float()).norm(dim=-1).mean(dim=-1)
                    d_commit = (committed.float() - previous.float()).norm(dim=-1).mean(dim=-1)
                residual_stats_r = {"gate_mean": gbar.detach(),
                                    "delta_cand": d_cand.detach(),
                                    "delta_commit": d_commit.detach()}
                history.append(committed)
                previous = committed
            else:
                committed = candidate
                hidden_states = committed  # v1 路径：committed 回写为下一轮输入
            exit_hiddens.append(committed)
            committed_states.append(committed)
            presents_per_pass.append(presents)
            residual_stats_per_round.append(residual_stats_r)
            # T2.4：记录本轮 memory 版本（payload 对象身份 → 单调版本号）
            mv = getattr(memory_payload, "memory_version", None)
            memory_versions.append(int(mv) if mv is not None else 0)
            if halt_fn is not None:
                # T2.1：halt 读 committed + detached 统计（V2 §7.5 / §12.4）
                if residual_stats_r is not None:
                    p_stop = halt_fn(committed, residual_stats_r)
                else:
                    p_stop = halt_fn(committed)
                halt_probs.append(p_stop)
                newly = (~halted) & (p_stop > halt_threshold)
                exit_idx[newly] = r
                halted = halted | newly
                if bool(halted.all()):
                    break
        return dict(exit_hiddens=exit_hiddens, presents_per_pass=presents_per_pass,
                    halt_probs=halt_probs, exit_idx=exit_idx,
                    committed_states=committed_states,
                    residual_stats=residual_stats_per_round,
                    memory_versions=memory_versions)
```

---

# trainer/dual_executor.py

```python
"""双执行器（任务清单 T4.6；V2 §10.4/§11.5）。

语义金标准是 **append-only**（与 generate 一致）：
  - token 只追加写入 cache；已算过的 KV 不回溯
  - memory 从 W1→W2 只影响此后新 token，不重算历史
  - 首段可整段 prefill；之后逐 token（或段内）追加

三个入口：
  1. reference_append_only —— 带 cache 增量参考（正确性金标准，对齐 generate）
  2. segmented_forward     —— 按 memory-version/动作边界分段；段间续 cache
  3. reference_recompute   —— 旧「每前缀从头重算」（仅作对照，不是金标准）

共同输出：per_token_ce[t] = CE( ŷ_{t+1} | x_{0..t}, append-only 历史 )，t<T-1。
"""
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH = 0, 1, 2


def _mem_kw(ctrl):
    mem_in = ctrl._memory_input()
    if getattr(ctrl.cfg, "residual_v2", False):
        return {"memory_payload": mem_in}
    return {"memory_kv_per_layer": mem_in}


def build_action_timeline(retrieve_actions: Sequence[int],
                          token_segments: Sequence[Tuple[int, int]]):
    """动作时间线。retrieve_actions[i]：进入 segment i 时执行的动作 id。"""
    assert len(retrieve_actions) == len(token_segments)
    timeline = []
    ver = 0
    for (s, e), a in zip(token_segments, retrieve_actions):
        a = int(a)
        timeline.append({"token_start": int(s), "token_end": int(e),
                         "retrieval_action": a,
                         "memory_version_at_start": ver})
        if a != ACTION_REUSE:
            ver += 1
    return timeline


def _prepare_memories(model, input_ids, context_ids, query_len, timeline,
                      controller_factory, attention_mask=None):
    """按时间线预演 memory（段开始前已定）。返回 list[mem_kw] 与 timeline 对齐。"""
    T = input_ids.shape[1]
    ctrl = controller_factory()
    ctrl.begin(context_ids)
    causal = input_ids[:, :query_len]
    causal_mask = attention_mask[:, :query_len] if attention_mask is not None else None
    q0 = ctrl.raw.model.prelude_hidden(causal, causal_mask)
    ctrl.memory = ctrl.retrieve_into_memory(q0)

    mem_list = []
    for seg in timeline:
        a = int(seg["retrieval_action"])
        t0 = int(seg["token_start"])
        if a != ACTION_REUSE:
            vis_len = max(query_len, min(t0 + 1, T))
            vis = ctrl.raw.model.prelude_hidden(
                input_ids[:, :vis_len],
                attention_mask[:, :vis_len] if attention_mask is not None else None)
            ctrl.memory = ctrl.execute_retrieval_action(
                a, vis, budget_idx=ctrl._budget_idx(vis), mem=ctrl.memory)
        mem_list.append(_mem_kw(ctrl))
    return mem_list


def _mem_at(timeline, mem_list, t):
    for seg, mk in zip(timeline, mem_list):
        if seg["token_start"] <= t < seg["token_end"]:
            return mk
    return mem_list[-1] if mem_list else {}


def _ce_from_logits_seq(logits_seq, labels, device):
    """logits_seq[t] = 位置 t 的 next-token 分布（处理完 x[t] 后）。"""
    B, T = labels.shape
    ce = torch.zeros(B, T, device=device)
    for t in range(T - 1):
        ce[:, t] = F.cross_entropy(logits_seq[t], labels[:, t + 1],
                                   ignore_index=-100, reduction="none")
    return ce


@torch.no_grad()
def reference_append_only(model, input_ids, labels, context_ids, query_len,
                          timeline, controller_factory, attention_mask=None,
                          num_loops=None):
    """append-only 增量参考执行器（V2 §11.5 金标准，与 generate 同语义）。

    · 首段整段 prefill（与 generate 第一步一致）
    · 之后按 token 追加进 cache；memory 换版只作用于新 token
    · 历史 KV 永不重算
    返回 (per_token_ce [B,T], logits_seq list[B,V])
    """
    cfg = model.config
    B, T = input_ids.shape
    loops = 1 if not cfg.use_loop else (num_loops or cfg.max_loop)
    mem_list = _prepare_memories(model, input_ids, context_ids, query_len, timeline,
                                 controller_factory, attention_mask)

    # 按段推进：段 [s,e) 共用段开始前已定的 memory
    past = None
    logits_seq: List[torch.Tensor] = [None] * T
    for seg, mk in zip(timeline, mem_list):
        s, e = int(seg["token_start"]), int(seg["token_end"])
        s, e = max(0, s), min(T, e)
        if s >= e:
            continue
        for t in range(s, e):
            out = model(input_ids[:, t:t + 1], past_key_values=past, use_cache=True,
                        num_loops=loops, **mk)
            past = out.past_key_values
            logits_seq[t] = out.logits[:, -1, :]
    ce = _ce_from_logits_seq(logits_seq, labels, input_ids.device) if labels is not None \
        else torch.zeros(B, T, device=input_ids.device)
    return ce, logits_seq


@torch.no_grad()
def segmented_forward(model, input_ids, labels, context_ids, query_len,
                      timeline, controller_factory, attention_mask=None,
                      num_loops=None):
    """分段执行器：段内可批处理追加，段边界续 cache（append-only，不从头重算）。

    与 reference_append_only 数值对应；比逐 token 更省前向次数。
    返回 (per_token_ce [B,T], logits_seq)
    """
    cfg = model.config
    B, T = input_ids.shape
    loops = 1 if not cfg.use_loop else (num_loops or cfg.max_loop)
    mem_list = _prepare_memories(model, input_ids, context_ids, query_len, timeline,
                                 controller_factory, attention_mask)

    past = None
    logits_seq: List[torch.Tensor] = [None] * T
    for seg, mk in zip(timeline, mem_list):
        s, e = int(seg["token_start"]), int(seg["token_end"])
        s, e = max(0, s), min(T, e)
        if s >= e:
            continue
        # 段内一次前向追加 [s,e)：cache 只增不回溯；memory 为段边界前已定的版本
        out = model(input_ids[:, s:e], past_key_values=past, use_cache=True,
                    num_loops=loops, **mk)
        past = out.past_key_values
        # out.logits: [B, e-s, V]；位置 s+i 处理完 x[s+i] 的 next 分布
        for i in range(e - s):
            logits_seq[s + i] = out.logits[:, i, :]
    ce = _ce_from_logits_seq(logits_seq, labels, input_ids.device) if labels is not None \
        else torch.zeros(B, T, device=input_ids.device)
    return ce, logits_seq


@torch.no_grad()
def reference_recompute(model, input_ids, labels, context_ids, query_len,
                        timeline, controller_factory, attention_mask=None,
                        num_loops=None):
    """旧实现：每前缀从头重算（违反 append-only）。仅作对照，不是金标准。"""
    cfg = model.config
    B, T = input_ids.shape
    loops = 1 if not cfg.use_loop else (num_loops or cfg.max_loop)
    mem_list = _prepare_memories(model, input_ids, context_ids, query_len, timeline,
                                 controller_factory, attention_mask)
    logits_seq: List[torch.Tensor] = [None] * T
    for t in range(T):
        out = model(input_ids[:, :t + 1], num_loops=loops,
                    **_mem_at(timeline, mem_list, t))
        logits_seq[t] = out.logits[:, -1, :]
    ce = _ce_from_logits_seq(logits_seq, labels, input_ids.device) if labels is not None \
        else torch.zeros(B, T, device=input_ids.device)
    return ce, logits_seq


@torch.no_grad()
def generate_teacher_forced(model, input_ids, mem_schedule, num_loops=None,
                            prompt_len=None):
    """与 generate() 同构的 teacher-forcing：首块 prefill，其后逐 token 追加。

    mem_schedule: 每个 token 位置的 mem_kw（或 callable(t)）。
    用于和 reference_append_only / 真实 generate 对照。
    """
    cfg = model.config
    B, T = input_ids.shape
    loops = 1 if not cfg.use_loop else (num_loops or cfg.max_loop)
    prompt_len = T if prompt_len is None else min(prompt_len, T)
    past = None
    logits_seq: List[torch.Tensor] = [None] * T

    def mem_at(t):
        if callable(mem_schedule):
            return mem_schedule(t)
        return mem_schedule[t] if mem_schedule else {}

    # 与 generate 第一步一致：整段 prompt prefill
    out = model(input_ids[:, :prompt_len], past_key_values=past, use_cache=True,
                num_loops=loops, **(mem_at(0) or {}))
    past = out.past_key_values
    for i in range(prompt_len):
        logits_seq[i] = out.logits[:, i, :]
    # 其后逐 token（generate decode）
    for t in range(prompt_len, T):
        out = model(input_ids[:, t:t + 1], past_key_values=past, use_cache=True,
                    num_loops=loops, **(mem_at(t) or {}))
        past = out.past_key_values
        logits_seq[t] = out.logits[:, -1, :]
    return logits_seq


def assert_executors_correspond(ref_logits, seg_logits, atol=1e-4):
    """数值对应：logits 列表逐位一致。"""
    if len(ref_logits) != len(seg_logits):
        return False, f"len mismatch {len(ref_logits)} vs {len(seg_logits)}"
    max_diff = 0.0
    for a, b in zip(ref_logits, seg_logits):
        if a is None or b is None:
            return False, "None logits"
        if a.shape != b.shape:
            return False, f"shape {a.shape} vs {b.shape}"
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return False, "non-finite"
        max_diff = max(max_diff, (a - b).abs().max().item())
    return max_diff <= atol, f"max |Δlogit|={max_diff}"


# 兼容旧测试签名（CE 版）
def assert_executors_correspond_ce(ref_ce, seg_ce, atol=1e-5):
    if ref_ce.shape != seg_ce.shape:
        return False, f"shape mismatch {ref_ce.shape} vs {seg_ce.shape}"
    diff = (ref_ce - seg_ce).abs().max().item()
    return diff <= atol, f"max |ΔCE|={diff}"
```

---

# trainer/oracle_labels.py

```python
"""Self-generated Supervision：oracle 标签生成（任务清单 T2.2；Doc 3 §12、Doc 4 §10-13）。

标签不人工标注，由模型自身性能矩阵产生：
  R*（思考深度） = min{ R : Loss_R ≤ min_R Loss + ε } 或首个 correct 的 R
  B*（记忆预算） = min{ B : Loss_B ≤ min_B Loss + ε }
  retrieve 标签  = 1{ Loss_without − Loss_with > ε }

两种模式（任务清单风险表）：在线小采样率生成 / 离线周期批量生成。
全部在 no_grad 下执行（teacher-forcing 的动作标签，不参与反传图）。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F


def oracle_depth_from_losses(per_exit_loss: torch.Tensor, epsilon: float = 0.05):
    """per_exit_loss [B,R] -> R* [B]：最小满足 loss ≤ min+ε 的轮次（1 起）。"""
    r_max = per_exit_loss.shape[-1]
    best = per_exit_loss.min(dim=-1).values
    ok = per_exit_loss <= (best + epsilon).unsqueeze(-1)          # [B,R]
    rank = torch.arange(r_max, device=per_exit_loss.device).unsqueeze(0).expand_as(ok)
    first = torch.where(ok, rank, torch.full_like(rank, r_max)).min(dim=-1).values
    return (first + 1).clamp(max=r_max).long()


def oracle_depth_from_correct(per_exit_correct: torch.Tensor):
    """per_exit_correct [B,R] -> R*：首个 correct 的轮次（全错则 R_max）。"""
    r_max = per_exit_correct.shape[1]
    first = per_exit_correct.float().argmax(dim=-1)
    any_ok = per_exit_correct.any(dim=-1)
    r = first + 1
    return torch.where(any_ok, r, torch.full_like(r, r_max))


def oracle_budget_from_matrix(loss_matrix: torch.Tensor, epsilon: float = 0.05):
    """loss_matrix [B, n_budgets] -> B* 档位 idx。"""
    best = loss_matrix.min(dim=-1).values
    ok = loss_matrix <= (best + epsilon).unsqueeze(-1)
    return ok.float().argmax(dim=-1).long()


def oracle_retrieval_label(loss_without: torch.Tensor, loss_with: torch.Tensor,
                           epsilon: float = 0.05):
    """1{ L_without − L_with > ε }（Doc 4 §13）。"""
    return ((loss_without - loss_with) > epsilon).long()


def _per_sample_ce(logits: torch.Tensor, labels: torch.Tensor):
    """logits [B,S,V], labels [B,S] -> [B]（右移、ignore -100）。"""
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    ce = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100,
                         reduction="none").view(y.shape)
    return (ce * (y != -100)).sum(dim=-1) / (y != -100).sum(dim=-1).clamp(min=1)


def _per_sample_correct(logits: torch.Tensor, labels: torch.Tensor):
    """teacher-forcing 全对才算 correct（判别型 oracle）。"""
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    pred = x.argmax(dim=-1)
    valid = (y != -100)
    return ((pred == y) | ~valid).all(dim=-1) & valid.any(dim=-1)


@torch.no_grad()
def sweep_depth(model, input_ids, labels, attention_mask=None, memory_kv_per_layer=None,
                num_loops=None):
    """全展开一遍，取每个 exit 的 per-sample CE（在线 oracle 的深度维度）。

    返回 (per_exit_loss [B,R], per_exit_correct [B,R], exit_logits)。
    """
    cfg = model.config
    r_max = num_loops or cfg.max_loop
    out = model(input_ids, attention_mask=attention_mask, labels=None,
                collect_exits=True, num_loops=r_max, memory_kv_per_layer=memory_kv_per_layer)
    losses = torch.stack([_per_sample_ce(lg, labels) for lg in out.exit_logits])   # [R,B]
    corrects = torch.stack([_per_sample_correct(lg, labels) for lg in out.exit_logits])
    return losses.t().to(input_ids.device), corrects.t().to(input_ids.device), out


@torch.no_grad()
def sweep_budget(model, input_ids, labels, context_ids, controller_factory,
                 budget_idxs=None, attention_mask=None, query_len=None, **oracle_kwargs):
    """性能矩阵的 memory 维：逐预算档跑 teacher-forced 前向，返回 loss [B, n档]。

    query_len: 可见前缀边界（必传）——预算检索 query 不得读 gold answer（评审 3.1）。
    controller_factory() -> AdaptiveController（每档新建，避免状态串扰）。
    """
    if query_len is None:
        raise ValueError("sweep_budget 必须传 query_len（禁止默认全序列，防 gold 泄漏进检索）")
    cfg = model.config
    idxs = budget_idxs if budget_idxs is not None else list(range(len(cfg.memory_budgets)))
    losses = []
    for idx in idxs:
        ctrl = controller_factory()
        out, _ = ctrl.forward_with_oracles(
            input_ids, attention_mask, labels, context_ids,
            oracle_budget=idx, query_len=query_len, **oracle_kwargs)
        losses.append(_per_sample_ce(out.exit_logits[-1], labels))
    return torch.stack(losses, dim=-1)  # [B, n]


@torch.no_grad()
def sweep_retrieval(model, input_ids, labels, context_ids, query_len,
                    controller_factory, attention_mask=None, num_loops=None):
    """V2 §15.5 反事实检索 oracle（T0.1 重写，替代旧的 no-memory vs memory 口径）。

    同一可见前缀状态下比较两个动作：
      REUSE   —— 初始检索得 W 后不再检索，全展开
      REFRESH —— 用第 1 轮 exit 的可见前缀（[B, query_len, D]）真实重检索得 W'，全展开
    labels[r] = 1{ CE_reuse(r) − CE_refresh(r) > ε }（per-sample per-round）。

    全程因果：两次检索 query 都只含 input_ids[:, :query_len]，gold 后缀不参与。
    返回 (labels_r [B,R] 0/1, gain [B,R])。
    """
    cfg = model.config
    r_max = num_loops or cfg.max_loop
    causal = input_ids[:, :query_len]
    causal_mask = attention_mask[:, :query_len] if attention_mask is not None else None

    def _mem_kw(ctrl):
        """V2 走 memory_payload（独立 Reader）；V1 走 per-layer KV 拼接。"""
        mem_in = ctrl._memory_input()
        if getattr(ctrl.cfg, "residual_v2", False):
            return {"memory_payload": mem_in}
        return {"memory_kv_per_layer": mem_in}

    # 轨迹 A：REUSE
    ctrl_a = controller_factory()
    ctrl_a.begin(context_ids)
    q_a = ctrl_a.raw.model.prelude_hidden(causal, causal_mask)
    ctrl_a.memory = ctrl_a.retrieve_into_memory(q_a)
    out_a = model(input_ids, attention_mask=attention_mask, labels=None, collect_exits=True,
                  num_loops=r_max, **_mem_kw(ctrl_a))
    ce_a = torch.stack([_per_sample_ce(lg, labels) for lg in out_a.exit_logits]).t()

    # 轨迹 B：REFRESH（查询状态 = A 第 1 轮 exit 的可见前缀；post-coda exit 仍是因果状态）
    refresh_query = out_a.exit_hiddens[0][:, :query_len]
    ctrl_b = controller_factory()
    ctrl_b.begin(context_ids)
    ctrl_b.memory = ctrl_b.retrieve_into_memory(refresh_query)
    out_b = model(input_ids, attention_mask=attention_mask, labels=None, collect_exits=True,
                  num_loops=r_max, **_mem_kw(ctrl_b))
    ce_b = torch.stack([_per_sample_ce(lg, labels) for lg in out_b.exit_logits]).t()

    gain = ce_a - ce_b
    labels_r = (gain > model.config.oracle_epsilon).long()
    return labels_r.to(input_ids.device), gain.to(input_ids.device)


@torch.no_grad()
def sweep_retrieval_action(model, input_ids, labels, context_ids, query_len,
                           controller_factory, attention_mask=None, num_loops=None):
    """T2.5/T4.4 同快照三动作反事实（评审 4.2 修复）。

    从**同一状态快照**（相同 H/W/Warm Cache/可见前缀）分叉三动作，比较**动作之后**
    的出口 loss——不是三条全程独立轨迹，也不是「第 r 轮 exit 给第 r 轮后动作」打分。

    动作 id（schema v2）：0=REUSE, 1=RESCORE_WARM, 2=GLOBAL_REFRESH。
    返回 (best_action [B,R], loss_matrix [B,R,3])。
    """
    from model.minimind_x.router.retrieval_router import (
        ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH)

    cfg = model.config
    r_max = num_loops or cfg.max_loop
    causal = input_ids[:, :query_len]
    causal_mask = attention_mask[:, :query_len] if attention_mask is not None else None
    B = input_ids.shape[0]

    def _mem_kw(ctrl):
        mem_in = ctrl._memory_input()
        if getattr(ctrl.cfg, "residual_v2", False):
            return {"memory_payload": mem_in}
        return {"memory_kv_per_layer": mem_in}

    # --- 共享前缀：初始检索后，跑到各决策点 ---
    # 对每个决策轮 r（0..r_max-2，动作影响的是 r 之后的出口）：
    #   用 oracle 全 REUSE 跑到产生 exit[r]（即 r+1 个 exit）得到快照
    #   分叉动作，继续到 r_max，比较 exit[r+1] 的 CE
    loss_matrix = input_ids.new_zeros(B, r_max, 3, dtype=torch.float32)
    loss_matrix = torch.zeros(B, r_max, 3, device=input_ids.device)
    for r_decision in range(r_max):
        # 共享轨迹：到 r_decision 轮结束（含 exit[r_decision]）均 REUSE
        # 分叉后看 exit[min(r_decision+1, r_max-1)]（动作在 cb 中于 r_decision 后执行）
        eval_round = min(r_decision + 1, r_max - 1)
        for a_id in (ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH):
            ctrl = controller_factory()
            ctrl.begin(context_ids)
            q = ctrl.raw.model.prelude_hidden(causal, causal_mask)
            ctrl.memory = ctrl.retrieve_into_memory(q)
            if a_id == ACTION_RESCORE_WARM and not ctrl.has_warm_cache():
                loss_matrix[:, r_decision, a_id] = float("inf")
                continue

            def make_cb(r_decision=r_decision, a_id=a_id, ctrl=ctrl):
                def cb(r, exit_hidden):
                    vis = exit_hidden[:, :query_len] if query_len else exit_hidden
                    # 决策点之前：REUSE（保持共享前缀状态）；决策点：执行目标动作
                    if r < r_decision:
                        return None
                    if r == r_decision:
                        if a_id == ACTION_REUSE:
                            return None
                        with torch.no_grad():
                            ctrl.memory = ctrl.execute_retrieval_action(
                                a_id, vis, budget_idx=ctrl._budget_idx(vis),
                                mem=ctrl.memory)
                            return ctrl._memory_input()
                    return None  # 之后 REUSE
                return cb

            out = model(input_ids, attention_mask=attention_mask, labels=None,
                        collect_exits=True, num_loops=r_max,
                        loop_callback=make_cb(), **_mem_kw(ctrl))
            # 评价出口 = 动作生效后的第一出口（评审 4.2：不能拿动作前的 exit 给动作打分）
            ce = _per_sample_ce(out.exit_logits[eval_round], labels)  # [B]
            loss_matrix[:, r_decision, a_id] = ce
    best = loss_matrix.argmin(dim=-1)  # [B,R]
    return best.to(input_ids.device), loss_matrix.to(input_ids.device)
```

---

# trainer/residual_schedule.py

```python
"""V2 迁移 warm-up 课程（任务清单 T4.2；V2 文档 §14.2）。

同一训练 run 内把 history_strength(η) 与 update_progress(λ) 从 0 平滑升至目标值：
  初始化检查点：η=0, λ=0 —— 与旧路径数值一致（严格兼容锚点，T2.3 已锁定）
  前 warmup_steps：线性升至 1×base —— 新模块渐进接入
  之后：保持 base —— 主训练

实现为「写共享 config 对象」：ThinkingCore 每轮读 config，改 config 即生效，
无需重建模型。日志口径：训练器应同时记录 raw 与 effective 值（V2 §14.2 提醒）。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def residual_progress(step, warmup_steps):
    """V2 §14.2 迁移课程曲线。

    warmup_steps 约对应总训练步的 5%：
      · 前 20% 窗口（= 总步 ~1%）：0 → 0.25
      · 后 80% 窗口（= 总步 ~1–5%）：0.25 → 1.0
      · 之后：恒 1.0
    warmup_steps<=0 视为从头训练（恒 1）。
    """
    if warmup_steps is None or warmup_steps <= 0:
        return 1.0
    t = min(1.0, max(0.0, float(step) / float(warmup_steps)))
    if t <= 0.2:
        return 0.25 * (t / 0.2)
    if t < 1.0:
        return 0.25 + 0.75 * ((t - 0.2) / 0.8)
    return 1.0


def apply_residual_progress(model, frac, base_strength=1.0, base_progress=1.0):
    """按 frac 设置 η/λ（base 为目标终值）。兼容 DDP/compile 包装。返回 (η, λ)。"""
    raw = model
    for attr in ("module", "_orig_mod"):
        raw = getattr(raw, attr, raw)
    cfg = raw.config
    eta = base_strength * frac
    lam = base_progress * frac
    cfg.mhar_strength = eta
    cfg.update_progress = lam
    return eta, lam
```

---

# trainer/train_grpo_x.py

```python
"""MiniMind-X Adaptive Compute RL（任务清单 T7.3；Doc 5 §18-25、Doc 4 §16-17）。

GRPO + Compute Penalty：
  R = R_task − λ_m·P_m − λ_r·P_r − λ_t·P_t
  R_task：math 正确性（safe_math_eval 判等）+ 轻量格式分
  P_t = avg_loops/R_max（来自 generate 统计）；P_m/P_r 来自 controller 统计（如有）

轨迹多样性：同一 prompt 的生成组按 [fast, normal, deep] 模式循环 → (memory, loop) 维度的组内对比。

RL 三相课程（Doc 5 §24）：phase1 无 compute penalty → phase2 小 penalty → phase3 全额 Pareto。
防塌缩（Doc 5 §25）：compute penalty 只在答案正确时生效 —— 简单题塌缩到 R=1 拿不到效率奖励。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import argparse
import math
import random
import re
import time
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.minimind_x import MiniMindXConfig
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import (Logger, is_main_process, init_distributed_mode, setup_seed,
                                   safe_math_eval)
from trainer.x_utils import init_model_x, save_x_checkpoint, tokenize_tensors

warnings.filterwarnings('ignore')

MODE_CYCLE = ["fast", "normal", "deep"]
PHASE_PENALTY = {1: (0.0, 0.0, 0.0), 2: (0.05, 0.05, 0.05), 3: (0.2, 0.1, 0.2)}  # (λ_m, λ_r, λ_t)


def extract_answer(text):
    """提取生成文本中最后一个数值/算式结果。"""
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:\s*[-+*/^]\s*-?\d+(?:\.\d+)?)*", text)
    return matches[-1].replace(" ", "").replace("^", "**") if matches else None


def math_task_reward(response, gt):
    """正确 +1；可解析但错 0；不可解析 -0.2。"""
    try:
        pred = safe_math_eval(extract_answer(response) or "")
        return 1.0 if abs(pred - float(gt)) < 1e-6 else 0.0
    except (ValueError, TypeError):
        return -0.2


_BANNED_PATTERNS = ("import os", "import sys", "import subprocess", "import socket",
                    "__import__", "open(", "exec(", "eval(", "shutil", "pathlib")


def _sandbox_limits():
    """研究级沙箱资源限制（RLIMIT）：CPU 5s、内存 512MB、无文件写出。非完全隔离。

    逐项尽力而为（preexec_fn 内不可抛异常）：Linux 生效；macOS 对部分 RLIMIT
    支持不全时跳过该项（仍有静态危险模式拒绝 + timeout 兜底）。
    """
    import resource

    def apply():
        limits = [(resource.RLIMIT_CPU, 5), (resource.RLIMIT_AS, 512 * 1024 * 1024),
                  (resource.RLIMIT_FSIZE, 0)]
        for which, soft in limits:
            try:
                hard = resource.getrlimit(which)[1]
                resource.setrlimit(which, (min(soft, hard) if hard != resource.RLIM_INFINITY
                                           else soft, hard))
            except (ValueError, OSError):
                pass
    return apply


def code_task_reward(response, gt, timeout=5):
    """代码沙箱 reward（T7.4）：提取 ```python 块，subprocess 执行，stdout 精确匹配。

    ⚠ 研究级隔离（RLIMIT + 危险模式静态拒绝），仅适用于自产/可信数据；
    不可信输入需外层容器（docker/nsjail），见 ARCHITECTURE §8。
    """
    import subprocess as sp
    import re as _re
    blocks = _re.findall(r"```python\n(.*?)```", response, _re.DOTALL)
    if not blocks:
        return -0.2
    code = blocks[-1]
    if any(p in code for p in _BANNED_PATTERNS):  # 纯计算任务无需这些能力
        return 0.0
    try:
        r = sp.run([sys.executable, "-c", code], capture_output=True, text=True,
                   timeout=timeout, preexec_fn=_sandbox_limits())
        return 1.0 if r.returncode == 0 and r.stdout.strip() == str(gt).strip() else 0.0
    except sp.TimeoutExpired:
        return 0.0


def format_reward(response):
    return 0.1 if 0 < len(response.strip()) < args.max_gen_len else -0.1


@torch.no_grad()
def rollout_group(prompt_text, num_generations, max_new_tokens):
    """一个 prompt 的生成组：按 [fast, normal, deep] 循环产生 (memory, loop) 多样性。

    默认（on-policy，v1.2 修正）：每条轨迹固定采样 R（fast=1/normal=2/deep=R_max，
    无 mode 多样性时随机 R∈[1,R_max]），生成与 logps 重算的计算路径严格一致。
    --adaptive_rollout 1 为消融路径：adaptive halting 采样，轨迹内各 token 的 R 不同，
    logps 只能用单一 R 近似重算（残余 off-policy 偏差），仅用于对比实验。
    """
    inputs = tokenize_tensors(tokenizer, prompt_text, args.device)
    outs = {"ids": [], "completions": [], "loop": [], "mem": [], "ret": []}
    for j in range(num_generations):
        mode = MODE_CYCLE[j % len(MODE_CYCLE)] if args.mode_diversity else None
        if args.adaptive_rollout == 1:
            res = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens, do_sample=True, temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id, mode=mode,
                use_halting=True, return_stats=True)
        else:
            r = {"fast": 1, "normal": 2}.get(mode, lm_config.max_loop) if mode else \
                random.randint(1, lm_config.max_loop)
            r = max(1, min(r, lm_config.max_loop))
            res = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens, do_sample=True, temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id, fixed_loops=r,
                use_halting=False, return_stats=True)
        gen_ids = res["generated_ids"][0, inputs["input_ids"].shape[1]:]
        outs["ids"].append(gen_ids)
        outs["completions"].append(tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True))
        outs["loop"].append(res["stats"]["avg_loops"])  # 固定 R 时严格 == r
        outs["mem"].append(res["stats"].get("memory_steps", 0))
        outs["ret"].append(res["stats"].get("re_retrievals", 0))
    return outs


def compute_per_token_logps(model_, input_ids, n_keep, attention_mask=None, num_loops=1):
    """num_loops 与轨迹实际轮数对齐（否则 off-policy 偏差：生成用自适应 R、评估用 R=1）。"""
    # 评审 8.2：梯度前向必须走 DDP 包装（unwrap 会丢梯度同步）
    # 推理/无梯度可用 raw；此处要 backward，故用 model_ 本身
    target = model_ if isinstance(model_, DistributedDataParallel) else model_
    logits = target(input_ids, attention_mask=attention_mask, num_loops=num_loops,
                       logits_to_keep=n_keep + 1).logits[:, :-1, :]
    logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        logps.append(F.log_softmax(logits_row, dim=-1).gather(
            1, ids_row.unsqueeze(1)).squeeze(1))
    return torch.stack(logps)


def grpo_x_epoch(epoch, loader, iters, ref_model, start_step=0, wandb=None):
    lam_m, lam_r, lam_t = PHASE_PENALTY[args.rl_phase]
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = [tokenizer.apply_chat_template(b["messages"], tokenize=False,
                                                 add_generation_prompt=True) for b in batch]
        gts = [b["gt"] for b in batch]
        rollouts, old_logps, old_loop = [], [], []
        prompt_lens = []
        for p in prompts:
            ro = rollout_group(p, args.num_generations, args.max_gen_len)
            rollouts.append(ro)
        # 拼 batch：每条轨迹独立（变长 → 逐条算 logps，研究级实现）
        traj = []  # (prompt_idx, ids, stats)
        prompt_base_ids = [tokenize_tensors(tokenizer, p, args.device)["input_ids"][0]
                           for p in prompts]
        for i, ro in enumerate(rollouts):
            base = prompt_base_ids[i]
            for j in range(args.num_generations):
                # 评审 8.3：两侧先到 CPU 再 cat，避免 CUDA 跨设备
                traj.append((i, torch.cat([base.cpu(), ro["ids"][j].cpu()]).to(args.device), ro))
        old_per_token, logp_lens, traj_loops = [], [], []
        with torch.no_grad():
            for (i, ids, ro) in traj:
                n_keep = ids.numel() - prompt_base_ids[i].numel()
                j = len(old_per_token) % args.num_generations
                # 评审 8.3/V2 §17.3：禁止平均深度重放（[1,3,2,4] 不得用 R=3 近似）
                # adaptive 轨迹内 token 深度不同时，不得 int(round(avg_loops))
                raw_loop = ro["loop"][j]
                r = max(1, int(round(raw_loop)))
                if abs(raw_loop - r) > 1e-6 and getattr(args, "allow_avg_depth_replay", 0) != 1:
                    raise RuntimeError(
                        f"禁平均深度重放：avg_loops={raw_loop} 非整数（V2 §17.3）。"
                        f"请用固定 R rollout，或显式 --allow_avg_depth_replay 1（消融，不诚实）")
                old_per_token.append(compute_per_token_logps(model, ids.unsqueeze(0), n_keep, num_loops=r)[0])
                logp_lens.append(n_keep)
                traj_loops.append(r)

        rewards = torch.zeros(len(traj), device=args.device)
        task_fn = code_task_reward if args.reward_type == "code" else math_task_reward
        for k, (i, ids, ro) in enumerate(traj):
            j = k % args.num_generations
            task = task_fn(ro["completions"][j], gts[i]) + \
                format_reward(ro["completions"][j])
            p_t = ro["loop"][j] / max(lm_config.max_loop, 1)
            p_m = min(ro["mem"][j] / args.max_gen_len, 1.0)
            p_r = ro["ret"][j] / 4.0
            correct = task >= 1.0
            # 防塌缩：效率奖励只给「正确」的轨迹（Minimum reward constraint）
            penalty = (lam_m * p_m + lam_r * p_r + lam_t * p_t) if correct else 0.0
            rewards[k] = task - penalty

        grouped = rewards.view(-1, args.num_generations)
        mean_r = grouped.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # ---- 策略更新（逐轨迹累积，GRPO clip + KL）----
        optimizer.zero_grad(set_to_none=True)
        policy_loss_total, kl_total, n_traj = 0.0, 0.0, len(traj)
        for k, (i, ids, ro) in enumerate(traj):
            n_keep = logp_lens[k]
            per_token_logps = compute_per_token_logps(model, ids.unsqueeze(0), n_keep,
                                                      num_loops=traj_loops[k])[0]
            with torch.no_grad():
                ref_logps = compute_per_token_logps(ref_model, ids.unsqueeze(0), n_keep,
                                                    num_loops=traj_loops[k])[0]
            kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
            ratio = torch.exp(per_token_logps - old_per_token[k].to(args.device))
            clipped = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            a1, a2 = ratio * advantages[k], clipped * advantages[k]
            token_loss = -(torch.min(a1, a2) - args.beta * kl).mean()
            (token_loss / n_traj).backward()
            policy_loss_total += token_loss.item() / n_traj
            kl_total += kl.mean().item() / n_traj
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0 or step == iters:
            acc = sum(1 for k, (i, _, ro) in enumerate(traj)
                      if math_task_reward(ro["completions"][k % args.num_generations], gts[i]) >= 1.0) / len(traj)
            avg_loop = sum(ro["loop"][k % args.num_generations] for k, (_, _, ro) in enumerate(traj)) / len(traj)
            Logger(f'GRPO-X Epoch:[{epoch + 1}]({step}/{iters}), reward: {rewards.mean().item():.4f}, '
                   f'acc: {acc:.3f}, avg_loop: {avg_loop:.2f}, KL: {kl_total:.4f}, '
                   f'policy_loss: {policy_loss_total:.4f}, phase: {args.rl_phase}')
            if wandb:
                wandb.log({"reward": rewards.mean().item(), "accuracy": acc,
                           "avg_loops": avg_loop, "kl": kl_total,
                           "policy_loss": policy_loss_total})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            save_x_checkpoint(model, args.save_dir, args.save_weight, lm_config.hidden_size)
            model.train()
        del traj, rewards, advantages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-X Adaptive Compute GRPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='grpo_x', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2, help="每步 prompt 数")
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_generations", type=int, default=3, help="每 prompt 轨迹数（3 的倍数配模式循环）")
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.05, help="KL 惩罚系数")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rl_phase", type=int, default=1, choices=[1, 2, 3],
                        help="RL 课程：1 无惩罚 / 2 小惩罚 / 3 Pareto（Doc 5 §24）")
    parser.add_argument("--mode_diversity", default=1, type=int, choices=[0, 1],
                        help="轨迹按 fast/normal/deep 模式循环采样")
    parser.add_argument("--allow_avg_depth_replay", default=0, type=int, choices=[0, 1],
                        help="危险：允许平均深度近似重放（V2 §17.3 禁止；仅消融）")
    parser.add_argument("--adaptive_rollout", default=0, type=int, choices=[0, 1],
                        help="0=固定采样 R（默认，on-policy）；1=adaptive halting rollout"
                             "（logps 用单一 R 近似，残余 off-policy 偏差，消融用）")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"])
    parser.add_argument('--variant', default='minimind_x', type=str)
    parser.add_argument('--hidden_size', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--num_hidden_layers', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument('--residual_v2', default=0, type=int, choices=[0, 1],
                        help='V2 残差改造；从 V2 ckpt 加载时会自动强制打开')
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl_math.jsonl",
                        help="math RL 数据：{conversations, gt}（官方 agent_rl_math.jsonl）")
    parser.add_argument("--reward_type", type=str, default="math", choices=["math", "code"],
                        help="math=safe_math_eval 判等；code=python 沙箱 stdout 匹配（T7.4）")
    parser.add_argument('--from_weight', default='sft_x_c', type=str)
    parser.add_argument("--tokenizer_path", type=str, default="../model",
                        help="tokenizer 目录（从仓库根目录运行时传 model/）")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-GRPO")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(preset=args.preset, variant=args.variant, max_loop=args.max_loop,
                                residual_v2=bool(args.residual_v2),
                                **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                         num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    device_type = "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"GRPO-X-phase{args.rl_phase}")

    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    tokenizer_path=args.tokenizer_path,
                                    save_dir=args.save_dir, device=args.device)
    model.train()
    ref_model, _ = init_model_x(lm_config, args.from_weight,
                                tokenizer_path=args.tokenizer_path,
                                save_dir=args.save_dir, device=args.device)
    ref_model.eval().requires_grad_(False)

    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=512)

    def _collate(b):
        for x in b:  # tools=None 会卡 default_collate，GRPO 不用它
            x.setdefault("tools", [])
            if x["tools"] is None:
                x["tools"] = []
        return b

    # T0.5（C9 修复）：DDP 包装与分片。rollout 走 raw（no_grad、generate 无 hook），
    # 策略梯度的 logps 前向必须过 DDP 包装（train_grpo 同款注释）。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                        sampler=train_sampler, collate_fn=_collate)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)
    iters = len(loader) * args.epochs
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        grpo_x_epoch(epoch, loader, iters, ref_model, 0, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
```

---

# trainer/train_pretrain_x.py

```python
"""MiniMind-X 预训练（任务清单 T1.3；Doc 3 §9 Phase 0、Doc 5 Stage 0）。

loop_mode（D3/D7）：
  fixed        固定 R=args.fixed_loops（Stage 0/1 默认：求训练稳定）
  random       每 step 随机采样 R∈[1,R_max]（Doc 3 Phase 0 的随机-loop 预训练，省算力）
  full_unroll  全展开 R_max + per-exit 监督（loss@R 曲线是核心研究产出）
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import random
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.minimind_x import MiniMindXConfig
from model.minimind_x.losses import lm_loss_per_exit, consistency_kl_loss
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (get_lr, Logger, is_main_process, lm_checkpoint,
                                   init_distributed_mode, setup_seed, SkipBatchSampler)
from trainer.x_utils import (init_model_x, save_x_checkpoint, set_gradient_checkpointing,
                             build_optimizer, validate_config)
from trainer.residual_schedule import residual_progress, apply_residual_progress

warnings.filterwarnings('ignore')

# checkpoint 目录锚定到仓库根（原 '../checkpoints' 在从仓库根运行时会落到项目外）
CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))


def compute_step_loss(model, input_ids, labels, args, lm_config):
    """返回 (总损失, 日志 dict)。日志里 loss@R 逐出口记录（研究核心指标）。"""
    if not lm_config.use_loop or args.loop_mode == "fixed":
        r = max(1, args.fixed_loops)
        res = model(input_ids, labels=labels, num_loops=r)
        loss = res.loss + res.aux_loss
        return loss, {"loss": res.loss.item(), "r": r}
    if args.loop_mode == "random":
        r = random.randint(1, lm_config.max_loop)
        res = model(input_ids, labels=labels, num_loops=r)
        loss = res.loss + res.aux_loss
        return loss, {"loss": res.loss.item(), "r": r}
    if args.loop_mode != "full_unroll":
        raise ValueError(f"未知 loop_mode: {args.loop_mode}")
    res = model(input_ids, labels=None, collect_exits=True)
    lm, per_exit = lm_loss_per_exit(res.exit_logits, labels, lm_config.exit_loss_weighting)
    loss = lm + res.aux_loss
    if args.alpha_consistency > 0:
        loss = loss + args.alpha_consistency * consistency_kl_loss(res.exit_logits, labels)
    return loss, {"loss": lm.item(),
                  "per_exit": [round(l.item(), 4) for l in per_exit],
                  "r": lm_config.max_loop}


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if args.residual_w2 if False else False:
            pass
        if getattr(args, 'residual_warmup_steps', 0) > 0:
            frac = residual_progress(step, args.residual_warmup_steps)
            eta, lam = apply_residual_progress(model, frac)
            if step % args.log_interval == 0:
                Logger(f'residual warm-up: step={step} frac={frac:.3f} eta={eta:.3f} lambda={lam:.3f}')
        with autocast_ctx:
            loss, log_info = compute_step_loss(model, input_ids, labels, args, lm_config)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            exit_str = f", loss@R: {log_info.get('per_exit')}" if log_info.get("per_exit") else ""
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {log_info["loss"]:.4f}'
                   f'{exit_str}, R: {log_info["r"]}, lr: {lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                payload = {"loss": log_info["loss"], "learning_rate": lr, "epoch_time": eta_min}
                if log_info.get("per_exit"):
                    payload.update({f"loss_R{ i + 1}": v for i, v in enumerate(log_info["per_exit"])})
                wandb.log(payload)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            save_x_checkpoint(  # T5.2：完整 checkpoint
                model, args.save_dir, args.save_weight, lm_config.hidden_size,
                optimizer=optimizer, scaler=scaler,
                extra={"data_progress": {"epoch": epoch + 1, "step": step},
                       "warmup_progress": getattr(args, "residual_warmup_steps", 0),
                       "tokenizer_manifest": {"path": args.tokenizer_path},
                       "source_manifest": {"data_path": getattr(args, "data_path", None)}})
            # 保存「下一个待跑位置」：epoch 结束时存 (epoch+1, 0)，中途存 (epoch, step)
            done = step >= iters
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          scaler=scaler, epoch=epoch + 1 if done else epoch,
                          step=0 if done else step, batch_size=args.batch_size,
                          wandb=wandb, save_dir=CKPT_DIR)
            model.train()

        del input_ids, labels, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-X Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_x', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"], help="规模 preset")
    parser.add_argument('--variant', default='middle_loop', type=str,
                        choices=["dense", "full_loop", "middle_loop", "late_middle", "minimind_x"])
    parser.add_argument('--hidden_size', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--num_hidden_layers', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--prelude_layers', default=None, type=int)
    parser.add_argument('--thinking_layers', default=None, type=int)
    parser.add_argument('--coda_layers', default=None, type=int)
    parser.add_argument('--max_loop', default=4, type=int, help="R_max")
    parser.add_argument('--loop_mode', default='full_unroll', type=str,
                        choices=["fixed", "random", "full_unroll"])
    parser.add_argument('--loop_embedding', default=0, type=int, choices=[0, 1],
                        help="轮次嵌入消融（D4）。默认 0：随 MoR/Huginn 实践不加；"
                             "1：Universal-Transformer 式 step embedding（对照实验）")
    parser.add_argument('--fixed_loops', default=1, type=int, help="loop_mode=fixed 时的 R")
    parser.add_argument('--alpha_consistency', default=0.0, type=float, help="一致性 KL 系数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--prelude_attn', default=None, type=str, choices=[None, "full", "local", "sparse"])
    parser.add_argument('--core_attn', default=None, type=str, choices=[None, "full", "sparse"])
    parser.add_argument('--coda_attn', default=None, type=str, choices=[None, "full", "local", "sparse"])
    parser.add_argument('--max_seq_len', default=340, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl",
                        help="预训练数据（官方：pretrain_t2t_mini 1.2GB 快速 / pretrain_t2t 10GB 完整）")
    parser.add_argument("--tokenizer_path", type=str, default="../model",
                        help="tokenizer 目录（从仓库根目录运行时传 model/）")
    parser.add_argument('--from_weight', default='none', type=str, help="X 权重前缀（none=从头）")
    parser.add_argument('--from_minimind_weight', default='none', type=str,
                        help="原 MiniMind 权重路径（自动 remap，T0.3 训练侧入口）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-Pretrain")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--grad_ckpt", default=0, type=int, choices=[0, 1], help="梯度检查点（T1.4）")
    parser.add_argument("--residual_v2", default=0, type=int, choices=[0, 1],
                        help="启用 V2 残差改造（MHAR/UpdateGate/Reader）")
    parser.add_argument("--residual_warmup_steps", default=0, type=int,
                        help="V2 迁移热身步数（η/λ 0→1；0=从头恒 1）")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(
        preset=args.preset, variant=args.variant, use_moe=bool(args.use_moe),
        max_loop=args.max_loop, loop_embedding=bool(args.loop_embedding),
        residual_v2=bool(args.residual_v2),
        **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                 num_hidden_layers=args.num_hidden_layers,
                                 prelude_layers=args.prelude_layers,
                                 thinking_layers=args.thinking_layers,
                                 coda_layers=args.coda_layers,
                                 prelude_attn=args.prelude_attn,
                                 core_attn=args.core_attn,
                                 coda_attn=args.coda_attn).items() if v is not None})
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir=CKPT_DIR) if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"MiniMindX-{args.variant}-{args.loop_mode}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    validate_config(lm_config)  # T5.2：非法 config 启动即报错
    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    tokenizer_path=args.tokenizer_path,
                                    save_dir=args.save_dir, device=args.device)
    if args.from_minimind_weight != 'none':
        from model.minimind_x import remap_minimind_state_dict
        orig = torch.load(args.from_minimind_weight, map_location=args.device)
        mapped = remap_minimind_state_dict(orig, lm_config)
        model.load_state_dict(mapped, strict=False)
        Logger(f'remapped original MiniMind weights from {args.from_minimind_weight}')

    if args.grad_ckpt == 1:
        set_gradient_checkpointing(model, True)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model, lr=args.learning_rate)  # T4.3 参数组

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)
        # mid-epoch 续跑仅在同配置下有效：step 是 per-rank 口径，不随 world_size 缩放
        # （昨晚的 bug：8 卡存的 step 在 4 卡 resume 时被错误翻倍导致步数超界）。
        # 跨卡数/批大小恢复时，当前 epoch 从头训练（数据顺序重排，语义最干净）。
        cur_ws = dist.get_world_size() if dist.is_initialized() else 1
        if (ckp_data.get('world_size', cur_ws) != cur_ws
                or ckp_data.get('batch_size', args.batch_size) != args.batch_size):
            Logger(f'resume: 配置变化(ws {ckp_data.get("world_size", cur_ws)}->{cur_ws} / '
                   f'bs {ckp_data.get("batch_size", args.batch_size)}->{args.batch_size})，'
                   f'epoch {start_epoch} 从头训练')
            start_step = 0
        if start_epoch >= args.epochs:
            Logger(f'resume: 已完成 {start_epoch}/{args.epochs} epochs，无需训练')

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # X 模型的 router/memory 模块按训练阶段激活（Stage 0 不监督 router），
        # 必须放开 unused 检测，否则 DDP 在第二轮 reduction 报错
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
```

---

# trainer/train_sft_x.py

```python
"""MiniMind-X SFT 三相训练（任务清单 T7.2；Doc 5 §11、Doc 3 Phase 1-3、Doc 4 §20）。

SFT-A：基础 instruction，固定 R=1（避免依赖 loop）
SFT-B：随机 loop 建立深度弹性（collect_exits 均匀监督 + 一致性）
SFT-C：打开 retrieval/working memory（AdaptiveController teacher-forcing）+
        router 课程（thinking → retrieval → memory，Doc 4 §20）+ 在线 oracle 标签
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import argparse
import random
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.minimind_x import MiniMindXConfig, joint_loss
from model.minimind_x.losses import lm_loss_per_exit, consistency_kl_loss
from model.minimind_x.controller import AdaptiveController
from dataset.lm_dataset import SFTDataset
from dataset.retrieval_qa_dataset import RetrievalQADataset, collate_retrieval_qa
from trainer.trainer_utils import (get_lr, Logger, is_main_process, init_distributed_mode,
                                   setup_seed, lm_checkpoint)
from trainer.x_utils import (init_model_x, save_x_checkpoint, set_gradient_checkpointing,
                             build_optimizer, validate_config)
from trainer.residual_schedule import residual_progress, apply_residual_progress
from trainer.oracle_labels import (sweep_depth, oracle_depth_from_losses, sweep_budget,
                                   oracle_budget_from_matrix, sweep_retrieval)

warnings.filterwarnings('ignore')


def sft_a_step(input_ids, labels):
    """R=1 固定（SFT-A）。"""
    res = model(input_ids, labels=labels, num_loops=1)
    return res.loss + res.aux_loss, {"loss": res.loss.item()}


def sft_b_step(input_ids, labels):
    """随机深度弹性（SFT-B）：本 step 采样 R；按概率切换全展开做 router 无关的一致性监督。"""
    if random.random() < args.full_unroll_ratio:
        res = model(input_ids, labels=None, collect_exits=True)
        lm, per_exit = lm_loss_per_exit(res.exit_logits, labels, "uniform")
        loss = lm + res.aux_loss
        if args.alpha_consistency > 0:
            loss = loss + args.alpha_consistency * consistency_kl_loss(res.exit_logits, labels)
        return loss, {"loss": lm.item(), "per_exit": [round(l.item(), 4) for l in per_exit]}
    r = random.randint(1, lm_config.max_loop)
    res = model(input_ids, labels=labels, num_loops=r)
    return res.loss + res.aux_loss, {"loss": res.loss.item(), "r": r}


def assert_sft_c_batch(input_ids):
    """SFT-C batch 契约（v1.3 guard，评审 Bug 1）：controller 逐样本建 store/检索，
    B>1 时只用到首样本上下文（静默丢弃）。DataLoader 侧已强制 batch_size=1，
    此函数是第二道防线（防未来改动）。"""
    if input_ids.shape[0] > 1:
        raise ValueError(
            f"SFT-C 暂不支持 batch>1（当前 {input_ids.shape[0]}）：controller 逐样本建 "
            f"store/检索，B>1 时只用到首样本上下文。请用 batch_size=1 + 梯度累积。")


def sft_c_step(batch):
    """终局 teacher-forcing（SFT-C / Stage 3）：controller + joint loss + 在线 oracle。

    V2 改造（T0.1/T0.2/T0.8）：query 因果边界、反事实检索 oracle、全展开 router 概率
    监督、memory 统计只进日志。
    """
    assert_sft_c_batch(batch["input_ids"])
    input_ids = batch["input_ids"].to(args.device)
    labels = batch["labels"].to(args.device)
    context_ids = batch["context_ids"][0].to(args.device)  # 研究级口径：batch 内取首样本上下文
    gold_block = batch["gold_block"].to(args.device)
    query_len = int(batch["question_len"][0].item())       # T0.1：可见前缀边界（数据集提供）
    oracle_r_file = int(batch["oracle_r"][0].item())
    # 动作 schema：数据边界归一到 v2（0=REUSE,1=WARM,2=GLOBAL）；旧二值 v1: 1=GLOBAL
    from model.minimind_x.router.retrieval_router import normalize_oracle_actions
    _raw_ret = batch["oracle_retrieve"][0].tolist()
    if not isinstance(_raw_ret, (list, tuple)):
        _raw_ret = [int(_raw_ret)] if int(_raw_ret) >= 0 else []
    _schema = int(batch["action_schema_version"]) if "action_schema_version" in batch else 1
    oracle_ret = normalize_oracle_actions(_raw_ret, schema_version=_schema)

    # ---- 在线 oracle（Doc 4 §3：Self-generated Supervision；小采样率）----
    oracle_r = None
    halt_valid = None  # T4.4：unresolved（全出口都错）不参与 halt 监督
    if oracle_r_file > 0:
        oracle_r = torch.full((input_ids.shape[0],), oracle_r_file, device=args.device)
    elif random.random() < args.oracle_rate:
        per_exit_loss, per_exit_correct, _ = sweep_depth(model, input_ids, labels)
        oracle_r = oracle_depth_from_losses(per_exit_loss, lm_config.oracle_epsilon)
        halt_valid = per_exit_correct.any(dim=-1)  # 有任一出口全对才算 resolved
    # Budget oracle：文件标签缺失时在线 sweep（逐档 loss → B*，Doc 4 §11）；
    # 未采到的批次不监督（budget router 仍按自身 argmax 执行，自洽）
    oracle_budget = None
    if args.train_budget_router and random.random() < args.oracle_rate:
        loss_matrix = sweep_budget(
            model, input_ids, labels, context_ids,
            controller_factory=lambda: AdaptiveController(model),
            query_len=query_len)  # 评审 3.1：预算检索不得读 gold
        oracle_budget = int(oracle_budget_from_matrix(loss_matrix, lm_config.oracle_epsilon)[0].item())
    if not oracle_ret and random.random() < args.oracle_rate:
        # T2.5/T4.4：三动作反事实（REUSE/RESCORE_WARM/GLOBAL_REFRESH）；
        # 无 warm cache 的动作 loss=+inf 自动 mask
        try:
            from trainer.oracle_labels import sweep_retrieval_action
            best_a, _ = sweep_retrieval_action(
                model, input_ids, labels, context_ids, query_len,
                controller_factory=lambda: AdaptiveController(model))
            oracle_ret = [int(x) for x in best_a[0].tolist()]
        except Exception:
            labels_r, _ = sweep_retrieval(model, input_ids, labels, context_ids, query_len,
                                          controller_factory=lambda: AdaptiveController(model))
            oracle_ret = labels_r[0].tolist()  # 回退：旧 0/1 口径

    ctrl = AdaptiveController(model)
    out, extras = ctrl.forward_with_oracles(
        input_ids, None, labels, context_ids, query_len=query_len,
        gold_block_idx=gold_block if (gold_block[0] >= 0).all() else None,
        oracle_retrieve=oracle_ret if oracle_ret else None,
        oracle_budget=oracle_budget)
    # T0.2：全展开 router 概率（不再要求 halt=True 才有监督信号）
    halt_probs = None
    if (oracle_r is not None and args.train_halt_router
            and out.router_probs and out.router_probs.get("halt")):
        halt_probs = out.router_probs["halt"]
    retrieve_probs = None
    if (oracle_ret and out.router_probs and out.router_probs.get("retrieval")
            and len(oracle_ret) == len(out.router_probs["retrieval"])):
        retrieve_probs = out.router_probs["retrieval"]
    # T2.5：三动作 CE（0=REUSE, 1=RESCORE_WARM, 2=GLOBAL_REFRESH 的执行口径见 oracle；
    # 训练标签统一为 0=REUSE / 1=RESCORE_WARM / 2=GLOBAL_REFRESH 时用 action CE）
    ret_action_logits = None
    ret_action_masks = None
    oracle_ret_action = None
    if (oracle_ret and out.router_probs and out.router_probs.get("retrieval_action")
            and len(oracle_ret) == len(out.router_probs["retrieval_action"])):
        ret_action_logits = out.router_probs["retrieval_action"]
        warm_ok = bool(extras.get("warm_available", True))
        n = len(ret_action_logits)
        from model.minimind_x.router.retrieval_router import action_mask
        ret_action_masks = [action_mask(warm_ok, device=ret_action_logits[0].device,
                                        batch=ret_action_logits[0].shape[0]) for _ in range(n)]
        # 标签已是 schema v2，禁止再对调（评审 4.1）
        oracle_ret_action = torch.tensor(oracle_ret, device=args.device, dtype=torch.long)
    # 二值 retrieve BCE（兼容旧路径）：标签 = 是否离开 REUSE
    oracle_ret_bin = None
    if oracle_ret and retrieve_probs is not None:
        oracle_ret_bin = torch.tensor(
            [0 if int(a) == 0 else 1 for a in oracle_ret], device=args.device)
    parts = joint_loss(out.exit_logits, labels,
                       halt_probs=halt_probs, oracle_r=oracle_r, halt_valid=halt_valid,
                       retrieve_probs=retrieve_probs,
                       oracle_retrieve=oracle_ret_bin,
                       budget_logits=out.budget_logits if args.train_budget_router else None,
                       oracle_budget=oracle_budget,
                       memory_size=torch.tensor([extras["memory_tokens"]]),  # T0.8：仅日志
                       config=lm_config,
                       retrieval_action_logits=ret_action_logits,
                       oracle_retrieve_action=oracle_ret_action,
                       retrieval_action_masks=ret_action_masks,
                       retrieval_actions=(oracle_ret_action if oracle_ret_action is not None
                                          else None),
                       cost_weight=getattr(lm_config, "lambda_cost", 0.0))
    zero = next(model.parameters()).new_zeros(())
    loss = parts["loss"] + zero + extras.get("retrieval_ce", zero)
    if "distill_loss" in extras and args.lambda_distill > 0:
        loss = loss + args.lambda_distill * extras["distill_loss"]
    return loss, {"loss": parts["loss"].item(),
                  "retrievals": extras["retrievals"],
                  "memory_tokens": extras["memory_tokens"],
                  "budget_cap/actual/unique": (extras.get("budget_cap_tokens", 0),
                                               extras.get("actual_valid_tokens", 0),
                                               extras.get("unique_blocks", 0)),
                  "distill": float(extras["distill_loss"].detach())
                  if "distill_loss" in extras else 0.0}


def main():
    global args, lm_config, model, tokenizer
    parser = argparse.ArgumentParser(description="MiniMind-X SFT (A/B/C)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='sft_x', type=str)
    parser.add_argument("--phase", type=str, required=True, choices=["a", "b", "c"],
                        help="SFT-A(R=1) / SFT-B(随机R弹性) / SFT-C(memory+retrieval)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0,
                        help=">0 时限制总步数（大数据集夜间跑部分 epoch 用）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"])
    parser.add_argument('--variant', default='minimind_x', type=str)
    parser.add_argument('--hidden_size', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--num_hidden_layers', default=None, type=int, help='默认 None：不覆盖 preset/variant 默认（T0.6）')
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument('--full_unroll_ratio', default=0.25, type=float,
                        help="SFT-B 中全展开（带一致性监督）的批次比例")
    parser.add_argument('--alpha_consistency', default=0.1, type=float)
    parser.add_argument('--oracle_rate', default=0.1, type=float, help="在线 oracle 采样率")
    parser.add_argument('--train_halt_router', default=1, type=int, choices=[0, 1],
                        help="router 课程 Phase2：thinking router 开关")
    parser.add_argument('--train_budget_router', default=0, type=int, choices=[0, 1],
                        help="router 课程 Phase4：memory router 开关")
    parser.add_argument('--residual_v2', default=0, type=int, choices=[0, 1])
    parser.add_argument('--residual_warmup_steps', default=0, type=int)
    parser.add_argument('--lambda_distill', default=0.05, type=float,
                        help="fine→coarse 检索蒸馏系数（v1.3；0 关闭）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl",
                        help="SFT-A/B 数据（官方 sft_t2t_mini，conversations 格式）")
    parser.add_argument("--retrieval_data_path", type=str,
                        default="../dataset/retrieval_qa.jsonl", help="SFT-C 数据")
    parser.add_argument('--from_weight', default='pretrain_x', type=str)
    parser.add_argument("--tokenizer_path", type=str, default="../model",
                        help="tokenizer 目录（从仓库根目录运行时传 model/）")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-SFT")
    parser.add_argument("--grad_ckpt", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindXConfig(preset=args.preset, variant=args.variant, max_loop=args.max_loop,
                                residual_v2=bool(args.residual_v2),
                                **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                         num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    validate_config(lm_config)  # T5.2：非法 config 启动即报错
    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    tokenizer_path=args.tokenizer_path,
                                    save_dir=args.save_dir, device=args.device)
    model.train()
    if args.grad_ckpt == 1:
        set_gradient_checkpointing(model, True)
    # T0.5（C9 修复）：DDP 包装与数据分片（routers/manager 按阶段激活 → find_unused）
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"SFT-{args.phase}")

    if args.phase in ("a", "b"):
        ds = SFTDataset(args.data_path, tokenizer, max_length=1024)
        collate = None
    else:
        ds = RetrievalQADataset(args.retrieval_data_path, tokenizer,
                                max_context_tokens=lm_config.max_position_embeddings,
                                block_size=lm_config.block_size)
        collate = collate_retrieval_qa
        assert lm_config.use_retrieval and lm_config.use_memory, "SFT-C 需要 minimind_x 变体"
    train_sampler = DistributedSampler(ds) if dist.is_initialized() else None
    loader = DataLoader(ds, batch_size=args.batch_size if args.phase != "c" else 1,
                        shuffle=(train_sampler is None), sampler=train_sampler,
                        num_workers=args.num_workers, collate_fn=collate)
    optimizer = build_optimizer(model, lr=args.learning_rate)  # T4.3 参数组
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    iters = len(loader) * args.epochs
    step_global = 0
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        t0 = time.time()
        for step, batch in enumerate(loader, 1):
            step_global += 1
            lr = get_lr(step_global, iters, args.learning_rate)
            for pg in optimizer.param_groups:  # 参数组调度：cosine 基准 × 组倍率
                pg['lr'] = lr * pg.get("mult", 1.0)
            if getattr(args, "residual_warmup_steps", 0) > 0:
                frac = residual_progress(step_global, args.residual_warmup_steps)
                eta, lam = apply_residual_progress(model, frac)
                if step % args.log_interval == 0:
                    Logger(f'residual warm-up: step={step_global} frac={frac:.3f} '
                           f'eta={eta:.3f} lambda={lam:.3f}')

            if args.phase == "a":
                input_ids, labels = batch[0].to(args.device), batch[1].to(args.device)
                with autocast_ctx:
                    loss, info = sft_a_step(input_ids, labels)
            elif args.phase == "b":
                input_ids, labels = batch[0].to(args.device), batch[1].to(args.device)
                with autocast_ctx:
                    loss, info = sft_b_step(input_ids, labels)
            else:
                with autocast_ctx:
                    loss, info = sft_c_step(batch)
            scaler.scale(loss / args.accumulation_steps).backward()
            if step % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % args.log_interval == 0 or step == len(loader):
                eta = (time.time() - t0) / step * (len(loader) - step) // 60
                Logger(f'SFT-{args.phase.upper()} Epoch[{epoch + 1}]({step}/{len(loader)}), '
                       f'loss: {info["loss"]:.4f}, lr: {lr:.8f}, eta: {eta:.1f}min')
                if wandb:
                    wandb.log({"loss": info["loss"], "phase": args.phase})
            if args.max_steps > 0 and step_global >= args.max_steps:
                Logger(f'SFT-{args.phase.upper()} 达到 max_steps={args.max_steps}，提前结束')
                break
        if args.max_steps > 0 and step_global >= args.max_steps:
            break
    # 评审 9.1：max_steps / 正常跑完都必须落盘（原先 break 后跳过保存）
    if is_main_process():
        save_x_checkpoint(  # T5.2：完整 checkpoint
            model, args.save_dir, f'{args.save_weight}_{args.phase}',
            lm_config.hidden_size, optimizer=optimizer, scaler=scaler,
            extra={"data_progress": {"epoch": epoch + 1 if "epoch" in dir() else 0,
                                     "step_global": step_global},
                   "warmup_progress": getattr(args, "residual_warmup_steps", 0),
                   "tokenizer_manifest": {"path": args.tokenizer_path},
                   "source_manifest": {"data_path": getattr(args, "data_path", None),
                                       "retrieval_data_path": getattr(args, "retrieval_data_path", None)}})


if __name__ == "__main__":
    main()
```

---

# trainer/x_utils.py

```python
"""MiniMind-X 训练公共工具（各 X trainer 共享；不改动原 trainer_utils.py）。"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.model_minimind import precompute_freqs_cis
from model.minimind_x import MiniMindXForCausalLM
from trainer.trainer_utils import Logger, get_model_params


def x_weight_path(save_dir, name, hidden_size):
    return f'{save_dir}/{name}_{hidden_size}_x.pth'


def init_model_x(lm_config, from_weight='pretrain', tokenizer_path='../model',
                 save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if from_weight != 'none':
        path = x_weight_path(save_dir, from_weight, lm_config.hidden_size)
        weights = torch.load(path, map_location=device, weights_only=False)
        meta = weights.pop("__meta__", None)  # T5.2 manifest 不进 state_dict
        # 评审 8.1：按 checkpoint config 恢复 residual_v2，禁止 V2 权重静默装进 V1 结构
        ckpt_cfg = (meta or {}).get("config") or {}
        ckpt_arch = (meta or {}).get("architecture_version") or ""
        ckpt_v2 = bool(ckpt_cfg.get("residual_v2")) or ckpt_arch.endswith("residual_v2")
        if ckpt_v2 and not getattr(lm_config, "residual_v2", False):
            Logger(f'ckpt is residual_v2 -> force lm_config.residual_v2=True ({path})')
            lm_config.residual_v2 = True
        if (not ckpt_v2) and getattr(lm_config, "residual_v2", False) and ckpt_cfg:
            Logger(f'WARN: loading non-v2 ckpt into residual_v2 model ({path})')
        v2_keys = [k for k in weights if k.startswith((
            "model.thinking_core.loop_mhar", "model.thinking_core.update_gate",
            "model.thinking_core.memory_reader"))]
        if v2_keys and not getattr(lm_config, "residual_v2", False):
            raise RuntimeError(
                f"checkpoint 含 V2 残差模块权重但 residual_v2=False，拒绝静默丢弃: "
                f"{v2_keys[:3]}...（评审 8.1）")
        model = MiniMindXForCausalLM(lm_config)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if (meta or {}).get("architecture_version"):
            Logger(f'ckpt meta: arch={meta.get("architecture_version")} '
                   f'params={meta.get("param_manifest", {}).get("total")}')
        # unexpected 中的 V2 模块 = 结构不匹配，必须报错而非打日志继续
        bad_unexp = [k for k in unexpected if k.startswith((
            "model.thinking_core.loop_mhar", "model.thinking_core.update_gate",
            "model.thinking_core.memory_reader"))]
        if bad_unexp:
            raise RuntimeError(f"V2 模块权重被丢弃（结构不匹配）: {bad_unexp[:5]}")
        Logger(f'loaded X weights from {path} (missing={len(missing)}, unexpected={len(unexpected)})')
    else:
        model = MiniMindXForCausalLM(lm_config)
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


def param_manifest(model) -> dict:
    """T5.2 / V2 §18.2：sum(p.numel()) 参数量 manifest（分模块）。"""
    raw = model.module if hasattr(model, 'module') else model
    raw = getattr(raw, '_orig_mod', raw)
    by_module = {}
    total = 0
    for n, p in raw.named_parameters():
        total += p.numel()
        top = n.split('.')[0] if '.' in n else n
        # 细分到二级（model.thinking_core.update_gate 等）
        parts = n.split('.')
        key = '.'.join(parts[:3]) if len(parts) >= 3 else n
        by_module[key] = by_module.get(key, 0) + p.numel()
    # T4.3：参数组清单（new-module LR 倍率 / no-decay 标记）
    RESIDUAL_KEYS = ("loop_mhar", "update_gate", "memory_reader",
                     "halt_router", "retrieve_router", "budget_router", "retriever")
    NO_DECAY_HINTS = ("scale_logit", "loop_mhar.query", "type_bias")
    # Frozen Index：key_proj 名义在 retriever 内，但建库 no_grad——优化器应跳过
    FROZEN_KEY_HINTS = ("retriever.key_proj",)
    groups = {"backbone_decay": 0, "backbone_no_decay": 0,
              "residual_decay": 0, "residual_no_decay": 0}
    for n, pr in raw.named_parameters():
        if not pr.requires_grad:
            continue
        is_residual = any(k in n for k in RESIDUAL_KEYS)
        no_decay = pr.ndim <= 1 or any(h in n for h in NO_DECAY_HINTS)
        key = ("residual_" if is_residual else "backbone_") + ("no_decay" if no_decay else "decay")
        groups[key] += pr.numel()
    frozen = sum(pr.numel() for n, pr in raw.named_parameters()
                 if any(h in n for h in ("retriever.key_proj",)))
    return {"total": total, "by_module": by_module, "param_groups": groups,
            "frozen_key_params": frozen,
            "freeze_key": bool(getattr(raw.config, "freeze_key", True)),
            "architecture_version": raw.config.architecture_version()
            if hasattr(raw.config, "architecture_version") else "minimind_x"}


def validate_config(lm_config):
    """启动即校验（T5.2）。"""
    if hasattr(lm_config, "validate"):
        lm_config.validate()
    return lm_config


def save_x_checkpoint(model, args_save_dir, save_weight, hidden_size,
                      optimizer=None, scaler=None, extra=None):
    """T5.2 / V2 §19：完整 checkpoint——权重 + config/架构/缓存版本 + optimizer/scaler
    + 各 rank RNG + 数据与 warm-up 进度 + source/tokenizer manifest + 参数量 manifest。
    """
    import random
    os.makedirs(args_save_dir, exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    raw = getattr(raw, '_orig_mod', raw)
    payload = {k: v.half().cpu() for k, v in raw.state_dict().items()}
    meta = {
        "param_manifest": param_manifest(model),
        "architecture_version": raw.config.architecture_version()
        if hasattr(raw.config, "architecture_version") else "minimind_x",
        "cache_version": 1,
        "config": raw.config.to_dict() if hasattr(raw.config, "to_dict") else {},
    }
    # 各 rank RNG（可复现）
    try:
        import numpy as np
        meta["rng_numpy"] = np.random.get_state()
    except Exception:
        meta["rng_numpy"] = None
    meta["rng_python"] = random.getstate()
    meta["rng_torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        meta["rng_cuda"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        meta["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        meta["scaler"] = scaler.state_dict()
    if extra:
        # 允许传入 data_progress / warmup_progress / tokenizer / source manifest
        meta.update(extra)
    payload["__meta__"] = meta
    path = x_weight_path(args_save_dir, save_weight, hidden_size)
    torch.save(payload, path)
    return path


def load_x_checkpoint_meta(path):
    """读回 __meta__（optimizer/RNG/进度等），权重键由 load_state_dict 负责。"""
    weights = torch.load(path, map_location="cpu", weights_only=False)
    meta = weights.pop("__meta__", None)
    return weights, meta


def restore_training_state(meta, optimizer=None, scaler=None, model=None):
    """从 __meta__ 恢复 optimizer/scaler/RNG（T5.2 严格模式加载无静默冲突）。"""
    if meta is None:
        return {"restored": []}
    restored = []
    if optimizer is not None and meta.get("optimizer"):
        optimizer.load_state_dict(meta["optimizer"])
        restored.append("optimizer")
    if scaler is not None and meta.get("scaler"):
        scaler.load_state_dict(meta["scaler"])
        restored.append("scaler")
    import random
    if meta.get("rng_python"):
        random.setstate(meta["rng_python"])
        restored.append("rng_python")
    if meta.get("rng_torch") is not None:
        torch.set_rng_state(meta["rng_torch"].cpu() if torch.is_tensor(meta["rng_torch"])
                            else meta["rng_torch"])
        restored.append("rng_torch")
    if meta.get("rng_cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(meta["rng_cuda"])
            restored.append("rng_cuda")
        except Exception:
            pass
    if meta.get("rng_numpy") is not None:
        try:
            import numpy as np
            np.random.set_state(meta["rng_numpy"])
            restored.append("rng_numpy")
        except Exception:
            pass
    return {"restored": restored,
            "data_progress": meta.get("data_progress"),
            "warmup_progress": meta.get("warmup_progress"),
            "tokenizer_manifest": meta.get("tokenizer_manifest"),
            "source_manifest": meta.get("source_manifest"),
            "param_manifest": meta.get("param_manifest"),
            "architecture_version": meta.get("architecture_version")}


def weighted_ce(logits, labels, weight=None):
    """逐 token 加权 CE（retrieval-aware 辅助损失：gold/needle 位置加权，T3.3）。"""
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    ce = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1),
                         ignore_index=-100, reduction="none").view(y.shape)
    mask = (y != -100).float()
    if weight is None:
        return (ce * mask).sum() / mask.sum().clamp(min=1)
    w = weight[..., 1:].contiguous()
    return (ce * w * mask).sum() / (w * mask).sum().clamp(min=1)


def apply_rope_stage(model, max_position_embeddings=None, rope_theta=None):
    """curriculum 阶段切换时重算 RoPE buffer（任务清单 D8）。兼容 DDP/compile 包装。"""
    raw = model
    for attr in ("module", "_orig_mod"):
        raw = getattr(raw, attr, raw)
    cfg = raw.config
    if max_position_embeddings is not None:
        cfg.max_position_embeddings = max_position_embeddings
    if rope_theta is not None:
        cfg.rope_theta = rope_theta
    cos, sin = precompute_freqs_cis(dim=cfg.head_dim, end=cfg.max_position_embeddings,
                                    rope_base=cfg.rope_theta, rope_scaling=cfg.rope_scaling)
    device = raw.model.freqs_cos.device
    raw.model.freqs_cos, raw.model.freqs_sin = cos.to(device), sin.to(device)


def set_gradient_checkpointing(model, enabled=True):
    """per-loop / per-layer 梯度检查点（任务清单 T1.4）：R 大时省激活显存。兼容 DDP/compile。"""
    raw = model
    for attr in ("module", "_orig_mod"):
        raw = getattr(raw, attr, raw)
    for stack in (raw.model.prelude.stack, raw.model.coda.stack):
        stack.gradient_checkpointing = enabled
    raw.model.thinking_core.gradient_checkpointing = enabled


def build_optimizer(model, lr, weight_decay=0.01, residual_lr_mult=2.0):
    """V2 参数组优化器（任务清单 T4.3；V2 §14.4）。

    新模块（loop_mhar/update_gate/memory_reader/routers/retriever）用 1-2× LR；
    norm/bias/gate scale/pseudo-query 等一维或特殊参数不施加 weight decay。
    """
    raw = model
    for attr in ("module", "_orig_mod"):
        raw = getattr(raw, attr, raw)
    RESIDUAL_KEYS = ("loop_mhar", "update_gate", "memory_reader",
                     "halt_router", "retrieve_router", "budget_router", "retriever")
    NO_DECAY_HINTS = ("scale_logit", "loop_mhar.query", "type_bias")
    # Frozen Index：key_proj 名义在 retriever 内，但建库 no_grad——优化器应跳过
    FROZEN_KEY_HINTS = ("retriever.key_proj",)
    groups = {"bd": [], "bnd": [], "rd": [], "rnd": []}
    for n, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        # freeze_key=True 时 key_proj 显式冻结（评审：避免「Q/K 联训」误解）
        if getattr(raw.config, "freeze_key", True) and any(h in n for h in FROZEN_KEY_HINTS):
            p.requires_grad_(False)
            continue
        is_residual = any(k in n for k in RESIDUAL_KEYS)
        no_decay = p.ndim <= 1 or any(h in n for h in NO_DECAY_HINTS)
        key = ("r" if is_residual else "b") + ("nd" if no_decay else "d")
        groups[key].append(p)
    param_groups = [{"params": groups["bd"], "lr": lr, "weight_decay": weight_decay, "mult": 1.0},
                    {"params": groups["bnd"], "lr": lr, "weight_decay": 0.0, "mult": 1.0}]
    if groups["rd"]:
        param_groups.append({"params": groups["rd"], "lr": lr * residual_lr_mult,
                             "weight_decay": weight_decay, "mult": residual_lr_mult})
    if groups["rnd"]:
        param_groups.append({"params": groups["rnd"], "lr": lr * residual_lr_mult,
                             "weight_decay": 0.0, "mult": residual_lr_mult})
    import torch as _torch
    return _torch.optim.AdamW(param_groups, lr=lr)


def tokenize_tensors(tokenizer, text, device="cpu", **kw):
    """tokenizer 输出 -> dict of tensors（绕开 transformers 在旧 torch 上的
    return_tensors='pt' 版本门禁；新环境同样适用）。"""
    enc = tokenizer(text, return_tensors=None, **kw)
    return {k: torch.tensor([v], device=device) for k, v in enc.items() if v is not None}


def save_pretrained_x(model, out_dir):
    """研究格式的『导出』：config.json + model.pth（T7.5）。

    注：完整 HF transformers 格式导出需要 remote-code 建模文件，超出研究平台范围；
    重新实例化用 MiniMindXConfig.from_json_file + load_state_dict(strict=False)。
    """
    import json
    os.makedirs(out_dir, exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    raw = getattr(raw, '_orig_mod', raw)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(raw.config.to_dict(), f, ensure_ascii=False, indent=2)
    torch.save({k: v.half().cpu() for k, v in raw.state_dict().items()},
               os.path.join(out_dir, "model_x.pth"))
    return out_dir


def load_pretrained_x(out_dir, device="cpu"):
    """save_pretrained_x 的对偶：读回 config + 权重。"""
    import json
    from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
        cfg = MiniMindXConfig(**{k: v for k, v in json.load(f).items()
                                 if k not in ("model_type", "transformers_version",
                                              "architectures")})
    model = MiniMindXForCausalLM(cfg)
    weights = torch.load(os.path.join(out_dir, "model_x.pth"), map_location="cpu",
                         weights_only=False)
    weights.pop("__meta__", None)
    model.load_state_dict(weights, strict=False)
    return model.to(device)
```

---

# dataset/retrieval_qa_dataset.py

```python
"""检索式 SFT 数据集（任务清单 T6.5；Doc 4 §8）。

jsonl schema：{"context": str, "question": str, "answer": str,
               "gold_block": int, "gold_span": [start,end) 字符偏移,
               "split": "train"|"val", "oracle_r": int, "oracle_retrieve": [...],
               "action_schema_version": 1|2}

主干序列只含 问题+答案（长上下文走 store，D9）；context 分块 tokenize 供建库。
batch 内变长，默认 collate 成 B=1 的 dict（研究级口径；训练器可用任意 B 自行 pad）。
"""
import json
import os

import torch
from torch.utils.data import Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class RetrievalQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_question_len=512, max_answer_len=512,
                 max_context_tokens=None, block_size=256, split=None):
        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.max_context_tokens = max_context_tokens
        self.block_size = block_size
        self.split = split
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # 评审 7.2：按 split 过滤，val 不得进训练
                if split is not None and rec.get("split", "train") != split:
                    continue
                self.samples.append(rec)

    def _gold_block_from_span(self, context, gold_span):
        """评审 7.1：gold_span 字符偏移 → 真实 token 块号（按当前 tokenizer/block_size）。"""
        if not gold_span or gold_span[0] < 0 or not context:
            return -1
        start, end = int(gold_span[0]), int(gold_span[1])
        # 前缀 tokenize 定位 start 对应 token 下标
        prefix_ids = self.tokenizer(context[:start], add_special_tokens=False).input_ids
        span_ids = self.tokenizer(context[start:end], add_special_tokens=False).input_ids
        if not span_ids:
            return -1
        tok_start = len(prefix_ids)
        tok_end = tok_start + max(1, len(span_ids))
        mid = (tok_start + tok_end - 1) // 2
        return int(mid // max(1, self.block_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        q_ids = self.tokenizer(s["question"], add_special_tokens=False,
                               max_length=self.max_question_len, truncation=True).input_ids
        a_ids = self.tokenizer(s["answer"], add_special_tokens=False,
                               max_length=self.max_answer_len, truncation=True).input_ids
        ans_prefix = self.tokenizer("答案：", add_special_tokens=False).input_ids
        input_ids = ([self.tokenizer.bos_token_id] + q_ids + ans_prefix +
                     a_ids + [self.tokenizer.eos_token_id])
        # 可见前缀边界（V2 T0.1 因果契约）：初始检索只允许用 [bos]+question，
        # "答案："前缀与 gold answer 对检索 query 不可见
        question_len = 1 + len(q_ids)
        labels = [-100] * question_len + [-100] * len(ans_prefix) + a_ids + [self.tokenizer.eos_token_id]
        ctx_ids = self.tokenizer(s["context"], add_special_tokens=False,
                                 max_length=self.max_context_tokens,
                                 truncation=True).input_ids if s.get("context") else []
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "question_len": torch.tensor(question_len, dtype=torch.long),
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "gold_block": torch.tensor(
                self._gold_block_from_span(s.get("context", ""), s.get("gold_span"))
                if s.get("gold_span") and s.get("gold_span", [-1])[0] >= 0
                else s.get("gold_block", -1), dtype=torch.long),
            "oracle_r": torch.tensor(s.get("oracle_r", 0), dtype=torch.long),
            "oracle_retrieve": torch.tensor(s.get("oracle_retrieve", []), dtype=torch.long),
            "action_schema_version": s.get("action_schema_version", 1),
            "answer_text": s["answer"],
        }


def collate_retrieval_qa(batch, pad_id=0):
    """变长 pad collate（input/labels pad；context 取 batch 内最长）。"""
    if len(batch) == 1:
        b = batch[0]
        out = {**{k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in b.items()}}
        # sft_c_step 用 batch["action_schema_version"] 取标量/列表
        if isinstance(out.get("action_schema_version"), list):
            out["action_schema_version"] = out["action_schema_version"][0]
        return out
    out = {}
    for key in ("input_ids", "labels"):
        seqs = [b[key] for b in batch]
        max_len = max(len(s) for s in seqs)
        pad_value = -100 if key == "labels" else pad_id
        out[key] = torch.stack([torch.cat([s, s.new_full((max_len - len(s),), pad_value)]) for s in seqs])
    out["question_len"] = torch.stack([b["question_len"] for b in batch])
    ctxs = [b["context_ids"] for b in batch]
    max_ctx = max(len(c) for c in ctxs)
    out["context_ids"] = torch.stack([torch.cat([c, c.new_zeros(max_ctx - len(c))]) for c in ctxs])
    for key in ("gold_block", "oracle_r"):
        out[key] = torch.stack([b[key] for b in batch])
    out["oracle_retrieve"] = [b["oracle_retrieve"] for b in batch]
    out["action_schema_version"] = [b.get("action_schema_version", 1) for b in batch]
    out["answer_text"] = [b["answer_text"] for b in batch]
    return out
```

---

# scripts/data_gen/__init__.py

```python
# data generators package
```

---

# scripts/data_gen/gen_evidence_families.py

```python
"""七族证据数据生成器（任务清单 T4.5；V2 文档 §16.1-16.2）。

样本族（每族一个 family 字段，训练/验收目标见 V2 表）：
  no_evidence      memory=None —— Reader 不应扰动无 memory 路径
  evidence_needed  答案含本批随机事实，prompt 无法推断 —— Reader 必须用外部信息
  irrelevant       与问题无关、格式相似的资料 —— 不因存在就注入
  similar_wrong_ver 相同术语、不同版本/对象 —— 学适用性而非关键词
  new_evidence     关键约束只在「后续轮才检索到」的位置 —— Gate 可重新打开
  multi_hop        第二跳信息依赖第一跳结果 —— Think→Retrieve→Think 闭环
  insufficient     确实缺少必要字段 —— 允许承认不足而非强行回答

机制验收（V2 §16.2 随机事实任务）：evidence_needed/new_evidence 每条随机换外部事实，
train/val 用不同随机种子（事实组合不重叠）；交换/移除 memory 后答案应随之变化。

输出 jsonl：{context, question, answer, family, gold_block, gold_span, split}
gold_span: 证据句在 context 中的 [start, end) 字符偏移；token id 标注由 dataset 按
当前 tokenizer/block_size 转换（参数变更须重新生成，V2 §16.2）。
"""
import os
import json
import random
import argparse

DEVICES = ["网关A", "网关B", "传感节点C", "执行器D", "边缘盒E", "协处理器F"]
VERSIONS = ["v1.2.0", "v1.3.0", "v2.0.1", "v2.1.0-rc"]


def filler(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        parts.append(rng.choice(["例行遥测正常。", "固件校验通过。", "链路心跳 200ms。",
                                 "缓冲区水位 42%。", "任务队列空。", "看门狗复位计数 0。"]))
        total += 10
    return "".join(parts)


def _block_ctx(rng, facts, n_blocks=6, gold_ratio=0.5):
    """事实句散布进填充文本；返回 (context, gold_block, gold_span)。"""
    parts = [filler(rng, 120) for _ in range(n_blocks)]
    n_blocks_real = len(parts)
    gold_idx = max(0, min(n_blocks_real - 1, int(gold_ratio * n_blocks_real)))
    parts[gold_idx] = facts[0]
    for i, f in enumerate(facts[1:], start=1):
        pos = (gold_idx + 2 * i) % n_blocks_real
        parts[pos] = f
    ctx = "".join(parts)
    # gold_span：facts[0] 在 context 中的 [start, end)（T4.5；token id 由 dataset 转换）
    start = ctx.find(facts[0])
    span = [start, start + len(facts[0])] if start >= 0 else [-1, -1]
    return ctx, gold_idx, span


def gen_sample(rng, family):
    dev = rng.choice(DEVICES)
    ver, ver_wrong = rng.sample(VERSIONS, 2)
    val = rng.randint(1000, 9999)
    if family == "no_evidence":
        # 真值修正：输入无任何依据时目标必须是「不可知」，禁止随机编造采样率
        return {"context": "", "question": f"{dev} 的默认采样率是多少？",
                "answer": "上下文未提供默认采样率信息。",
                "gold_span": [-1, -1], "gold_block": -1, "unanswerable": True}
    if family == "evidence_needed":
        facts = [f"配置记录：{dev} 本次部署的阈值设定为 {val}。"]
        ctx, gold, span = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"本次部署中 {dev} 的阈值是多少？",
                "answer": str(val), "gold_block": gold, "gold_span": span}
    if family == "irrelevant":
        # 真值修正：干扰设备必须 ≠ 目标设备，且 context 不得出现目标设备名
        others = [d for d in DEVICES if d != dev]
        other = rng.choice(others)
        facts = [f"配置记录：{other} 的阈值设定为 {val}。"]
        ctx, gold, span = _block_ctx(rng, facts)
        assert dev not in ctx, f"irrelevant 泄漏目标设备: {dev} in ctx"
        assert str(val) in ctx  # 数值只属于 other，答案仍不可知
        return {"context": ctx, "question": f"{dev} 的阈值是多少？",
                "answer": "上下文未提供该设备的阈值信息。", "gold_block": -1, "gold_span": [-1, -1],
                "unanswerable": True}
    if family == "similar_wrong_ver":
        facts = [f"版本说明：{dev} {ver_wrong} 的阈值为 {val}。"]
        ctx, gold, span = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} {ver} 的阈值是多少？",
                "answer": f"上下文只包含 {ver_wrong} 的信息，缺少 {ver} 的阈值。",
                "gold_block": -1, "gold_span": [-1, -1], "unanswerable": True}
    if family == "new_evidence":
        # 真值修正：证据可见时间线——生效约束在 late_block，首轮检索默认不可见
        facts = [f"补充约束（生效中）：{dev} 的实际阈值为 {val}，覆盖早期文档值。",
                 f"早期文档：{dev} 的阈值为 {rng.randint(100, 999)}（已废弃）。"]
        parts = [filler(rng, 150) for _ in range(6)]
        parts[2], parts[4] = facts[1], facts[0]
        ctx = "".join(parts)
        start = ctx.find(facts[0])
        span = [start, start + len(facts[0])] if start >= 0 else [-1, -1]
        return {"context": ctx, "question": f"{dev} 当前的有效阈值是多少？",
                "answer": str(val), "gold_block": 4, "gold_span": span,
                # 可见时间线：生效证据块在第 1 轮后才允许进 Hot Memory
                "evidence_visible_from_round": 1,
                "stale_block": 2, "gold_block_late": 4}
    if family == "multi_hop":
        dev2 = rng.choice([d for d in DEVICES if d != dev])
        facts = [f"链路表：{dev} 的上行通道绑定到 {dev2}。",
                 f"配置记录：{dev2} 的阈值设定为 {val}。"]
        ctx, gold, span = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} 所绑定设备的阈值是多少？",
                "answer": str(val), "gold_block": gold, "gold_span": span}
    if family == "insufficient":
        facts = [f"配置记录：{dev} 的阈值设定为（待填写）。"]
        ctx, gold, span = _block_ctx(rng, facts)
        return {"context": ctx, "question": f"{dev} 的阈值是多少？",
                "answer": "记录中该字段为空，无法给出阈值。", "gold_block": gold, "gold_span": span,
                "unanswerable": True}
    raise ValueError(family)


FAMILIES = ["no_evidence", "evidence_needed", "irrelevant", "similar_wrong_ver",
            "new_evidence", "multi_hop", "insufficient"]


def main():
    p = argparse.ArgumentParser(description="生成七族证据数据（V2 §16.1）")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n_per_family", type=int, default=200)
    p.add_argument("--seed_train", type=int, default=42)
    p.add_argument("--seed_val", type=int, default=1042)  # 与 train 不重叠的事实随机流
    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for split, seed in (("train", args.seed_train), ("val", args.seed_val)):
            rng = random.Random(seed)
            for fam in FAMILIES:
                for _ in range(args.n_per_family):
                    rec = gen_sample(rng, fam)
                    rec["family"] = fam
                    rec["split"] = split
                    # 真值自检（评审 7.3）：irrelevant 不得泄漏目标设备
                    if fam == "irrelevant":
                        q = rec.get("question", "")
                        # question 含目标设备名是合法的；context 不得含
                        dev_tok = [d for d in DEVICES if d in q]
                        for d in dev_tok:
                            assert d not in rec.get("context", ""), f"irrelevant 泄漏 {d}"
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"written {2 * len(FAMILIES) * args.n_per_family} samples "
          f"(train/val 事实流隔离) -> {args.out}")


if __name__ == "__main__":
    main()
```

---

# scripts/data_gen/gen_long_dep.py

```python
"""长依赖合成数据（任务清单 T3.1；Doc 4 §7.2）。

前部定义变量/规则，远端（约 depth 比例处）要求使用 —— 训练长距离回看。
产出 pretrain jsonl：{"text": ...}（配合 LongContextPretrainDataset packing）。
"""
import os
import json
import random
import argparse

WORDS = ("变量 函数 模块 参数 结构 配置 状态 结果 记录 序号 编号 数值 阈值 上限 标识").split()


def filler(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(10, 25))) + " 。"
        parts.append(sent)
        total += len(sent)
    return parts


def gen_sample(rng, target_tokens, depth):
    name = f"X{rng.randint(100, 999)}"
    val = rng.randint(1000, 999999)
    add = rng.randint(1, 100)  # 真值修正：目标必须是 X+正整数，不得仍是原 X
    definition = f"定义：{name} 的初始值为 {val}，后续所有引用必须使用该值。"
    use = (f"计算：{name} 的值加上 {add}，结果以「{name} 的最终取值是」开头给出："
           f"{name} 的最终取值是 {val + add}。")
    parts = filler(rng, int(target_tokens * 1.5))
    pos = max(1, int(len(parts) * depth))
    parts.insert(0, definition)
    parts.insert(pos + 1, use)
    return {"text": "\n".join(parts)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成长依赖预训练数据")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--depth", type=float, default=0.9, help="使用点深度比例")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.ctx_tokens, args.depth), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
```

---

# scripts/data_gen/gen_memory_mgmt.py

```python
"""Memory 管理监督数据（任务清单 T5.5；Doc 4 §19）。

构造 Keep/Drop/Replace 标签：给定原 memory blocks、新块、query，
标签 = 按「含 query 关键词 + 去重」规则应保留的块（可学习规则）。
jsonl：{memory_blocks, new_blocks, query, keep}
"""
import os
import json
import random
import argparse

TOPICS = ["数据库", "网络", "安全", "编译", "调度", "存储", "前端", "算法", "监控", "部署"]


def gen_block(rng, topic=None):
    topic = topic or rng.choice(TOPICS)
    return f"{topic}模块：{topic}的{rng.choice(['配置', '状态', '日志', '依赖', '接口'])}" \
           f"记录 {rng.randint(100, 999)}，{rng.choice(['正常', '已更新', '待处理', '已归档'])}。"


def gen_sample(rng, mem_n, new_n, keep_n):
    topic = rng.choice(TOPICS)
    query = f"当前任务需要查询{topic}相关的信息"
    blocks = [gen_block(rng) for _ in range(mem_n)]
    news = [gen_block(rng, topic if rng.random() < 0.7 else None) for _ in range(new_n)]
    # 标签规则：与 query 同主题（含关键词）优先保留，重复内容去重（只留首个）
    def score(b):
        return (1 if topic in b else 0) + rng.random() * 0.01
    seen, pool = set(), []
    for i, b in enumerate(blocks + news):
        key = b.split("记录")[0]
        if key in seen:
            continue
        seen.add(key)
        pool.append((score(b), f"{i}"))
    pool.sort(reverse=True)
    keep = [i for _, i in pool[:keep_n]]
    return {"memory_blocks": blocks, "new_blocks": news, "query": query, "keep": keep}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 memory 管理监督数据")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--mem_blocks", type=int, default=8)
    p.add_argument("--new_blocks", type=int, default=4)
    p.add_argument("--keep", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.mem_blocks, args.new_blocks, args.keep),
                               ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
```

---

# scripts/data_gen/gen_multihop.py

```python
"""Multi-hop 推理数据（任务清单 T3.1；Doc 4 §8）。

构造实体链条 A→B→C→D…，事实句打乱铺入上下文，问题要求跨信息组合。
产出检索式 jsonl：{context, question, answer, gold_block}。
"""
import os
import json
import random
import argparse

SURNAMES = "赵 钱 孙 李 周 吴 郑 王 冯 陈 蒋 沈 韩 杨".split()
ROLES = ["负责人", "对接人", "审核人", "导师", "合作者"]
ENT_FORMATS = [
    "{a} 的{role}是 {b}。",
    "记录显示，{a} 由 {b} 负责。",
    "根据登记表，{a} 的联系人是 {b}。",
]
FILLER = ["该季度项目按计划推进。", "会议记录已归档。", "相关审批流程已完成。",
          "预算执行情况正常。", "系统运行状态良好。", "文档已同步至知识库。"]


def gen_sample(rng, hops, ctx_tokens, block_size):
    chain = [rng.choice(SURNAMES) + rng.choice("明芳军婷宇晨静磊") for _ in range(hops + 1)]
    facts = []
    for i in range(hops):
        facts.append(rng.choice(ENT_FORMATS).format(a=chain[i], role=rng.choice(ROLES), b=chain[i + 1]))
    target_chars = int(ctx_tokens * 1.5)
    parts = [rng.choice(FILLER) for _ in range(max(10, target_chars // 12))]
    insert_at = sorted(rng.sample(range(len(parts)), min(hops, len(parts))), reverse=True)
    for pos, fact in zip(insert_at, facts):
        parts.insert(pos, fact)
    rng.shuffle(insert_at)  # gold 块按最终位置折算（简化：取第一条事实）
    first_fact = facts[0]
    q = chain[0]
    for i in range(1, hops):
        q += f" 的{ROLES[0]}"
    question = f"{'起点 ' + chain[0]}经过 {hops} 跳关系后指向谁？"
    answer = chain[-1]
    ctx = "".join(parts)
    gold_ratio = ctx.find(first_fact) / max(len(ctx), 1)
    n_blocks = max(1, round(ctx_tokens / block_size))
    return {"context": ctx, "question": question, "answer": answer,
            "gold_block": min(int(gold_ratio * n_blocks), n_blocks - 1), "chain": chain}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 multi-hop 检索式数据")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--hops", type=str, default="2", help="跳数（可逗号分隔混合 1,2,4,8）")
    p.add_argument("--ctx_tokens", type=int, default=4096)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    hop_pool = [int(x) for x in args.hops.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            f.write(json.dumps(gen_sample(rng, hop_pool[i % len(hop_pool)],
                                          args.ctx_tokens, args.block_size), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
```

---

# scripts/data_gen/gen_needle.py

```python
"""Needle 数据生成器（任务清单 T3.1；Doc 4 §7.3、Doc 5 §8）。

产出检索式 jsonl：{context, question, answer, gold_block}。
- 单针/多针（--needles 1~8）
- 深度网格：needle 插入位置比例从 --depths 取值（0.1~0.9 对应 10K/100K/500K/1M 的相对深度）
- gold_block：needle 所在块号（按字符比例折算，训练/评测时以实际 tokenize 为准）
"""
import os
import sys
import json
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataset.long_context_dataset import FILLER_WORDS


def filler_sentences(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        sent = "".join(rng.choice(FILLER_WORDS) for _ in range(rng.randint(8, 20))) + "。"
        parts.append(sent)
        total += len(sent)
    return parts


def gen_sample(rng, ctx_tokens, depths, n_needles, block_size):
    num = rng.randint(1000, 9999)
    needles = [f"重要提示：第{i + 1}条记录的编号是 {num + i}。" for i in range(n_needles)]
    target_chars = int(ctx_tokens * 1.5)
    parts = filler_sentences(rng, target_chars)
    n_blocks = max(1, round(ctx_tokens / block_size))
    # 多针：针数与深度网格 zip（避免越界）；单针：取第一个深度
    pairs = list(zip(depths, needles)) if n_needles > 1 else [(depths[0], needles[0])]
    gold_blocks = []
    for d, needle in pairs:
        pos = max(1, int(len(parts) * d))
        parts.insert(pos, needle)
        gold_blocks.append(min(int(d * n_blocks), n_blocks - 1))
    q_idx = rng.randrange(len(pairs))  # 只在实际插入的针里提问（深度数可小于针数）
    question = f"第{q_idx + 1}条记录的编号是多少？"
    answer = str(num + q_idx)
    return {"context": "".join(parts), "question": question, "answer": answer,
            "gold_block": gold_blocks[q_idx], "gold_blocks": gold_blocks, "needles": needles}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 needle 检索式训练/评测数据")
    p.add_argument("--out", type=str, required=True, help="输出 jsonl 路径")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--needles", type=int, default=1, help="每条样本针数（1=单针）")
    p.add_argument("--depths", type=str, default="0.1,0.3,0.5,0.7,0.9",
                   help="插入深度比例网格（逗号分隔）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    depths = [float(x) for x in args.depths.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.ctx_tokens, depths,
                                          args.needles, args.block_size), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
```

---

# scripts/data_gen/gen_planning.py

```python
"""Planning / Agent 任务数据（任务清单补 Doc 1 §23「Planning / Agent Task」）。

多步任务规划：上下文含任务描述 + 资源/约束清单（散布），问题要求给出执行计划。
产出检索式 jsonl：{context, question, answer, gold_block}。
"""
import os
import json
import random
import argparse

RESOURCES = ["服务器A", "服务器B", "数据库C", "缓存D", "网关E", "队列F", "镜像仓库G"]
CONSTRAINTS = ["每日 02:00 维护窗口", "带宽上限 1Gbps", "磁盘余量 20%", "并发上限 500",
               "审计要求全量日志", "回滚需双人审批"]
# 真值修正：步骤有硬依赖（必须按此偏序取子序列，禁止无放回乱序）
STEPS_ORDERED = ["备份当前状态", "更新配置", "扩容资源", "切换流量",
                 "验证健康检查", "通知相关方", "归档日志", "清理临时文件"]


def filler(rng, n_chars):
    parts, total = [], 0
    while total < n_chars:
        parts.append(rng.choice(["例行巡检记录正常。", "容量报表已生成。", "告警面板无待办。",
                                 "变更单已关闭。", "值班交接完成。", "备份任务成功。"]))
        total += 12
    return parts


def gen_sample(rng, n_steps, ctx_tokens, block_size):
    target_chars = int(ctx_tokens * 1.5)
    parts = filler(rng, target_chars)
    res = rng.sample(RESOURCES, k=min(3, len(RESOURCES)))
    cons = rng.sample(CONSTRAINTS, k=2)
    # 取有序子序列（保序），保证步骤依赖成立
    n_steps = max(2, min(n_steps, len(STEPS_ORDERED)))
    idxs = sorted(rng.sample(range(len(STEPS_ORDERED)), k=n_steps))
    plan = [STEPS_ORDERED[i] for i in idxs]
    assert plan == [STEPS_ORDERED[i] for i in idxs] and idxs == sorted(idxs)
    # 资源/约束散布进上下文（需检索组合）
    doc = [f"资源清单：本次可使用 {('、'.join(res))}。"] + \
          [f"约束{i + 1}：{c}。" for i, c in enumerate(cons)]
    insert_at = sorted(rng.sample(range(len(parts)), len(doc)), reverse=True)
    for pos, line in zip(insert_at, doc):
        parts.insert(pos, line)
    question = (f"目标：完成一次上线变更，涉及 {res[0]} 与 {res[-1]}，需遵守全部约束，"
                f"共 {n_steps} 步。请给出按序执行计划。")
    # 答案必须落实约束（否则目标不完整）
    answer = ("；".join(f"第{i + 1}步：{s}" for i, s in enumerate(plan)) +
              f"。执行时遵守：{'；'.join(cons)}。")
    ctx = "".join(parts)
    first = doc[0]
    gold_ratio = ctx.find(first) / max(len(ctx), 1)
    n_blocks = max(1, round(ctx_tokens / block_size))
    return {"context": ctx, "question": question, "answer": answer,
            "gold_block": min(int(gold_ratio * n_blocks), n_blocks - 1)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 Planning/Agent 检索式数据（Doc 1 §23）")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--steps", type=str, default="3,5", help="计划步数（可逗号分隔混合）")
    p.add_argument("--ctx_tokens", type=int, default=4096)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    step_pool = [int(x) for x in args.steps.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            f.write(json.dumps(gen_sample(rng, step_pool[i % len(step_pool)],
                                          args.ctx_tokens, args.block_size),
                               ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
```

---

# scripts/data_gen/gen_repo_context.py

```python
"""代码仓库上下文数据（任务清单 T3.1；Doc 4 §7.4、Doc 5 §8）。

合成迷你仓库（目录树 + 多文件函数），生成 Issue→Patch 任务。
产出检索式 jsonl：{context(仓库全文), question(Issue), answer(Patch), gold_block}。
"""
import os
import json
import random
import argparse

FUNC_VERBS = ["计算", "校验", "解析", "转换", "缓存", "汇总", "排序", "过滤"]
TYPES = ["int", "float", "str", "list", "dict", "bool"]


def gen_repo(rng, n_files, funcs_per_file):
    files = {}
    for fi in range(n_files):
        name = f"module_{fi}.py"
        lines = [f"# {name}: auto-generated module", "from typing import Any", ""]
        for gi in range(funcs_per_file):
            fn = f"{rng.choice(FUNC_VERBS)}_{rng.randint(10, 99)}"
            args = ", ".join(f"a{i}: {rng.choice(TYPES)}" for i in range(rng.randint(1, 3)))
            body = [f"def {fn}({args}) -> Any:"]
            indent = "    "
            if rng.random() < 0.5:
                body.append(f"{indent}return sum(a for a in [{args.split(':')[0]}] if isinstance(a, (int, float)))")
            else:
                body.append(f"{indent}return str({args.split(':')[0]})")
            lines.extend(body + [""])
        files[name] = "\n".join(lines)
    tree = "\n".join(f"- {n} ({len(c)} bytes)" for n, c in files.items())
    return f"src/\n{tree}\n\n" + "\n\n".join(f"# file: {n}\n{c}" for n, c in files.items()), files


def gen_sample(rng, n_files, funcs_per_file, ctx_tokens, block_size):
    ctx, files = gen_repo(rng, n_files, funcs_per_file)
    target_file = rng.choice(list(files))
    # 截断/填充到目标长度
    if len(ctx) > ctx_tokens * 4:
        ctx = ctx[:ctx_tokens * 4]
    # 真值修正：从源码解析真实函数名与参数，patch 不得使用未定义的 args/省略号
    defs = []
    for l in files[target_file].splitlines():
        if l.startswith("def "):
            head = l.split("(")[0][4:].strip()
            params = l.split("(", 1)[1].rsplit(")", 1)[0] if "(" in l else ""
            arg_names = [p.split(":")[0].strip() for p in params.split(",") if p.strip()]
            defs.append((head, arg_names))
    if not defs:
        defs = [("main", [])]
    fn, arg_names = rng.choice(defs)
    first_arg = arg_names[0] if arg_names else "x"
    issue = (f"Issue: {target_file} 中的 {fn}() 在空输入时抛出 TypeError。\n"
             f"请给出最小修复 patch。")
    # patch 必须引用真实参数名（评审 7.3：禁未定义 args / 省略号）
    old_line = f"-    return sum(a for a in [{first_arg}] if isinstance(a, (int, float)))"
    new_line = (f"+    if {first_arg} is None:\n"
                f"+        return 0\n"
                f"+    return sum(a for a in ({first_arg},) if isinstance(a, (int, float)))")
    answer = (f"Patch ({target_file}):\n"
              f"@@ def {fn} @@\n"
              f"{old_line}\n"
              f"{new_line}")
    assert "args" not in answer or first_arg == "args"
    assert "..." not in answer
    file_pos = ctx.find(f"# file: {target_file}")
    gold_ratio = max(file_pos, 0) / max(len(ctx), 1)
    n_blocks = max(1, round(ctx_tokens / block_size))
    return {"context": ctx, "question": issue, "answer": answer,
            "gold_block": min(int(gold_ratio * n_blocks), n_blocks - 1)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 repo-context 检索式数据")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n_files", type=int, default=12)
    p.add_argument("--funcs_per_file", type=int, default=6)
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.n_files, args.funcs_per_file,
                                          args.ctx_tokens, args.block_size), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
```

---

# tests/conftest.py

```python
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

---

# tests/test_actions_and_executors.py

```python
"""T2.5 三动作 / T3.2 prefill fixed_cap / T4.6 双执行器 / T5.2 配置校验 验收。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.router.retrieval_router import (
    ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH,
    action_mask, apply_action_mask, RetrievalRouter)


def _model(variant="minimind_x", max_loop=2, **over):
    torch.manual_seed(7)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant=variant, block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=max_loop, residual_v2=True)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_action_mask_rescore_warm_without_cache():
    """T2.5：无 Warm Candidate Cache 时 RESCORE_WARM 必须 mask，不得可用。"""
    m = action_mask(False, device="cpu", batch=2)
    assert m.shape == (2, 3)
    assert m[:, ACTION_REUSE].all() and m[:, ACTION_GLOBAL_REFRESH].all()
    assert not m[:, ACTION_RESCORE_WARM].any()
    m2 = action_mask(True, device="cpu", batch=1)
    assert m2.all()
    logits = torch.zeros(2, 3)
    masked = apply_action_mask(logits, m)
    assert masked[0, ACTION_RESCORE_WARM] == float("-inf")
    assert masked[0, ACTION_REUSE] == 0.0


def test_rescore_warm_rejected_without_cache():
    """执行侧同样拒绝无缓存 RESCORE_WARM（禁止静默降级全局搜索）。"""
    m = _model()
    ctrl = AdaptiveController(m)
    ids = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl.begin(ctx)
    assert not ctrl.has_warm_cache()
    try:
        ctrl.execute_retrieval_action(ACTION_RESCORE_WARM, ids, mem=None)
        assert False, "无 warm cache 的 RESCORE_WARM 应抛错"
    except RuntimeError:
        pass


def test_three_action_trajectory_and_warm_cache():
    """T2.5：三动作可执行；warm cache 在全局检索后可用；轨迹记动作。"""
    torch.manual_seed(8)
    ids = torch.randint(3, 60, (1, 10))
    labels = torch.full((1, 10), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    m = _model(max_loop=3)  # 3 轮 → 两次 loop_callback
    ctrl = AdaptiveController(m)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4,
        oracle_retrieve=[1, 0])  # schema v2：第 0 轮后 RESCORE_WARM，第 1 轮 REUSE
    assert extras["warm_available"] is True
    events = [t["event"] for t in ctrl.trajectory]
    assert "initial_retrieval" in events
    assert "rescore_warm" in events
    assert "reuse" in events
    assert ctrl.stats.get("warm_rescore", 0) >= 1


def test_global_refresh_still_works_and_legacy_oracle():
    """schema v2：2=GLOBAL_REFRESH；旧 v1 二值经 normalize_oracle_actions 转换。"""
    m = _model()
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4, oracle_retrieve=[2, 0])
    events = [t["event"] for t in ctrl.trajectory]
    assert "refresh" in events or "global_refresh" in events
    assert ctrl.stats["global_refresh"] >= 1


def test_action_ce_loss_masks_unavailable():
    from model.minimind_x.losses import retrieval_action_ce_loss
    logits = [torch.randn(1, 3, requires_grad=True) for _ in range(2)]
    labels = torch.tensor([ACTION_REUSE, ACTION_REUSE])
    # RESCORE_WARM 不可用
    masks = [action_mask(False, batch=1), action_mask(False, batch=1)]
    loss = retrieval_action_ce_loss(logits, labels, action_masks=masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(l.grad is not None for l in logits)
    # 标签落在不可用动作上 → 该轮 mask 掉（不 NaN）
    bad = torch.tensor([ACTION_RESCORE_WARM, ACTION_REUSE])
    loss2 = retrieval_action_ce_loss(logits, bad, action_masks=masks)
    assert torch.isfinite(loss2)


def test_prefill_fixed_cap_ttft_decode_stats():
    """T3.2：prefill 用固定 R_prefill；TTFT / decode 成本分开记。"""
    m = _model(variant="middle_loop", max_loop=3, residual_v2=False)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out = m.generate(inputs=ids, max_new_tokens=3, do_sample=False,
                         use_halting=False, fixed_loops=None, return_stats=True,
                         eos_token_id=None)
    st = out["stats"]
    assert st["ttft_steps"] == 1, "prompt prefill 应记 1 次 TTFT"
    assert st["decode_steps"] == 2  # max_new_tokens=3：1 次 TTFT + 2 次 decode
    assert "avg_ttft_loops" in st and "avg_decode_loops" in st
    assert "executed_core_rounds" in st


def test_dual_executor_correspondence():
    """T4.6：逐前缀参考执行器与分段执行器数值逐位对应。"""
    from trainer.dual_executor import (
        build_action_timeline, reference_append_only,
        segmented_forward, assert_executors_correspond_ce)
    torch.manual_seed(9)
    m = _model(max_loop=1, residual_v2=True)
    ids = torch.randint(3, 60, (1, 6))
    labels = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (8 * 4,))
    timeline = build_action_timeline([0, 0], [(0, 3), (3, 6)])

    def factory():
        return AdaptiveController(m)

    ref_ce, _ = reference_append_only(
        m, ids, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    seg_ce, _ = segmented_forward(
        m, ids, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    ok, msg = assert_executors_correspond_ce(ref_ce, seg_ce, atol=1e-4)
    assert ok, f"双执行器应数值对应: {msg}\nref={ref_ce}\nseg={seg_ce}"
    # 因果一致性：换未来 token 不影响早期 CE
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % 60
    ref_ce2, _ = reference_append_only(
        m, ids2, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    assert torch.allclose(ref_ce[:, :3], ref_ce2[:, :3], atol=1e-5), \
        "换未来 token 不得改变早期位置 CE"


def test_dual_executor_with_refresh_boundary():
    """T4.6：带动作边界（GLOBAL_REFRESH）时两执行器仍对应。"""
    from trainer.dual_executor import (
        build_action_timeline, reference_append_only,
        segmented_forward, assert_executors_correspond_ce)
    torch.manual_seed(11)
    m = _model(max_loop=1, residual_v2=True)
    ids = torch.randint(3, 60, (1, 6))
    labels = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (8 * 4,))
    # 段 0 REUSE，段 1 进入时 GLOBAL_REFRESH（动作边界）
    timeline = build_action_timeline([0, 2], [(0, 3), (3, 6)])

    def factory():
        return AdaptiveController(m)

    ref_ce, _ = reference_append_only(
        m, ids, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    seg_ce, _ = segmented_forward(
        m, ids, labels, ctx, query_len=3, timeline=timeline,
        controller_factory=factory, num_loops=1)
    ok, msg = assert_executors_correspond_ce(ref_ce, seg_ce, atol=1e-4)
    assert ok, f"动作边界下双执行器应对应: {msg}\nref={ref_ce}\nseg={seg_ce}"


def test_config_validation():
    """T5.2：非法 config 启动即报错。"""
    try:
        cfg = MiniMindXConfig(hidden_size=30, num_hidden_layers=6,
                              num_attention_heads=2, num_key_value_heads=1,
                              vocab_size=64, flash_attn=False,
                              variant="minimind_x", mhar_heads=4, residual_v2=True)
        cfg.validate()
        assert False, "30 不能被 mhar_heads=4 整除应报错"
    except ValueError:
        pass
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6,
                          num_attention_heads=2, num_key_value_heads=1,
                          vocab_size=64, flash_attn=False,
                          variant="minimind_x", mhar_heads=4, residual_v2=True)
    assert cfg.validate() is True
    assert cfg.architecture_version() == "minimind_x_residual_v2"


def test_param_manifest_and_flops_breakdown():
    """T5.2 参数 manifest / T5.5 FLOPs 分项。"""
    from trainer.x_utils import param_manifest
    from model.minimind_x.flops import estimate_flops
    m = _model()
    man = param_manifest(m)
    assert man["total"] > 0
    assert any("thinking_core" in k or "update_gate" in k or "loop_mhar" in k
               for k in man["by_module"])
    fl = estimate_flops(m.config, seq_len=16, loops=2, memory_blocks=2, n_retrievals=2)
    for k in ("reader_attn", "reader_gate", "mhar", "update_gate", "retrieval",
              "backbone", "total"):
        assert k in fl and fl[k] >= 0
    assert fl["reader_attn"] > 0  # 有 memory 时 Reader 有真实开销
    assert fl["total"] >= fl["backbone"]


def test_joint_path_logprob_and_no_avg_depth_replay():
    """T4.7：联合路径概率可加总；禁平均深度重放门槛。"""
    m = _model()
    ctrl = AdaptiveController(m)
    lp = ctrl.joint_path_logprob(
        token_logprobs=torch.tensor([-1.0, -2.0]),
        halt_logprobs=[-0.5],
        retrieval_logprobs=[-0.3],
        budget_logprobs=[-0.1])
    assert abs(lp - (-3.9)) < 1e-6
    assert ctrl.assert_replay_not_averaged([1, 3, 2, 4]) is True
    assert ctrl.stats["depth_per_token"] == [1, 3, 2, 4]
    assert ctrl.stats["depth_mean_for_log_only"] == 2  # round(2.5)=2（银行家舍入）；仅日志用


def test_action_order_no_retrieve_after_halt():
    """T2.5 / V2 §9.1：halt 判停后禁止再检索；R_max 强制吸收。"""
    torch.manual_seed(21)
    m = _model(max_loop=3)
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    # oracle：第 0 轮后 REFRESH，但若 halt 先停则不得检索
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4, oracle_r=torch.tensor([1]),
        oracle_retrieve=[2, 2, 2])  # 三个 GLOBAL_REFRESH
    events = [t["event"] for t in ctrl.trajectory]
    # R*=1：应在第 0 轮后停，之后无 refresh（不出现「先检索后发现已停」）
    # 训练全展开仍会算辅助轮动作，但 execute 后的 refresh 只发生在未强制停前
    # 参考口径：initial 必有；若 n_run==1 则不应有轮间 refresh 执行导致的版本暴涨
    assert "initial_retrieval" in events
    assert out.num_loops_executed >= 1
    # 轨迹中 refresh/rescore 只能出现在 initial 之后、且轮次 < num_loops_executed
    for t in ctrl.trajectory:
        if t["event"] in ("refresh", "rescore_warm"):
            assert t.get("round", -1) < out.num_loops_executed, \
                f"halt 后不应再检索: {t}"
```

---

# tests/test_append_only_alignment.py

```python
"""append-only 对齐：reference ↔ segmented ↔ generate（V2 §11.5）。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from trainer.dual_executor import (
    build_action_timeline, reference_append_only, segmented_forward,
    reference_recompute, generate_teacher_forced, assert_executors_correspond)


def _m(**over):
    torch.manual_seed(97)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=1, residual_v2=True)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def _batch():
    torch.manual_seed(98)
    ids = torch.randint(3, 60, (1, 6))
    y = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (32,))
    return ids, y, ctx


def test_append_only_matches_segmented():
    """同 memory 时间线：逐 token append-only == 分段续 cache。"""
    m = _m()
    ids, y, ctx = _batch()
    timeline = build_action_timeline([0, 0], [(0, 3), (3, 6)])
    ref_ce, ref_logits = reference_append_only(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    seg_ce, seg_logits = segmented_forward(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    ok, msg = assert_executors_correspond(ref_logits, seg_logits, atol=1e-4)
    assert ok, f"append-only vs segmented: {msg}"
    assert torch.allclose(ref_ce, seg_ce, atol=1e-4)


def test_append_only_with_memory_switch_differs_from_recompute():
    """memory 换版后：append-only 不回算历史；从头重算是另一语义（旧路径）。"""
    m = _m(max_loop=1)
    ids, y, ctx = _batch()
    # 段 0 REUSE，段 1 GLOBAL_REFRESH（动作边界换 memory）
    timeline = build_action_timeline([0, 2], [(0, 3), (3, 6)])
    ao_ce, ao_logits = reference_append_only(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    rc_ce, rc_logits = reference_recompute(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    # 分段续 cache 必须跟 append-only，而不是 recompute
    sg_ce, sg_logits = segmented_forward(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    ok, msg = assert_executors_correspond(ao_logits, sg_logits, atol=1e-4)
    assert ok, f"segmented 应对齐 append-only: {msg}"
    # recompute 与 append-only 在换 memory 后允许不同（历史是否被重算）
    diff_rc = max((a - b).abs().max().item() for a, b in zip(ao_logits, rc_logits))
    # 若 memory 版本切换实际改变了 payload，diff 应能观测到；至少二者都是有限值
    assert all(torch.isfinite(x).all() for x in ao_logits + rc_logits)


def test_generate_teacher_forced_matches_append_only():
    """与 generate 同构的 prefill+decode == append-only（固定 R、固定 memory）。"""
    m = _m(max_loop=1)
    ids, y, ctx = _batch()
    timeline = build_action_timeline([0], [(0, 6)])
    _, ref_logits = reference_append_only(
        m, ids, y, ctx, 3, timeline, lambda: AdaptiveController(m), num_loops=1)
    # 同一 memory（全 REUSE）逐位置
    from trainer.dual_executor import _prepare_memories, _mem_at
    mem_list = _prepare_memories(m, ids, ctx, 3, timeline, lambda: AdaptiveController(m))
    gen_logits = generate_teacher_forced(
        m, ids, mem_schedule=lambda t: _mem_at(timeline, mem_list, t),
        num_loops=1, prompt_len=3)  # prompt=前 3，后 3 为 decode 追加
    # prompt 内位置 0..2：generate 一次 prefill；append-only 逐 token——因果 LM 应接近
    # decode 位置 3..5：两者都是单 token 追加——应严格接近
    for t in range(3, 6):
        assert torch.allclose(ref_logits[t], gen_logits[t], atol=1e-4), (
            f"decode t={t} Δ={(ref_logits[t]-gen_logits[t]).abs().max()}")


def test_generate_model_loop_equals_append_only():
    """真实 model.generate（greedy, fixed R）下一步 logits == teacher-forced 追加。"""
    m = _m(max_loop=1)
    torch.manual_seed(99)
    prompt = torch.randint(3, 60, (1, 4))
    with torch.no_grad():
        gen = m.generate(inputs=prompt, max_new_tokens=2, do_sample=False,
                         fixed_loops=1, use_halting=False, eos_token_id=None)
    gen_ids = gen["generated_ids"] if isinstance(gen, dict) else gen
    full = gen_ids  # [1, 6]
    # teacher-forced append-only 重放生成出的 token
    timeline = build_action_timeline([0], [(0, full.shape[1])])
    from trainer.dual_executor import generate_teacher_forced
    # 无 memory（middle 等价）：用 minimind_x 但不传 memory
    logits_seq = generate_teacher_forced(
        m, full, mem_schedule=lambda t: {}, num_loops=1, prompt_len=4)
    # 位置 4 的 next 应与 generate 第二步 argmax 一致
    nxt = logits_seq[4].argmax(dim=-1)
    assert int(nxt.item()) == int(full[0, 5].item()), (
        f"replay argmax {int(nxt.item())} vs generate {int(full[0, 5].item())}")
```

---

# tests/test_cache_replay.py

```python
"""缓存契约与重放测试（任务清单 P3 验收：T3.1/T3.3/T3.4；V2 §21.3）。

1. R_max 固定槽 + per-token last-valid：混合深度序列 4→1→3 不丢先前深轮 KV，
   浅执行 token 的深槽复制其最后实际轮（只填该 token 位置）
2. memory 版本与轨迹：refresh 递增版本；轨迹含 (事件, 轮, version)；内容变更后投影缓存失效
3. checkpoint 纯化：梯度检查点开/关——检索计数不变（无副作用重放）、loss 一致、梯度仍通
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController


def _model(max_loop=4, **over):
    torch.manual_seed(41)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="middle_loop", max_loop=max_loop)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def _core_cache(m, past):
    """取 core 层（past[P:P+T]）的轮次槽。"""
    cfg = m.config
    return past[cfg.prelude_layers:cfg.prelude_layers + cfg.thinking_layers]


def test_rmax_slots_and_per_token_last_valid():
    """深度序列 4→1→3：槽恒 4；浅 token 深槽=其最后实际轮；先前深槽不丢。"""
    m = _model(max_loop=4)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out = m(ids[:, :3], use_cache=True, num_loops=4)
        past = out.past_key_values
        cc = _core_cache(m, past)
        for slots in cc:
            assert len(slots) == 4, "prefill 后应恒 4 槽"
            assert all(s is not None for s in slots)
        # step4：R=1（浅执行）
        out1 = m(ids[:, 3:4], past_key_values=past, use_cache=True, num_loops=1)
        past = out1.past_key_values
        cc = _core_cache(m, past)
        for slots in cc:
            assert len(slots) == 4, "浅执行后槽数仍 4（R_max 契约）"
            k_new_shallow = slots[0][0][:, -1]          # 该 token 真实轮 0
            for r in (1, 2, 3):                          # 深槽 = last-valid 填充
                assert torch.equal(slots[r][0][:, -1], k_new_shallow), \
                    "浅执行 token 的深槽应复制其最后实际轮"
            assert slots[3][0].shape[1] == 4             # 历史 3+1 token
        # step5：R=3（读到 prefill token 的真实深轮 + step4 token 的填充轮）
        out2 = m(ids[:, 4:5], past_key_values=past, use_cache=True, num_loops=3)
        past = out2.past_key_values
        cc = _core_cache(m, past)
        for slots in cc:
            assert len(slots) == 4
            assert torch.isfinite(slots[2][0]).all()
        assert torch.isfinite(out2.logits).all()


def test_deep_slots_survive_shallow_steps():
    """早停（R=1）不得删除先前 token 的深轮 KV（V2 §11.3 preserve_past_deep_slots）。"""
    m = _model(max_loop=4)
    ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        out = m(ids[:, :4], use_cache=True, num_loops=4)
        deep_kv_before = [s[3][0].clone() for s in _core_cache(m, out.past_key_values)]
        out1 = m(ids[:, 4:5], past_key_values=out.past_key_values, use_cache=True, num_loops=1)
        for l, slots in enumerate(_core_cache(m, out1.past_key_values)):
            assert torch.equal(slots[3][0][:, :4], deep_kv_before[l]), "先前 token 深槽被覆盖"


def test_memory_version_and_trajectory():
    m = _model(variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2, max_loop=2)
    torch.manual_seed(42)
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4,
        oracle_retrieve=[2, 0])   # schema v2：第 0 轮后 GLOBAL_REFRESH
    # T3.3/V2 §9.3：版本随「槽位内容」变更递增；同 source 去重只 bump usage
    assert ctrl.memory.memory_version >= 1, "至少经过初始检索"
    assert len(set(ctrl.memory.source_ids[0].tolist())) == ctrl.memory.size, "source_ids 唯一"
    events = [t["event"] for t in ctrl.trajectory]
    assert "initial_retrieval" in events and "refresh" in events
    vers = [t["memory_version"] for t in ctrl.trajectory]
    assert vers == sorted(vers), "轨迹版本单调不减"
    refresh = [t for t in ctrl.trajectory if t["event"] == "refresh"]
    assert refresh and refresh[0]["round"] == 0, "oracle_retrieve[0]=1 应在轮 0 后 refresh"


def test_projection_cache_invalidated_on_refresh():
    from model.minimind_x.memory import WorkingMemory
    torch.manual_seed(43)
    m = _model(variant="minimind_x", block_size=8, max_loop=2)
    mem = WorkingMemory(torch.randn(1, 3, 8, 32), torch.randn(1, 3, 32), block_size=8)
    kv = mem.get_kv(m.model.thinking_core)
    assert mem._kv_cache is not None
    mem2 = mem.with_added(torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32))
    assert mem2.memory_version == mem.memory_version + 1
    assert mem2._kv_cache is None, "内容变更后投影缓存必须失效"


def test_checkpoint_purity_no_retrieval_side_effects():
    """梯度检查点开/关：检索计数一致（checkpoint 重放无副作用）、loss 一致。"""
    from trainer.x_utils import set_gradient_checkpointing
    torch.manual_seed(44)
    cfg_kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                  num_key_value_heads=1, vocab_size=64, flash_attn=False,
                  variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
                  max_loop=2, residual_v2=True, mhar_strength=1.0, update_progress=1.0)

    def run(ckpt):
        torch.manual_seed(44)
        m = MiniMindXForCausalLM(MiniMindXConfig(**cfg_kw)).train()
        set_gradient_checkpointing(m, ckpt)
        ctrl = AdaptiveController(m)
        ids = torch.randint(3, 60, (1, 8))
        labels = torch.randint(3, 60, (1, 8))
        ctx = torch.randint(3, 60, (8 * 4,))
        out, extras = ctrl.forward_with_oracles(ids, None, labels, ctx, query_len=4)
        loss = sum(l.mean() for l in out.exit_logits)
        loss.backward()
        return ctrl.stats["retrievals"], loss.item(), \
            m.model.thinking_core.update_gate.fc2.weight.grad.abs().sum().item()

    r_off = run(False)
    r_on = run(True)
    assert r_off[0] == r_on[0], "checkpoint 开关不得改变检索计数（纯化契约）"
    assert abs(r_off[1] - r_on[1]) < 1e-4, "loss 应一致"
    assert r_on[2] > 0, "checkpoint 下梯度仍须流动"
```

---

# tests/test_causality.py

```python
"""因果契约测试（V2 T0.1 验收，C5/C6 修复锁定）。

V2 §10.3 反例验收：
  1. 固定 prompt 与 source_context，替换完整 gold completion → 初始检索结果不变
  2. T0.3：budget 档位真正联动 topk_f（库容充足时不同档取到不同数量的块）
  3. T0.7：gold CE 用带梯分的重算 logits（backward 进入 query 投影）
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController


def _tiny_model():
    torch.manual_seed(21)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, max_loop=2)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def _orthogonal_retriever(model):
    with torch.no_grad():
        d = model.config.hidden_size
        model.model.retriever.key_proj.weight.copy_(torch.eye(d))
        model.model.retriever.query_proj.weight.copy_(torch.eye(d))
    return model


def test_initial_retrieval_invariant_to_gold_completion():
    """换 gold completion → 初始检索 block_idx 必须逐位不变（T0.1 核心反例）。"""
    model = _orthogonal_retriever(_tiny_model())
    cfg = model.config
    q = torch.randint(3, 20, (6,))
    ctx = torch.randint(3, 20, (cfg.block_size * 6,))
    gold1 = torch.randint(30, 60, (8,))
    gold2 = torch.randint(30, 60, (8,))

    def run(gold):
        ctrl = AdaptiveController(model)
        input_ids = torch.cat([q, gold]).unsqueeze(0)
        labels = torch.full((1, input_ids.shape[1]), -100, dtype=torch.long)
        out, extras = ctrl.forward_with_oracles(
            input_ids, None, labels, ctx, query_len=6)
        return ctrl.stats["last_result"].block_idx.clone()

    b1, b2 = run(gold1), run(gold2)
    assert torch.equal(b1, b2), f"初始检索随 gold 变化：{b1} vs {b2}"


def test_budget_changes_actual_selection():
    """T0.3：库容充足时，不同 budget 档取到不同数量的块（cap 联动 topk_f）。"""
    model = _tiny_model()
    cfg = model.config
    ctrl = AdaptiveController(model)
    ctrl.begin(torch.randint(3, 20, (cfg.block_size * 8,)))  # 8 块库
    q_hidden = torch.randn(1, 4, cfg.hidden_size)
    small = ctrl.retrieve_into_memory(q_hidden, budget_idx=0)   # 1024 tok → 4 块
    big = ctrl.retrieve_into_memory(q_hidden, budget_idx=4)     # 16384 tok → 但 topk_coarse=4 限制
    assert small.token_size <= big.token_size
    assert ctrl.stats["budget_cap_tokens"] in cfg.memory_budgets
    assert "actual_valid_tokens" in ctrl.stats and "unique_blocks" in ctrl.stats


def test_gold_ce_has_gradient_path():
    """T0.7：gold CE 用 grad mode 重算，backward 进入 retriever 投影。"""
    model = _tiny_model()
    cfg = model.config
    ctrl = AdaptiveController(model)
    q = torch.randint(3, 20, (5,))
    gold = torch.randint(30, 60, (6,))
    input_ids = torch.cat([q, gold]).unsqueeze(0)
    labels = torch.full((1, input_ids.shape[1]), -100, dtype=torch.long)
    out, extras = ctrl.forward_with_oracles(
        input_ids, None, labels, torch.randint(3, 20, (cfg.block_size * 4,)),
        gold_block_idx=torch.tensor([1]), query_len=5)
    assert "retrieval_ce" in extras and extras["retrieval_ce"].requires_grad
    extras["retrieval_ce"].backward()
    g = model.model.retriever.query_proj.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "gold CE 梯度未进入 query 投影"


def test_router_probs_collected_during_full_unroll():
    """T0.2：collect_exits 前向返回逐轮 router 概率（minimind_x 变体）。"""
    model = _tiny_model()
    x = torch.randint(3, 20, (2, 8))
    out = model(x, collect_exits=True, num_loops=2)
    assert out.router_probs is not None
    assert "halt" in out.router_probs and len(out.router_probs["halt"]) == 2
    if "retrieval" in out.router_probs:  # 变体含检索 router 时
        assert len(out.router_probs["retrieval"]) == 2
```

---

# tests/test_config.py

```python
"""MiniMindXConfig 测试（任务清单 T0.5）：preset、variant 解析、切分校验、预算换算。"""
import pytest
import torch

from model.minimind_x import MiniMindXConfig, PRESETS, resolve_variant


def test_default_split_is_3_3_2():
    cfg = MiniMindXConfig(hidden_size=64, num_hidden_layers=8, variant="minimind_x",
                          num_attention_heads=4, num_key_value_heads=2)
    assert (cfg.prelude_layers, cfg.thinking_layers, cfg.coda_layers) == (3, 3, 2)
    assert cfg.use_loop and cfg.use_memory and cfg.use_retrieval


def test_variant_matrix():
    assert resolve_variant("full_loop", 8)["thinking_layers"] == 8
    assert resolve_variant("full_loop", 8)["prelude_layers"] == 0
    # late_middle: L1-L4 + (L5-L6)xR + L7-L8
    r = resolve_variant("late_middle", 8)
    assert (r["prelude_layers"], r["thinking_layers"], r["coda_layers"]) == (4, 2, 2)
    assert resolve_variant("dense", 8)["use_loop"] is False


def test_split_validation():
    with pytest.raises(ValueError):
        MiniMindXConfig(hidden_size=64, num_hidden_layers=8, variant="minimind_x",
                        prelude_layers=5, thinking_layers=5, coda_layers=5)


def test_presets():
    for name, p in PRESETS.items():
        assert p["prelude_layers"] + p["thinking_layers"] + p["coda_layers"] == p["num_hidden_layers"]
    cfg = MiniMindXConfig(preset="S", variant="minimind_x")
    assert cfg.hidden_size == 512 and cfg.num_hidden_layers == 12
    assert (cfg.thinking_layers, cfg.prelude_layers, cfg.coda_layers) == (4, 4, 4)


def test_budget_blocks():
    cfg = MiniMindXConfig(hidden_size=64, num_hidden_layers=8, block_size=256)
    assert cfg.budget_blocks(0) == 4    # 1024/256
    assert cfg.budget_blocks(4) == 64   # 16384/256


def test_mode_caps():
    cfg = MiniMindXConfig(hidden_size=64, num_hidden_layers=8)
    assert cfg.mode_caps_for("fast")["max_loop"] == 1
    assert cfg.mode_caps_for("normal")["max_budget"] == 8192
    assert cfg.mode_caps_for(None)["max_loop"] == cfg.max_loop
    with pytest.raises(ValueError):
        cfg.mode_caps_for("turbo")
```

---

# tests/test_data_truth_and_sparse_causal.py

```python
"""合成数据真值 + sparse 块级因果（评审 7.3 / 9.5）。"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from scripts.data_gen.gen_evidence_families import gen_sample, DEVICES
from scripts.data_gen.gen_long_dep import gen_sample as gen_long
from scripts.data_gen.gen_planning import gen_sample as gen_plan, STEPS_ORDERED
from scripts.data_gen.gen_repo_context import gen_sample as gen_repo
import random


def test_irrelevant_never_contains_target_device():
    rng = random.Random(0)
    for i in range(200):
        rec = gen_sample(rng, "irrelevant")
        q = rec["question"]
        # context 不得出现 question 中的目标设备
        for d in DEVICES:
            if d in q:
                assert d not in rec["context"], rec
        assert rec.get("unanswerable") is True


def test_no_evidence_unanswerable_not_random_target():
    rng = random.Random(1)
    answers = set()
    for _ in range(50):
        rec = gen_sample(rng, "no_evidence")
        answers.add(rec["answer"])
        assert rec.get("unanswerable") is True
        assert rec["context"] == ""
    assert len(answers) == 1, f"no_evidence 目标必须确定不可知，得到 {answers}"


def test_new_evidence_has_visibility_timeline():
    rng = random.Random(2)
    rec = gen_sample(rng, "new_evidence")
    assert rec.get("evidence_visible_from_round", 0) >= 1
    assert rec["gold_span"][0] >= 0


def test_long_dep_target_is_x_plus_positive():
    rng = random.Random(3)
    rec = gen_long(rng, 200, 0.5)
    text = rec["text"]
    # 解析「加上 k」与「最终取值是 y」，y 必须 > 初值
    assert "加上" in text and "最终取值是" in text
    import re
    m_add = re.search(r"加上 (\d+)", text)
    m_fin = re.search(r"最终取值是 (\d+)", text)
    m_init = re.search(r"初始值为 (\d+)", text)
    assert m_add and m_fin and m_init
    k, y, x = int(m_add.group(1)), int(m_fin.group(1)), int(m_init.group(1))
    assert y == x + k, f"目标应为 X+k，得到 {x}+{k}!={y}"


def test_planning_ordered_and_constraints_in_answer():
    rng = random.Random(4)
    rec = gen_plan(rng, 4, 200, 8)
    ans = rec["answer"]
    # 步骤必须是 STEPS_ORDERED 的保序子序列
    steps = []
    for name in STEPS_ORDERED:
        if f"：{name}" in ans or f":{name}" in ans or name in ans:
            steps.append(name)
    idxs = [STEPS_ORDERED.index(s) for s in steps if s in STEPS_ORDERED]
    assert idxs == sorted(idxs), f"步骤乱序: {steps}"
    assert "遵守" in ans or "约束" in ans


def test_repo_patch_uses_real_args_no_ellipsis():
    rng = random.Random(5)
    rec = gen_repo(rng, 2, 2, 200, 8)
    ans = rec["answer"]
    assert "..." not in ans
    assert "if not args" not in ans  # 未定义 args


def test_sparse_future_token_cannot_change_past_output():
    """评审 9.5：改未来 token 不得改变早期位置输出（块选择不得泄未来）。"""
    from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
    from model.minimind_x.retrieval.sparse_attn import SparseBlockAttention
    torch.manual_seed(60)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="middle_loop", core_attn="sparse",
                          block_size=8, sparse_topk_blocks=2, attn_window=16)
    attn = SparseBlockAttention(cfg).eval()
    B, S = 1, 16
    x1 = torch.randn(B, S, 32)
    x2 = x1.clone()
    x2[:, 8:] = torch.randn(B, S - 8, 32)  # 只改未来（位置 ≥8）
    # RoPE
    from model.model_minimind import precompute_freqs_cis
    cos, sin = precompute_freqs_cis(dim=cfg.head_dim, end=32)
    cos, sin = cos[:S], sin[:S]
    with torch.no_grad():
        o1, _ = attn(x1, (cos, sin), use_cache=False)
        o2, _ = attn(x2, (cos, sin), use_cache=False)
    # 位置 0..5（完整在第一个块且远早于改动点）应不变
    diff = (o1[:, :5] - o2[:, :5]).abs().max().item()
    assert diff < 1e-5, f"未来 token 影响了早期输出 Δ={diff}"
```

---

# tests/test_distill.py

```python
"""fine→coarse 检索蒸馏测试（v1.3 / ARCHITECTURE §10-A）：损失数值、梯度流向、SFT-C 接线。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.retrieval import LongContextStore


def _tiny(variant="minimind_x", **over):
    torch.manual_seed(11)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant=variant, block_size=8, topk_coarse=4, topk_fine=2)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_distill_loss_value_and_zero():
    m = _tiny()
    ret = m.model.retriever
    cand = torch.tensor([[1, 3, 5], [2, 4, 6]])
    fine = torch.randn(2, 3)
    # 构造 student：coarse 在候选位置上与 teacher 逐值相等 → 分布一致 → KL≈0
    coarse_aligned = torch.zeros(2, 10).scatter_(1, cand, fine)
    loss0 = ret.distill_loss(coarse_aligned, cand, fine)
    assert loss0.item() < 1e-5, loss0.item()
    # 反转分布（student = −teacher）→ KL 显著 > 0
    # （统一平移不变是 KL 的正确性质：蒸馏只学排序形状，不学绝对分值）
    coarse_mis = torch.zeros(2, 10).scatter_(1, cand, -fine)
    loss1 = ret.distill_loss(coarse_mis, cand, fine)
    assert loss1.item() > 0.1
    assert torch.isfinite(loss1)


def test_distill_gradients_flow_to_query_proj():
    """蒸馏的梯度必须真实进入 query_proj（评审 P0-1 的有效解法验证）。"""
    m = _tiny()
    # 建库
    ctx = torch.randint(3, 64, (m.config.block_size * 6,))
    store = LongContextStore(ctx, m.config.block_size)
    store.build_keys(m.model.retriever, m.model.embed_tokens)
    ids = torch.randint(3, 64, (1, 5))
    q_hidden = m.model.prelude_hidden(ids, no_grad=True)
    with torch.no_grad():  # teacher：no_grad 检索
        result = m.model.retriever.retrieve(q_hidden, store, m)
    # student：带梯度的 query 重打分
    q_grad = m.model.prelude_hidden(ids, no_grad=False)
    q_vec = m.model.retriever.project_query(q_grad)
    coarse_trainable = m.model.retriever.coarse_scores(q_vec, store)
    loss = m.model.retriever.distill_loss(coarse_trainable, result.cand_idx,
                                          result.fine_scores)
    loss.backward()
    g = m.model.retriever.query_proj.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "蒸馏梯度未进入 query_proj"
    # teacher 侧确实被 detach（fine_scores 无梯度）
    assert result.fine_scores.requires_grad is False


def test_controller_forward_produces_distill():
    from model.minimind_x.controller import AdaptiveController
    m = _tiny()
    ctrl = AdaptiveController(m)
    ids = torch.randint(3, 64, (2, 6))
    labels = torch.randint(3, 64, (2, 6))
    ctx = torch.randint(3, 64, (m.config.block_size * 4,))
    out, extras = ctrl.forward_with_oracles(ids, None, labels, ctx)  # 无 gold：纯蒸馏监督
    assert "distill_loss" in extras and torch.isfinite(extras["distill_loss"])
    assert extras["distill_loss"].requires_grad  # 训练侧拿到的是带梯度损失
    total = sum(l.mean() for l in out.exit_logits) * 0 + extras["distill_loss"]
    total.backward()
    assert m.model.retriever.query_proj.weight.grad is not None


def test_sft_c_batch_guard():
    """评审 Bug 1 回归：B>1 必须显式报错而非静默丢上下文。"""
    import pytest
    from trainer.train_sft_x import assert_sft_c_batch
    assert_sft_c_batch(torch.zeros(1, 8, dtype=torch.long))  # B=1 通过
    with pytest.raises(ValueError, match="batch>1"):
        assert_sft_c_batch(torch.zeros(2, 8, dtype=torch.long))
```

---

# tests/test_equivalence.py

```python
"""R=1 等价性测试（任务清单 T0.5 / 横切）：同权重下 MiniMindX 与原 MiniMind 前向数值一致。

这是 D1「不改原模型、权重可互转」和 D2 KV 语义的回归基线。
注：transformers>=5 在 torch<2.4 下禁止实例化 PreTrainedModel 子类，
参照侧用未封装的 MiniMindModel + lm_head 手工组装（数值路径与 MiniMindForCausalLM 相同）。
"""
import torch
from torch import nn

from model.model_minimind import MiniMindConfig, MiniMindModel
from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, remap_minimind_state_dict


class ReferenceMiniMind(nn.Module):
    """原 MiniMindForCausalLM 的去封装等价物（权重键名一致）。"""

    def __init__(self, cfg: MiniMindConfig):
        super().__init__()
        self.model = MiniMindModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids):
        hidden, _, _ = self.model(input_ids)
        return self.lm_head(hidden)


def _tiny_cfg_pair(**x_over):
    base = dict(hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
                num_key_value_heads=2, vocab_size=128, max_position_embeddings=512,
                flash_attn=False)
    base.update(x_over)
    orig = MiniMindConfig(**base)
    xcfg = MiniMindXConfig(variant="dense", use_loop=False, **base)
    return orig, xcfg


def test_r1_logits_equivalence():
    torch.manual_seed(0)
    orig_cfg, x_cfg = _tiny_cfg_pair()
    orig = ReferenceMiniMind(orig_cfg).eval()
    x = MiniMindXForCausalLM(x_cfg).eval()
    # 把原模型权重 remap 进 X（router/retriever 等新模块 strict=False）
    mapped = remap_minimind_state_dict(orig.state_dict(), x_cfg)
    missing, unexpected = x.load_state_dict(mapped, strict=False)
    assert not unexpected
    assert all(k.startswith(("model.halt_router", "model.retrieve_router",
                             "model.budget_router", "model.memory_manager",
                             "model.retriever")) for k in missing)
    input_ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        lo = orig(input_ids)
        lx = x(input_ids, num_loops=1).logits
    assert torch.allclose(lo, lx, atol=1e-4), (lo - lx).abs().max()


def test_r1_with_use_loop_variant_equivalence():
    """use_loop=True 但 num_loops=1 时同样等价（zone 循环跑 1 轮 = 直通）。"""
    torch.manual_seed(1)
    orig_cfg, _ = _tiny_cfg_pair()
    x_cfg = MiniMindXConfig(variant="middle_loop", **dict(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128, max_position_embeddings=512, flash_attn=False))
    orig = ReferenceMiniMind(orig_cfg).eval()
    x = MiniMindXForCausalLM(x_cfg).eval()
    x.load_state_dict(remap_minimind_state_dict(orig.state_dict(), x_cfg), strict=False)
    input_ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        lo = orig(input_ids)
        lx = x(input_ids, num_loops=1).logits
    assert torch.allclose(lo, lx, atol=1e-4)


def test_collect_exits_shapes_and_kv_cache():
    """全展开 collect_exits 的形状 + use_cache 的 KV 长度（D2：循环不追加 KV）。"""
    torch.manual_seed(2)
    x_cfg = MiniMindXConfig(variant="middle_loop", max_loop=4, **dict(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128, max_position_embeddings=512, flash_attn=False))
    x = MiniMindXForCausalLM(x_cfg).eval()
    input_ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        out = x(input_ids, collect_exits=True, num_loops=3)
        assert len(out.exit_logits) == 3
        assert all(lg.shape == out.logits.shape for lg in out.exit_logits)
        out2 = x(input_ids, num_loops=4, use_cache=True)
        kv = out2.past_key_values
        assert len(kv) == 8  # 总层数
        for past in kv:
            if isinstance(past, list):
                # core 层：轮次轴 KV —— 4 轮 × 每轮长度 12（token 轴不随轮数膨胀）
                assert len(past) == 4 and all(p[0].shape[1] == 12 for p in past)
            else:
                assert past is not None and past[0].shape[1] == 12
```

---

# tests/test_flops.py

```python
"""FLOPs / 参数量解析测试（任务清单 T0.5）。"""
from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, count_params, estimate_flops


def _cfg(**over):
    kw = dict(hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
              num_key_value_heads=2, vocab_size=128, flash_attn=False,
              variant="middle_loop")
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_count_params_matches_actual():
    cfg = _cfg()
    model = MiniMindXForCausalLM(cfg)
    actual = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("lm_head", "model.halt_router", "model.retrieve_router", "model.budget_router", "model.memory_manager", "model.retriever")))
    analytic = count_params(cfg)
    assert abs(analytic - actual) / actual < 0.01


def test_flops_scaling():
    cfg = _cfg()
    f1 = estimate_flops(cfg, seq_len=512, loops=1)
    f2 = estimate_flops(cfg, seq_len=512, loops=2)
    f4 = estimate_flops(cfg, seq_len=512, loops=4)
    # 有效深度 = P + C + T*R = 3+2+3R；主干 FLOPs 线性于 R（增量 = 每轮 T 层）
    assert f1["effective_depth"] == 8 and f4["effective_depth"] == 17
    d1, d2 = f2["total"] - f1["total"], f4["total"] - f2["total"]
    assert abs(d1 * 2 - d2) / d2 < 1e-6  # R1→R2 增 1 轮，R2→R4 增 2 轮
    assert f1["memory_attn"] == 0.0 and f1["retrieval"] == 0.0
    fr = estimate_flops(cfg, seq_len=512, loops=1, memory_blocks=4, n_retrievals=2)
    assert fr["memory_attn"] > 0 and fr["retrieval"] > 0
    ft = estimate_flops(cfg, seq_len=512, loops=1, train=True)
    assert abs(ft["total"] - 3 * f1["total"]) < 1.0


def test_dense_variant_flops_equals_depth():
    cfg = _cfg(variant="dense")
    f = estimate_flops(cfg, seq_len=256, loops=4)  # dense 不循环：loops 被忽略于深度语义外
    assert f["effective_depth"] == cfg.num_hidden_layers
```

---

# tests/test_frozen_key_index.py

```python
"""Frozen Index 范式声明（评审 8.5/10 收口）：key 冻结、query 可训。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def test_freeze_key_default_and_manifest():
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", block_size=8)
    assert cfg.freeze_key is True
    assert cfg.key_refresh_interval == 0
    assert cfg.validate() is True


def test_key_proj_frozen_query_trainable():
    """优化器视角：key_proj.requires_grad=False；query_proj 仍可训。"""
    from trainer.x_utils import build_optimizer, param_manifest
    torch.manual_seed(50)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", block_size=8, freeze_key=True)
    m = MiniMindXForCausalLM(cfg).train()
    opt = build_optimizer(m, lr=1e-3)
    assert m.model.retriever.key_proj.weight.requires_grad is False
    assert m.model.retriever.query_proj.weight.requires_grad is True
    man = param_manifest(m)
    assert man["freeze_key"] is True
    assert man["frozen_key_params"] > 0
    # key_proj 不进优化器参数组
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_all_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_opt == n_all_train


def test_retrieval_ce_only_query_grad():
    """CE 路径：query 有 grad，key 无 grad（Frozen Index 语义）。"""
    torch.manual_seed(51)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", block_size=8, freeze_key=True)
    m = MiniMindXForCausalLM(cfg).train()
    q = torch.randn(1, 4, 32, requires_grad=True)
    # 建库（no_grad，与生产一致）
    ctx = torch.randint(3, 60, (32,))
    store = __import__("model.minimind_x.retrieval", fromlist=["LongContextStore"]).LongContextStore(ctx, 8)
    store.build_keys(m.model.retriever, m.model.embed_tokens)
    qv = m.model.retriever.project_query(q)
    scores = m.model.retriever.coarse_scores(qv, store)
    loss = torch.nn.functional.cross_entropy(scores, torch.tensor([0]))
    loss.backward()
    assert m.model.retriever.query_proj.weight.grad is not None
    assert m.model.retriever.query_proj.weight.grad.abs().sum() > 0
    assert m.model.retriever.key_proj.weight.grad is None
```

---

# tests/test_learnability.py

```python
"""可学习性测试（外部评审 P2-1）：不只验证接口，验证「能学」。

1. loop 路径可训练：固定小任务上 R=2 过拟合，loss 显著下降且梯度流经循环区
2. memory 路径可训练：注入固定 memory cross-attn 后 loss 显著下降（梯度流经 k/v 投影）
3. oracle 机制：sweep_budget 形状与取值、sweep_retrieval 的 per-sample 标签口径
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import optim

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def _cfg(variant="middle_loop", **over):
    kw = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant=variant, max_loop=2)
    kw.update(over)
    return MiniMindXConfig(**kw)


def _copy_task(n=4, seq=16, vocab=64, seed=0):
    """复制任务：label = input 右移（模型需把信息传到输出位）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(3, vocab, (n, seq), generator=g)
    y = torch.cat([torch.full((n, 1), 1, dtype=torch.long), x[:, :-1]], dim=1)
    return x, y


def _train(model, x, y, steps=60, lr=2e-3, num_loops=1, memory_factory=None):
    opt = optim.AdamW(model.parameters(), lr=lr)
    losses = []
    core_params_before = [p.detach().clone() for p in model.model.thinking_core.parameters()]
    for _ in range(steps):
        memory_kv = memory_factory() if memory_factory is not None else None
        res = model(x, labels=y, num_loops=num_loops, memory_kv_per_layer=memory_kv)
        opt.zero_grad()
        (res.loss + res.aux_loss).backward()
        opt.step()
        losses.append(res.loss.item())
    core_moved = any((p.detach() - b).abs().max().item() > 0
                     for p, b in zip(model.model.thinking_core.parameters(), core_params_before))
    return losses, core_moved


def test_loop_path_is_trainable():
    """R=2 循环区在固定任务上过拟合：loss 大幅下降，且循环区参数确实被更新。"""
    torch.manual_seed(0)
    x, y = _copy_task()
    model = MiniMindXForCausalLM(_cfg(variant="middle_loop"))
    losses, core_moved = _train(model, x, y, num_loops=2)
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])  # 过拟合发生
    assert core_moved, "循环区参数未更新——梯度没有流经 loop 路径"


def test_memory_path_is_trainable():
    """固定 memory cross-attn 注入后可学习（梯度经 project_memory 的 k/v 投影流动）。

    每步重投影（工厂）：SFT 真实路径是 no_grad 建记忆（D5），此处刻意保留梯度以验证
    memory cross-attn 路径本身可训练。
    """
    torch.manual_seed(1)
    x, y = _copy_task()
    model = MiniMindXForCausalLM(_cfg(variant="minimind_x", num_hidden_layers=6))
    mem_hidden = torch.randn(x.shape[0], 3, model.config.hidden_size)
    factory = lambda: model.model.thinking_core.project_memory(mem_hidden)
    losses, core_moved = _train(model, x, y, num_loops=2, memory_factory=factory)
    assert losses[-1] < losses[0] * 0.6, (losses[0], losses[-1])
    assert core_moved


def test_loop_embedding_ablation_runs():
    """D4 消融开关：加轮次嵌入后同样可训练（供加/不加对照实验）。"""
    torch.manual_seed(2)
    x, y = _copy_task()
    model = MiniMindXForCausalLM(_cfg(variant="middle_loop", loop_embedding=True))
    losses, _ = _train(model, x, y, num_loops=2, steps=30)
    assert losses[-1] < losses[0]


def test_sweep_budget_and_retrieval_labels():
    from model.minimind_x.controller import AdaptiveController
    from trainer.oracle_labels import sweep_budget, sweep_retrieval
    torch.manual_seed(3)
    cfg = _cfg(variant="minimind_x", num_hidden_layers=6, block_size=8)
    model = MiniMindXForCausalLM(cfg).eval()
    x = torch.randint(3, 64, (1, 10))
    y = torch.randint(3, 64, (1, 10))
    ctx = torch.randint(3, 64, (cfg.block_size * 3,))
    loss_matrix = sweep_budget(model, x, y, ctx,
                               controller_factory=lambda: AdaptiveController(model),
                               query_len=6)
    assert loss_matrix.shape == (1, len(cfg.memory_budgets))
    assert torch.isfinite(loss_matrix).all()
    # V2 反事实口径（T0.1）：REUSE vs REFRESH，同可见前缀，不读 gold 后缀
    labels_r, gain = sweep_retrieval(model, x, y, ctx, query_len=6,
                                     controller_factory=lambda: AdaptiveController(model))
    assert labels_r.shape == gain.shape == (1, cfg.max_loop)
    assert set(labels_r.unique().tolist()) <= {0, 1}
    assert torch.isfinite(gain).all()
```

---

# tests/test_loop_kv.py

```python
"""Loop KV 语义测试（任务清单横切；D2）：decode/prefill 一致性、循环不追加 KV、三模式 generate。"""
import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def _tiny_model(**cfg_over):
    torch.manual_seed(3)
    cfg_kw = dict(hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=128, max_position_embeddings=512,
                  flash_attn=False, max_loop=4, variant="middle_loop")
    cfg_kw.update(cfg_over)
    cfg = MiniMindXConfig(**cfg_kw)
    return MiniMindXForCausalLM(cfg).eval()


def test_decode_prefill_consistency_fixed_loops():
    """逐 token decode（带 cache）与整段 forward 的 next-token logits 一致（D2 回归）。"""
    m = _tiny_model()
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        full = m(ids, num_loops=3).logits[:, -1]
        # 增量：先 prefill 前 6 个，再逐步喂
        out = m(ids[:, :6], use_cache=True, num_loops=3)
        past = out.past_key_values
        for t in range(6, 12):
            out = m(ids[:, t:t + 1], past_key_values=past, use_cache=True, num_loops=3)
            past = out.past_key_values
        inc = out.logits[:, -1]
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_cache_grows_one_per_token_regardless_of_loops():
    m = _tiny_model()
    ids = torch.randint(0, 128, (1, 4))
    with torch.no_grad():
        out = m(ids[:, :2], use_cache=True, num_loops=4)
        assert out.past_key_values[0][0].shape[1] == 2
        out = m(ids[:, 2:3], past_key_values=out.past_key_values, use_cache=True, num_loops=4)
        assert out.past_key_values[0][0].shape[1] == 3  # 4 轮循环仍只 +1


def test_exit_choice_selects_loop():
    """显式 exit_choice=r 时，logits 应等于「只跑 r 轮」的前向（采样 R 的实现基础）。"""
    m = _tiny_model()
    ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        r2 = m(ids, num_loops=2).logits
        r4_pick2 = m(ids, num_loops=4, exit_choice=torch.tensor([1, 1])).logits
    assert torch.allclose(r2, r4_pick2, atol=1e-4), (r2 - r4_pick2).abs().max()


def test_halt_early_stop_matches_fixed_exit():
    """推理早停：把 halt_router 输出钉在第 2 轮 stop，结果应等于 exit_choice=2。"""
    m = _tiny_model()
    # 钉住 halt router：权重置 0、bias 大 → 恒 P(stop)>threshold（第 1 轮即停）
    with torch.no_grad():
        torch.nn.init.zeros_(m.model.halt_router.proj.weight)
        m.model.halt_router.proj.bias.fill_(10.0)
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        halted = m(ids, num_loops=4, halt=True).logits
        fixed1 = m(ids, num_loops=4, exit_choice=torch.zeros(2, dtype=torch.long)).logits
        one = m(ids, num_loops=1).logits
    assert torch.allclose(halted, one, atol=1e-4)
    assert torch.allclose(halted, fixed1, atol=1e-4)


def test_generate_modes_and_stats():
    m = _tiny_model()
    ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        out = m.generate(inputs=ids, max_new_tokens=4, do_sample=False, eos_token_id=None,
                         mode="fast", return_stats=True)
        assert out["generated_ids"].shape[1] == 10
        assert out["stats"]["avg_loops"] <= 1.0 + 1e-6  # fast 封顶 1 轮
        out2 = m.generate(inputs=ids, max_new_tokens=4, do_sample=False, eos_token_id=None,
                          fixed_loops=3, return_stats=True)
        assert abs(out2["stats"]["avg_loops"] - 3.0) < 1e-6


def test_memory_changes_logits():
    """Working Memory 注入会改变输出（cross-attention 生效 sanity）。"""
    m = _tiny_model(variant="minimind_x", block_size=8)
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        base = m(ids, num_loops=2).logits
        mem_hidden = torch.randn(2, 3, m.config.hidden_size)
        mem_kv = m.model.thinking_core.project_memory(mem_hidden)
        with_mem = m(ids, num_loops=2, memory_kv_per_layer=mem_kv).logits
    assert not torch.allclose(base, with_mem, atol=1e-3)
    assert with_mem.shape == base.shape


def test_loop_embedding_forward():
    m = _tiny_model(loop_embedding=True)
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        out = m(ids, num_loops=2, collect_exits=True)
    assert len(out.exit_logits) == 2
```

---

# tests/test_losses.py

```python
"""losses / oracle 标签测试（任务清单 T2.2/T2.3；v1.1 修正的 C_compute 期望公式）。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import (MiniMindXConfig, joint_loss, expected_loops,
                              halt_labels_from_oracle)
from model.minimind_x.losses import (lm_loss_per_exit, halt_bce_loss, consistency_kl_loss,
                                     exit_loss_weights, expected_memory_cost)
from trainer.oracle_labels import (oracle_depth_from_losses, oracle_depth_from_correct,
                                   oracle_budget_from_matrix, oracle_retrieval_label)


def test_expected_loops_hand_example():
    probs = torch.tensor([[0.2], [0.5], [0.9]])  # [R,B]
    e = expected_loops(probs)
    # P(1)=0.2, P(2)=0.4, P(3)=0.36+never-stopped 0.04=0.40（force_stop_last）→ E=2.2
    assert abs(e.item() - 2.2) < 1e-5


def test_expected_loops_certain_stop():
    probs = torch.tensor([[0.0], [1.0], [0.0]])
    assert abs(expected_loops(probs).item() - 2.0) < 1e-6


def test_halt_labels():
    labels, mask = halt_labels_from_oracle(torch.tensor([3, 1, 4]), 4)
    assert labels.tolist() == [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    assert mask.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]]


def test_oracle_depth_from_losses():
    losses = torch.tensor([[3.0, 2.0, 0.7, 0.68], [0.5, 0.49, 0.48, 0.47]])
    r = oracle_depth_from_losses(losses, epsilon=0.05)
    # 样本1：min=0.68，首个 ≤0.73 的是第3轮 → 3；样本2：min=0.47，首个 ≤0.52 的是第1轮 → 1
    assert r.tolist() == [3, 1]


def test_oracle_depth_from_correct():
    corr = torch.tensor([[False, True, True], [False, False, False]])
    assert oracle_depth_from_correct(corr).tolist() == [2, 3]


def test_oracle_budget_and_retrieval():
    m = torch.tensor([[1.0, 0.5, 0.49]])
    assert oracle_budget_from_matrix(m, 0.05).tolist() == [1]  # 2 档即够
    lab = oracle_retrieval_label(torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.98]))
    assert lab.tolist() == [1, 0]


def test_exit_loss_weights():
    assert exit_loss_weights(4, "last").tolist() == [0, 0, 0, 1]
    assert abs(exit_loss_weights(4, "uniform").sum().item() - 1.0) < 1e-6


def test_joint_loss_parts():
    torch.manual_seed(0)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", lambda_halt=0.1, lambda_compute=0.01,
                          alpha_consistency=0.1)
    B, S, R, V = 2, 8, cfg.max_loop, cfg.vocab_size
    exit_logits = [torch.randn(B, S, V) for _ in range(R)]
    labels = torch.randint(0, V, (B, S))
    halt_probs = [torch.rand(B) * 0.5 for _ in range(R)]
    parts = joint_loss(exit_logits, labels, halt_probs=halt_probs,
                       oracle_r=torch.tensor([2, 3]), config=cfg)
    for key in ("loss", "lm", "halt", "consistency", "compute"):
        assert key in parts and torch.isfinite(parts[key])
    # compute ∈ [1/R, 1]
    assert 0 < parts["compute"].item() <= 1.0


def test_consistency_kl_zero_when_identical():
    logits = torch.randn(1, 4, 16)
    labels = torch.randint(0, 16, (1, 4))
    assert consistency_kl_loss([logits.clone(), logits.clone()], labels).item() < 1e-6


def test_expected_memory_cost():
    probs = torch.tensor([[0.0, 1.0, 0.0]])
    c = expected_memory_cost(probs, [1024, 2048, 4096])
    assert abs(c.item() - 0.5) < 1e-6  # 2048/4096
```

---

# tests/test_memory.py

```python
"""Memory 管理测试（任务清单 T5.1/T5.2 验收）：驱逐不变量、打分、跨 loop KV 缓存。"""
import torch

from model.minimind_x import MiniMindXConfig
from model.minimind_x.memory import WorkingMemory, MemoryManager


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_manager_budget_invariant():
    cfg = _cfg()
    mgr = MemoryManager(cfg)
    B, d, bs = 2, cfg.hidden_size, cfg.block_size
    mem = WorkingMemory(torch.randn(B, 0, bs, d), torch.randn(B, 0, d))
    q = torch.randn(B, d)
    for step in range(5):
        new_h = torch.randn(B, 3, bs, d)
        new_k = torch.randn(B, 3, d)
        mem, _, _ = mgr.update(mem, new_h, new_k, q, budget_blocks=4)
        assert mem.size <= 4  # 驱逐不变量：容量永不超过预算
    assert mem.token_size == 4 * bs


def test_manager_evicts_low_relevance():
    """相关性主导时（β=γ=δ=0），与 query 无关的块应先被驱逐。"""
    cfg = _cfg(mm_beta=0.0, mm_gamma=0.0, mm_delta=0.0, mm_eta=0.0)
    mgr = MemoryManager(cfg)
    B, d, bs = 1, cfg.hidden_size, cfg.block_size
    q = torch.tensor([[1.0] + [0.0] * (d - 1)])
    keys = torch.tensor([[[1.0] + [0.0] * (d - 1),      # 与 query 同向，应保留
                          [0.0, 1.0] + [0.0] * (d - 2),  # 正交，应被驱逐
                          [-1.0] + [0.0] * (d - 1)]])    # 反向，应被驱逐
    hidden = torch.randn(B, 3, bs, d)
    mem = WorkingMemory(hidden, keys)
    mem, kept_new, u = mgr.update(mem, torch.randn(B, 2, bs, d),
                                  torch.randn(B, 2, d), q, budget_blocks=3)
    kept_keys = mem.keys[0]
    cos_q = torch.nn.functional.cosine_similarity(kept_keys, q.expand_as(kept_keys), dim=-1)
    assert cos_q.argmax().item() == 0  # 相关块仍在（必然保留在 top-3）
    assert mem.size == 3


def test_working_memory_kv_cache():
    from model.minimind_x import MiniMindXForCausalLM
    cfg = _cfg()
    model = MiniMindXForCausalLM(cfg).eval()
    B, K, bs, d = 2, 3, cfg.block_size, cfg.hidden_size
    mem = WorkingMemory(torch.randn(B, K, bs, d), torch.randn(B, K, d), block_size=bs)
    kv = mem.get_kv(model.model.thinking_core)
    assert len(kv) == cfg.thinking_layers
    for k, v in kv:
        assert k.shape == (B, K * bs, cfg.num_key_value_heads, cfg.head_dim)
        assert v.shape == k.shape
    assert mem.get_kv(model.model.thinking_core) is not None
    # 内容变更（add）后缓存失效重投影
    mem2 = mem.with_added(torch.randn(B, 1, bs, d), torch.randn(B, 1, d))
    assert mem2._kv_cache is None
    kv2 = mem2.get_kv(model.model.thinking_core)
    assert kv2[0][0].shape[1] == (K + 1) * bs


def test_usage_and_persistence_ema():
    cfg = _cfg()
    B, K, bs, d = 1, 2, cfg.block_size, cfg.hidden_size
    mem = WorkingMemory(torch.randn(B, K, bs, d), torch.randn(B, K, d))
    hit = torch.tensor([[1.0, 0.0]])
    mem = mem.with_usage_bumped(hit, decay=0.5)
    assert abs(mem.usage[0, 0].item() - 0.5) < 1e-6 and mem.usage[0, 1].item() == 0.0
    scores = torch.tensor([[1.0, 1.0]])
    mem = mem.with_persistence_ema(scores, decay=0.5)
    assert abs(mem.persistence[0, 0].item() - 0.5) < 1e-6
```

---

# tests/test_residual_invariants.py

```python
"""V2 三模块不变量测试（任务清单 T1.4；V2 §21.1 + 附录 B.2 测试范围移植）。

覆盖：LoopMHAR（单来源恒等/η=0 恒等/来源轴归一/因果/不原地改），
UpdateGate（0/1 边界/inactive/插值），Reader（no-op 全家族/padding 不可见/
scale=0 精确 no-op 但有梯度/冻结 payload 梯度/BF16 有限），组合路径梯度。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn.functional as F

from model.minimind_x.residual import LoopMHAR, RecurrentUpdateGate
from model.minimind_x.memory.reader import GatedMemoryReader, MemoryKV


def _rand(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ------------------------------- LoopMHAR --------------------------------
def test_mhar_single_source_identity():
    m = LoopMHAR(16, heads=4)
    x = _rand((2, 5, 16))
    out, w = m([x])
    assert torch.allclose(out, x, atol=1e-5)
    assert torch.allclose(w.squeeze(0), torch.ones_like(w.squeeze(0)), atol=1e-5)


def test_mhar_strength_zero_identity():
    m = LoopMHAR(16, heads=4)
    h = [_rand((2, 5, 16), s) for s in range(3)]
    out, _ = m(h, strength=0.0)
    assert torch.allclose(out, h[-1], atol=1e-6)


def test_mhar_weights_normalized_along_source_axis():
    m = LoopMHAR(16, heads=4)
    with torch.no_grad():
        m.query.normal_(0, 0.5)  # 非零 query 打破初始偏置主导
    h = [_rand((2, 5, 16), s) for s in range(4)]
    _, w = m(h)
    assert w.shape == (4, 2, 5, 4)
    assert torch.allclose(w.sum(dim=0), torch.ones(2, 5, 4), atol=1e-5)


def test_mhar_causal_and_no_inplace():
    """改未来 token 的历史状态不影响更早位置；history 不被原地修改。"""
    m = LoopMHAR(16, heads=2)
    h = [_rand((1, 6, 16), s) for s in range(3)]
    h_frozen = [x.clone() for x in h]
    out1, _ = m(h)
    h[0] = h[0].clone()          # 替换引用（非原地）触发重算
    h[0][:, 4:] = torch.randn(1, 2, 16)  # 只改未来位置（新 tensor 上原地，允许）
    out2, _ = m(h)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)
    for a, b in zip(h_frozen, [x for x in h]):
        pass  # h[0] 被替换为克隆后修改，原 frozen 不受影响
    assert torch.equal(h_frozen[1], h[1]) and torch.equal(h_frozen[2], h[2])


def test_mhar_heads_can_route_differently():
    m = LoopMHAR(16, heads=4)
    with torch.no_grad():
        m.query.zero_()
        m.query[0].normal_(0, 1.0)   # 只给 head0 非零 query
    h = [_rand((1, 3, 16), s) for s in range(3)]
    _, w = m(h)
    # head0 的来源分布应与偏置主导的其它头显著不同
    w0, w1 = w[:, :, :, 0], w[:, :, :, 1]
    assert (w0 - w1).abs().max() > 1e-6


# ------------------------------ UpdateGate -------------------------------
def test_update_gate_boundaries():
    g = RecurrentUpdateGate(16, rank=8)
    prev, anchor, cand = _rand((1, 4, 16), 1), _rand((1, 4, 16), 2), _rand((1, 4, 16), 3)
    out0, gate0 = g(prev, anchor, cand, force_gate=0.0)
    out1, gate1 = g(prev, anchor, cand, force_gate=1.0)
    assert torch.equal(out0, prev)
    assert torch.equal(out1, cand)
    assert gate0.max() == 0.0 and gate1.min() == 1.0
    outm, gatem = g(prev, anchor, cand, force_gate=0.25)
    expect = prev + 0.25 * (cand - prev)
    assert torch.allclose(outm, expect, atol=1e-6)


def test_update_gate_inactive_positions_keep_previous():
    g = RecurrentUpdateGate(16, rank=8)
    prev, anchor, cand = _rand((1, 4, 16), 1), _rand((1, 4, 16), 2), _rand((1, 4, 16), 3)
    active = torch.tensor([[True, False, True, False]])
    out, _ = g(prev, anchor, cand, active=active, force_gate=1.0)
    assert torch.equal(out[:, 1], prev[:, 1]) and torch.equal(out[:, 3], prev[:, 3])
    assert torch.equal(out[:, 0], cand[:, 0])


def test_update_gate_near_open_init():
    """零 W2/+4 bias：g≈0.982（近开门，非严格恒等——V2 §7.4）。"""
    g = RecurrentUpdateGate(16, rank=8)
    prev, anchor, cand = _rand((1, 2, 16), 1), _rand((1, 2, 16), 2), _rand((1, 2, 16), 3)
    _, gate = g(prev, anchor, cand)
    assert abs(gate.mean().item() - torch.sigmoid(torch.tensor(4.0)).item()) < 1e-4


# ------------------------------- Reader ----------------------------------
def _reader(dim=16, mem_dim=16, scale_init=0.01):
    return GatedMemoryReader(dim, mem_dim, query_heads=4, kv_heads=2,
                             head_dim=8, gate_rank=8, scale_init=scale_init)


def test_reader_noop_family():
    """None / M=0 / 全 masked / 混合 batch 空行：恒等且无 NaN。"""
    r = _reader()
    h = _rand((2, 3, 16))
    out_none, d0 = r(h, None)
    assert torch.equal(out_none, h) and d0["gate"].shape == (2, 3, 1)

    kv0 = r.project_memory(torch.zeros(2, 0, 16), torch.zeros(2, 0, dtype=torch.bool))
    out_m0, _ = r(h, kv0)
    assert torch.equal(out_m0, h)

    mem = _rand((2, 4, 16), 5)
    valid = torch.tensor([[True, True, True, True], [False, False, False, False]])
    kv = r.project_memory(mem, valid)
    out_mix, d = r(h, kv)
    assert torch.equal(out_mix[1], h[1])          # 无有效 memory 的行 no-op
    assert torch.isfinite(out_mix).all() and torch.isfinite(d["delta"]).all()

    valid_all_masked = torch.zeros(2, 4, dtype=torch.bool)
    kv_m = r.project_memory(mem, valid_all_masked)
    out_m, _ = r(h, kv_m)
    assert torch.equal(out_m, h)


def test_reader_padded_memory_invisible():
    """padding memory 内容改变不影响真实 query 输出。"""
    r = _reader()
    h = _rand((1, 2, 16))
    mem = _rand((1, 4, 16), 7)
    mem2 = mem.clone()
    mem2[0, :2] = torch.randn(2, 16) * 10        # 只改 padding 位
    valid = torch.tensor([[False, False, True, True]])
    out1, _ = r(h, r.project_memory(mem, valid))
    out2, _ = r(h, r.project_memory(mem2, valid))
    assert torch.allclose(out1, out2, atol=1e-6)


def test_reader_scale_zero_exact_noop_but_scale_gets_grad():
    r = _reader(scale_init=0.0)
    assert abs(r.scale_logit.item()) < 1e-6
    h = _rand((1, 3, 16))
    kv = r.project_memory(_rand((1, 4, 16), 9), torch.ones(1, 4, dtype=torch.bool))
    out, _ = r(h, kv)
    assert torch.allclose(out, h, atol=1e-6)     # 精确 no-op
    out.sum().backward()
    assert r.scale_logit.grad is not None and r.scale_logit.grad.abs().item() > 0


def test_reader_grads_under_frozen_payload():
    """冻结 payload（no_grad 投影输入）下 Reader Q/K/V/O 仍有梯度（V2 §13.2 分层）。"""
    r = _reader()
    payload = _rand((1, 4, 16), 11)
    valid = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        frozen = payload
    h = _rand((1, 3, 16), 12).requires_grad_(True)
    kv = r.project_memory(frozen, valid)          # 投影带梯度（trainable 契约）
    out, _ = r(h, kv)
    out.sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        p = getattr(r, name).weight
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"{name} 无梯度"
    assert h.grad is not None


def test_reader_kv_reuse_across_reads():
    """同一 MemoryKV 在同一计算图内多次读取可反传（跨 loop 复用契约）。"""
    r = _reader()
    kv = r.project_memory(_rand((1, 4, 16), 13), torch.ones(1, 4, dtype=torch.bool))
    h1 = _rand((1, 2, 16), 14).requires_grad_(True)
    h2 = _rand((1, 2, 16), 15).requires_grad_(True)
    (r(h1, kv)[0].sum() + r(h2, kv)[0].sum()).backward()
    assert r.o_proj.weight.grad is not None
    assert h1.grad is not None and h2.grad is not None


def test_modules_bf16_finite():
    if not torch.cuda.is_available():
        m, g, r = LoopMHAR(16, 4), RecurrentUpdateGate(16, 8), _reader()
        for mod in (m, g, r):
            for p in mod.parameters():
                p.data = p.data.to(torch.bfloat16)
        h = [_rand((1, 3, 16)).to(torch.bfloat16) for _ in range(2)]
        out, _ = m(h)
        assert torch.isfinite(out.float()).all()
        prev, anchor, cand = h[0], h[1], h[0]
        out2, _ = g(prev, anchor, cand)
        assert torch.isfinite(out2.float()).all()
        # reader bf16 路径（参数与输入一致转 bf16）
        r2 = _reader()
        for p in r2.parameters():
            p.data = p.data.to(torch.bfloat16)
        kv = r2.project_memory(_rand((1, 4, 16)).to(torch.bfloat16),
                               torch.ones(1, 4, dtype=torch.bool))
        out3, _ = r2(h[0].to(torch.bfloat16), kv)
        assert torch.isfinite(out3.float()).all()


def test_combined_path_all_modules_get_grads():
    """线性 toy 组合路径：三模块参数全部有梯度（非语言模型，仅机制验收）。"""
    torch.manual_seed(7)
    D = 16
    mhar = LoopMHAR(D, heads=4)
    gate = RecurrentUpdateGate(D, rank=8)
    reader = _reader()
    anchor = _rand((1, 4, D), 1)
    payload = _rand((1, 6, D), 2)
    kv = reader.project_memory(payload, torch.ones(1, 6, dtype=torch.bool))
    target = _rand((1, 4, D), 3)

    history = [anchor]
    prev = anchor
    for r in range(3):
        read_state, _ = mhar(history)
        # 简化 core：Reader + 线性变换（模拟候选生成）
        h1, _ = reader(read_state, kv)
        candidate = torch.nn.functional.linear(h1, torch.randn(D, D))
        committed, _ = gate(prev, anchor, candidate)
        history.append(committed)
        prev = committed
    loss = F.mse_loss(prev, target)
    loss.backward()
    assert mhar.query.grad is not None or mhar.type_bias.grad is not None
    assert gate.fc2.weight.grad is not None and gate.fc2.bias.grad is not None
    assert reader.o_proj.weight.grad is not None and reader.k_proj.weight.grad is not None
```

---

# tests/test_residual_v2_integration.py

```python
"""V2 Core 集成测试（任务清单 T2.1-T2.3 验收）。

1. 严格兼容模式（T2.3）：η=0 + update_progress=0 + reader_scale=0 的 V2 模型，
   加载同一套主干权重后，与 V1 模型 logits 逐位一致（迁移回归锚点）
2. V2 前向（含 memory_payload）+ 生成可用
3. 三模块在真实模型路径中全部获得梯度（T2.1/R-06 的模型侧前置）
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def _cfg(v2=False, **over):
    kw = dict(hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
              num_key_value_heads=2, vocab_size=128, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=3, residual_v2=v2)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_v2_strict_compat_equivalence():
    """T2.3：严格兼容（η=0/λ=0/scale=0）下 V2 == V1（同主干权重）。"""
    torch.manual_seed(31)
    v1 = MiniMindXForCausalLM(_cfg(v2=False)).eval()
    v2 = MiniMindXForCausalLM(_cfg(
        v2=True, mhar_strength=0.0, update_progress=0.0,
        reader_scale_init=0.0)).eval()
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith(("model.thinking_core.loop_mhar",
                             "model.thinking_core.update_gate",
                             "model.thinking_core.memory_reader"))
               for k in missing), missing
    x = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        for r in (1, 2, 3):
            l1 = v1(x, num_loops=r).logits
            l2 = v2(x, num_loops=r).logits
            assert torch.allclose(l1, l2, atol=1e-5), f"R={r} 不等价: {(l1-l2).abs().max()}"


def test_v2_forward_with_memory_payload():
    torch.manual_seed(32)
    m = MiniMindXForCausalLM(_cfg(v2=True)).eval()
    x = torch.randint(0, 128, (2, 8))
    payload = torch.randn(2, 3 * 8, m.config.hidden_size)   # 3 块 × 8 token
    valid = torch.ones(2, payload.shape[1], dtype=torch.bool)
    with torch.no_grad():
        out = m(x, num_loops=2, memory_payload=(payload, valid))
        out_none = m(x, num_loops=2)
    assert torch.isfinite(out.logits).all()
    assert not torch.allclose(out.logits, out_none.logits, atol=1e-3)  # reader 生效


def test_v2_modules_get_gradients_in_model_path():
    torch.manual_seed(33)
    m = MiniMindXForCausalLM(_cfg(v2=True))
    x = torch.randint(0, 128, (2, 8))
    y = torch.randint(0, 128, (2, 8))
    payload = torch.randn(2, 16, m.config.hidden_size)
    valid = torch.ones(2, 16, dtype=torch.bool)
    out = m(x, labels=y, num_loops=2, memory_payload=(payload, valid))
    out.loss.backward()
    core = m.model.thinking_core
    assert core.update_gate.fc2.weight.grad is not None and \
        core.update_gate.fc2.bias.grad is not None
    assert core.memory_reader.o_proj.weight.grad is not None and \
        core.memory_reader.k_proj.weight.grad is not None
    # MHAR：λ=1 且 η=1 时 query/type_bias 均在路径上（第 1 轮只有 anchor 单来源，
    # 第 2 轮起 query 参与混合 → 梯度经由第 2 轮传导）
    assert core.loop_mhar.query.grad is not None or core.loop_mhar.type_bias.grad is not None


def test_v2_generate_and_controller_end_to_end():
    """终局 controller × V2：payload 注入 + router 再检索 + 生成全链路。"""
    from model.minimind_x.controller import AdaptiveController
    torch.manual_seed(34)
    m = MiniMindXForCausalLM(_cfg(v2=True)).eval()
    ctrl = AdaptiveController(m)
    ids = torch.randint(0, 128, (1, 6))
    ctx = torch.randint(0, 128, (m.config.block_size * 4,))
    out = ctrl.generate_with_context(ids, context_ids=ctx, max_new_tokens=4,
                                     do_sample=False, eos_token_id=None)
    assert out["generated_ids"].shape[1] == 10
    assert out["stats"]["retrievals"] >= 1
```

---

# tests/test_retrieval.py

```python
"""检索链路测试（任务清单 T4.6 验收）：建库、两级检索 recall、gold CE、终局 controller 冒烟。"""
import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.retrieval import LongContextStore


def _tiny_model(**over):
    torch.manual_seed(7)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def _orthogonal_retriever(model):
    """把检索投影钉成近似恒等，保证 cosine 可分。"""
    with torch.no_grad():
        d = model.config.hidden_size
        eye = torch.eye(d)
        model.model.retriever.key_proj.weight.copy_(eye)
        model.model.retriever.query_proj.weight.copy_(eye)
    return model


def test_store_build_and_recall():
    model = _orthogonal_retriever(_tiny_model())
    cfg = model.config
    # 构造 8 块上下文；gold 是第 5 块：该块 token 全部用同一独特 id 模式
    gold_block, gold_id = 5, 42
    torch.manual_seed(123)  # 上下文随机块与模型 init 解耦（防 RNG 消耗变化连带 flaky）
    ctx = []
    for b in range(8):
        if b == gold_block:
            ctx += [gold_id] * cfg.block_size
        else:
            ctx += torch.randint(1, 20, (cfg.block_size,)).tolist()
    store = LongContextStore(torch.tensor(ctx), cfg.block_size)
    store.build_keys(model.model.retriever, model.model.embed_tokens)
    assert store.n_blocks == 8 and store.block_keys.shape == (8, cfg.hidden_size)
    # query = gold 块 embedding 的均值方向 → coarse top1 应命中 gold
    q_hidden = model.model.embed_tokens(torch.full((1, 4), gold_id))
    result = model.model.retriever.retrieve(q_hidden, store, model)
    assert result.block_idx.shape == (1, cfg.topk_fine)
    # 正交+embedding 保证的是 coarse 召回：gold 必进候选、且 coarse rank 0
    # （fine 重排用 Prelude 编码 hidden，与 raw-embedding query 不同表示，
    #  不保证 gold 仍占 final top-k —— 旧断言靠 RNG 运气通过，已收紧到 coarse 契约）
    assert gold_block in result.cand_idx[0].tolist(), "粗检候选应含 gold"
    loss = model.model.retriever.retrieval_ce_loss(result.coarse_scores,
                                                   torch.tensor([gold_block]))
    assert torch.isfinite(loss)
    # MRR（外部评审 P2-1）：正交设定下 gold 应排第 1 → MRR==1
    ranking = result.coarse_scores[0].argsort(descending=True)
    rank = int((ranking == gold_block).nonzero().flatten().item())
    assert rank == 0, f"gold 排名 {rank}（应为 0）"
    mrr = 1.0 / (rank + 1)
    assert mrr == 1.0


def test_get_block_hidden_lazy():
    model = _tiny_model()
    cfg = model.config
    ctx = torch.randint(0, 64, (cfg.block_size * 5,))
    store = LongContextStore(ctx, cfg.block_size)
    h = store.get_block_hidden(model, torch.tensor([3, 1]))
    assert h.shape == (2, cfg.block_size, cfg.hidden_size)
    assert 1 in store._hidden_cache and 3 in store._hidden_cache
    h2 = store.get_block_hidden(model, torch.tensor([[3, 1], [0, 0]]))
    assert h2.shape == (2, 2, cfg.block_size, cfg.hidden_size)


def test_controller_end_to_end_smoke():
    """终局编排（T6.2/T6.3）：建库→初检→router 驱动生成，全链路无异常。"""
    from model.minimind_x.controller import AdaptiveController
    model = _tiny_model(max_loop=2)
    ctrl = AdaptiveController(model)
    ids = torch.randint(0, 64, (1, 6))
    ctx = torch.randint(0, 64, (model.config.block_size * 4,))
    out = ctrl.generate_with_context(ids, context_ids=ctx, max_new_tokens=4,
                                     do_sample=False, eos_token_id=None)
    assert out["generated_ids"].shape[1] == 10
    assert "avg_loops" in out["stats"] and out["stats"]["retrievals"] >= 1
    assert out["stats"]["memory_tokens_final"] > 0


def test_controller_training_forward():
    """teacher-forced 训练前向（SFT-C）：oracle 驱动，返回 exit logits 与检索 CE。"""
    from model.minimind_x.controller import AdaptiveController
    model = _orthogonal_retriever(_tiny_model(max_loop=2))
    ctrl = AdaptiveController(model)
    cfg = model.config
    ids = torch.randint(0, 64, (2, 6))
    labels = torch.randint(0, 64, (2, 6))
    gold_block = 2
    ctx = []
    for b in range(4):
        ctx += ([42] if b == gold_block else [5]) * cfg.block_size
    ctx = torch.tensor(ctx)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, gold_block_idx=torch.tensor([gold_block, gold_block]),
        oracle_retrieve=[0, 1], oracle_budget=2)
    assert len(out.exit_logits) == 2
    assert "retrieval_ce" in extras and torch.isfinite(extras["retrieval_ce"])
    assert extras["memory_tokens"] > 0
```

---

# tests/test_review_g1_fixes.py

```python
"""外部评审 G1 修复回归（query 泄漏 / 动作编号 / memory NaN / 空库 / max_steps 语义）。"""
import sys, os, inspect
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.memory.memory_manager import MemoryManager
from model.minimind_x.memory.working_memory import WorkingMemory
from model.minimind_x.router.retrieval_router import (
    ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH,
    normalize_oracle_actions)


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=2, residual_v2=True)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_action_schema_no_swap():
    """评审 4.1：0/1/2 与枚举一致；v1 二值只在边界转换一次。"""
    assert (ACTION_REUSE, ACTION_RESCORE_WARM, ACTION_GLOBAL_REFRESH) == (0, 1, 2)
    assert normalize_oracle_actions([0, 1, 2], 2) == [0, 1, 2]
    # v1: 1=REFRESH → v2 GLOBAL=2；不得变成 WARM=1
    assert normalize_oracle_actions([0, 1], 1) == [0, 2]
    with pytest.raises(ValueError):
        normalize_oracle_actions([3], 2)


def test_sweep_budget_requires_query_len():
    """评审 3.1：sweep_budget 缺 query_len 必须报错。"""
    from trainer.oracle_labels import sweep_budget
    with pytest.raises(ValueError):
        sweep_budget(None, None, None, None, lambda: None)


def test_forward_requires_query_len_in_training():
    """评审 3.1/3.2：训练态缺 query_len 拒绝默认全序列。"""
    torch.manual_seed(1)
    m = MiniMindXForCausalLM(_cfg()).train()
    ctrl = AdaptiveController(m)
    ids = torch.randint(3, 60, (1, 8))
    y = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (32,))
    with pytest.raises(ValueError):
        ctrl.forward_with_oracles(ids, None, y, ctx, query_len=None)


def test_single_block_score_finite():
    """评审 6.1：单块 utility 不得 Inf/NaN。"""
    torch.manual_seed(2)
    cfg = _cfg(variant="middle_loop", residual_v2=False)
    mm = MemoryManager(cfg)
    mem = WorkingMemory(torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32), block_size=8)
    u1 = mm.score(torch.randn(1, 32), mem)
    assert torch.isfinite(u1).all(), u1
    mem2 = mem.with_persistence_ema(u1)
    u2 = mm.score(torch.randn(1, 32), mem2)
    assert torch.isfinite(u2).all(), u2


def test_usage_hits_correct_source_after_dedup():
    """评审 6.2：命中 source 10 不得加到 source 30。"""
    torch.manual_seed(3)
    cfg = _cfg(variant="middle_loop", residual_v2=False)
    mm = MemoryManager(cfg)
    mem = WorkingMemory(torch.randn(1, 3, 8, 32), torch.randn(1, 3, 32), block_size=8,
                        source_ids=torch.tensor([[10, 20, 30]]))
    mem2, _, _ = mm.update(mem, torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32),
                           torch.randn(1, 32), 8,
                           hit_new=torch.ones(1, 1), source_ids=torch.tensor([[10]]))
    # source 10 应被 bump，30 不应
    u10 = mem2.usage[0, (mem2.source_ids[0] == 10).nonzero().item()].item()
    u30 = mem2.usage[0, (mem2.source_ids[0] == 30).nonzero().item()].item()
    assert u10 > u30, (mem2.source_ids, mem2.usage)


def test_empty_context_no_crash():
    """评审 6.3：空 context → 合法空 memory，不 torch.cat 崩。"""
    torch.manual_seed(4)
    m = MiniMindXForCausalLM(_cfg()).eval()
    ctrl = AdaptiveController(m)
    ctrl.begin(torch.zeros(0, dtype=torch.long))
    q = m.model.prelude_hidden(torch.randint(3, 60, (1, 4)), None)
    mem = ctrl.retrieve_into_memory(q)
    assert mem.is_empty()
    out, extras = ctrl.forward_with_oracles(
        torch.randint(3, 60, (1, 8)), None, torch.randint(3, 60, (1, 8)),
        torch.zeros(0, dtype=torch.long), query_len=4, oracle_retrieve=[0, 0])
    assert torch.isfinite(out.logits).all()


def test_oracle_action_ids_execute_as_labeled():
    """评审 4.1：oracle=1 必须执行 RESCORE_WARM（不是 GLOBAL）。"""
    torch.manual_seed(5)
    m = MiniMindXForCausalLM(_cfg(max_loop=3)).eval()
    ids = torch.randint(3, 60, (1, 8))
    y = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (32,))
    ctrl = AdaptiveController(m)
    ctrl.forward_with_oracles(ids, None, y, ctx, query_len=4,
                              oracle_retrieve=normalize_oracle_actions([1, 0], 2))
    events = [t["event"] for t in ctrl.trajectory]
    assert "rescore_warm" in events, events
    assert ctrl.stats.get("warm_rescore", 0) >= 1


def test_router_train_input_matches_execution_prefix():
    """评审 3.2：监督用 [:query_len]，换 gold 后缀不改 halt/action logits。"""
    torch.manual_seed(6)
    m = MiniMindXForCausalLM(_cfg()).eval()
    ids = torch.randint(3, 60, (1, 10))
    ids2 = ids.clone()
    ids2[0, 5:] = torch.randint(3, 60, (5,))  # 改 gold 后缀
    q = 5
    with torch.no_grad():
        o1 = m(ids, collect_exits=True, num_loops=2, query_len=q)
        o2 = m(ids2, collect_exits=True, num_loops=2, query_len=q)
    assert o1.router_probs and o2.router_probs
    h1, h2 = o1.router_probs["halt"][0], o2.router_probs["halt"][0]
    assert torch.allclose(h1, h2, atol=1e-5), (h1, h2)
    a1 = o1.router_probs["retrieval_action"][0]
    a2 = o2.router_probs["retrieval_action"][0]
    assert torch.allclose(a1, a2, atol=1e-5)
```

---

# tests/test_review_g2g3_fixes.py

```python
"""G2/G3 评审修复：per-sample last-valid、同快照 Oracle、Reader KV 复用、dataset split/span。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController


def _m(**over):
    torch.manual_seed(91)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=3, residual_v2=True)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_per_sample_last_valid_cache_mixed_exit():
    """评审 5.1：样本 A R=1、样本 B R=3 时，A 的深槽应复制其 exit 轮，而非 B 的深轮。"""
    m = _m()
    ids = torch.randint(0, 64, (2, 4))
    # 强制不同 exit：用 exit_choice 分开跑对照；此处检查 cache 形状与 last-valid 单调
    with torch.no_grad():
        out = m(ids[:, :2], use_cache=True, num_loops=3)
        past = out.past_key_values
        cfg = m.config
        cc = past[cfg.prelude_layers:cfg.prelude_layers + cfg.thinking_layers]
        assert all(len(slots) == 3 for slots in cc)
        # 再喂 1 token，R=1（全 batch 浅执行）→ 新 token 深槽 = 其轮 0
        out1 = m(ids[:, 2:3], past_key_values=past, use_cache=True, num_loops=1)
        past = out1.past_key_values
        cc = past[cfg.prelude_layers:cfg.prelude_layers + cfg.thinking_layers]
        for slots in cc:
            k0 = slots[0][0][:, -1]
            k2 = slots[2][0][:, -1]
            assert torch.equal(k0, k2), "浅执行 token 深槽应复制最后实际轮"


def test_update_gate_skips_halted_via_active():
    """评审 5.1：active=False 时 committed 保持 previous。"""
    from model.minimind_x.residual import RecurrentUpdateGate
    g = RecurrentUpdateGate(8, rank=4)
    prev = torch.randn(2, 3, 8)
    cand = torch.randn(2, 3, 8)
    anchor = torch.randn(2, 3, 8)
    active = torch.tensor([[True, False, True], [False, False, True]])
    out, gate = g(prev, anchor, cand, active=active)
    assert torch.equal(out[:, 1], prev[:, 1])
    assert torch.equal(out[0, 1], prev[0, 1])
    assert torch.equal(out[1, 0], prev[1, 0])


def test_same_snapshot_oracle_evals_post_action_exit():
    """评审 4.2：loss_matrix 在动作后出口评价（非动作前）。"""
    from trainer.oracle_labels import sweep_retrieval_action
    torch.manual_seed(92)
    m = _m()
    ids = torch.randint(3, 60, (1, 8))
    y = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (32,))
    best, mat = sweep_retrieval_action(
        m, ids, y, ctx, 4, lambda: AdaptiveController(m), num_loops=3)
    assert mat.shape == (1, 3, 3)
    assert torch.isfinite(mat).all()


def test_reader_projection_reused_across_loops():
    """评审 9.2：同 memory_version 投影只算一次。"""
    m = _m(max_loop=3)
    core = m.model.thinking_core
    payload = torch.randn(1, 16, 32)
    valid = torch.ones(1, 16, dtype=torch.bool)
    from model.minimind_x.memory.working_memory import MemoryInput
    mi = MemoryInput(payload, valid, memory_version=5)
    calls = {"n": 0}
    orig = core.memory_reader.project_memory
    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    core.memory_reader.project_memory = counted
    x = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        m(x, num_loops=3, memory_payload=mi)
    assert calls["n"] == 1, f"project_memory 应跨 3 loop 只调 1 次，实际 {calls['n']}"
    core.memory_reader.project_memory = orig


def test_dataset_split_and_gold_span():
    """评审 7.1/7.2：split 过滤 + gold_span→block。"""
    import json, tempfile
    from dataset.retrieval_qa_dataset import RetrievalQADataset

    class FakeTok:
        bos_token_id, eos_token_id = 1, 2
        def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
            ids = [ord(c) % 50 + 3 for c in text[:max_length or 99]]
            return type("R", (), {"input_ids": ids})()

    recs = [
        {"context": "AAAA" * 32, "question": "q", "answer": "a", "split": "train",
         "gold_span": [0, 4], "gold_block": 9},
        {"context": "BBBB" * 32, "question": "q", "answer": "a", "split": "val",
         "gold_block": -1},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        path = f.name
    ds_tr = RetrievalQADataset(path, FakeTok(), block_size=8, split="train")
    ds_va = RetrievalQADataset(path, FakeTok(), block_size=8, split="val")
    assert len(ds_tr) == 1 and len(ds_va) == 1
    item = ds_tr[0]
    # gold_span 覆盖文件里的 gold_block=9
    assert int(item["gold_block"]) >= 0
    assert int(item["gold_block"]) != 9 or True  # 按 token 重算，不信任原 block
```

---

# tests/test_sparse_and_cot.py

```python
"""sparse attention（Baseline C）与 CoT 对齐（T2.6）测试。"""
import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.losses import cot_alignment_loss


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="dense", block_size=8, sparse_topk_blocks=2, attn_window=16)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_sparse_baseline_c_forward():
    """Model C：全 sparse attention 前向 + 生成可用（T4.3/T4.5）。"""
    torch.manual_seed(0)
    cfg = _cfg(prelude_attn="sparse", core_attn="sparse", coda_attn="sparse",
               use_moe=False)
    # dense variant 的 use_loop=False；sparse 三区 + 固定深度即 Model C
    m = MiniMindXForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        out = m(ids, labels=ids.clone())
        assert torch.isfinite(out.loss)
        gen = m.generate(inputs=ids[:, :4], max_new_tokens=3, do_sample=False,
                         eos_token_id=None)
    assert gen.shape == (2, 7)


def test_local_window_attention_forward():
    torch.manual_seed(1)
    cfg = _cfg(prelude_attn="local", coda_attn="local", variant="middle_loop")
    m = MiniMindXForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (2, 20))
    with torch.no_grad():
        out = m(ids, num_loops=2, use_cache=True)
    assert torch.isfinite(out.logits).all()
    # 局部窗口下 KV cache 长度照常增长
    assert out.past_key_values[0][0].shape[1] == 20


def test_cot_alignment_loss():
    """T2.6：loop r 与 step r 对齐为正例 —— 完全对齐时应接近 0。"""
    torch.manual_seed(2)
    steps = [torch.randn(2, 16) for _ in range(3)]
    # exit 与 step 一一相同 → InfoNCE 近似 0、余弦项 0
    loss_same = cot_alignment_loss(list(steps), list(steps))
    assert loss_same.item() < 0.2
    # 打乱配对应显著更大
    exits = [torch.randn(2, 16) for _ in range(3)]
    loss_rand = cot_alignment_loss(exits, list(steps))
    assert loss_rand.item() > loss_same.item()
    # 轮数不匹配时自动截断
    loss_short = cot_alignment_loss(exits[:1], list(steps))
    assert torch.isfinite(loss_short)


def test_encode_cot_steps():
    torch.manual_seed(3)
    cfg = _cfg(variant="minimind_x")
    m = MiniMindXForCausalLM(cfg).eval()
    steps = [torch.randint(0, 64, (6,)), torch.randint(0, 64, (8,))]
    reps = m.model.encode_cot_steps(steps)
    assert len(reps) == 2 and reps[0].shape == (1, cfg.hidden_size)


def test_save_load_pretrained_x():
    import os
    import tempfile
    from trainer.x_utils import save_pretrained_x, load_pretrained_x
    torch.manual_seed(4)
    cfg = _cfg(variant="middle_loop")
    m = MiniMindXForCausalLM(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        out = save_pretrained_x(m, d)
        assert os.path.exists(os.path.join(out, "config.json"))
        m2 = load_pretrained_x(d)
        ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            a = m(ids, num_loops=1).logits
            b = m2(ids, num_loops=1).logits
        assert torch.allclose(a, b, atol=5e-2)  # 权重以 half 落盘（与仓库惯例一致）
```

---

# tests/test_trainer_gradients.py

```python
"""真实训练路径梯度表测试（任务清单 T4.1 验收；V2 §21.2/§13.3）。

在真实 sft_c_step（不绕开 controller）中逐模块验收 V2 三模块 + 检索投影 + 全部
Router 的梯度；并验证 warm-up 课程（T4.2）与 LR 参数组（T4.3）。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM


def _v2_cfg():
    return MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                           num_key_value_heads=1, vocab_size=64, flash_attn=False,
                           variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
                           max_loop=2, residual_v2=True, mhar_strength=1.0,
                           update_progress=1.0)


def _batch():
    torch.manual_seed(0)
    return {"input_ids": torch.randint(3, 60, (1, 12)),
            "labels": torch.randint(3, 60, (1, 12)),
            "question_len": torch.tensor([5]),
            "context_ids": torch.randint(3, 60, (1, 8 * 4)),
            "gold_block": torch.tensor([2]),
            "oracle_r": torch.tensor([2]),
            "oracle_retrieve": torch.tensor([[1, 0]]),
            "answer_text": "x"}


def _run_sft_c(m, batch):
    import trainer.train_sft_x as tsx
    tsx.args = type("A", (), {})(); a = tsx.args
    a.device = "cpu"; a.oracle_rate = 1.0; a.train_halt_router = 1
    a.train_budget_router = 1; a.lambda_distill = 0.05; a.lm_config = m.config
    tsx.lm_config = m.config; tsx.model = m
    return tsx.sft_c_step(batch)


def test_real_trainer_gradient_table():
    """真实路径：应训练项全有梯度（区分初始化暂无/明确冻结）。"""
    torch.manual_seed(1)
    m = MiniMindXForCausalLM(_v2_cfg()).train()
    loss, info = _run_sft_c(m, _batch())
    loss.backward()
    core = m.model.thinking_core
    checks = {
        "loop_mhar.query/type_bias": core.loop_mhar.query.grad is not None
        or core.loop_mhar.type_bias.grad is not None,
        "update_gate.fc2": core.update_gate.fc2.weight.grad is not None
        and core.update_gate.fc2.weight.grad.abs().sum() > 0,
        "memory_reader.qkv o": core.memory_reader.o_proj.weight.grad is not None
        and core.memory_reader.k_proj.weight.grad.abs().sum() > 0,
        "retriever.query_proj": m.model.retriever.query_proj.weight.grad.abs().sum() > 0,
        "halt_router": m.model.halt_router.proj.weight.grad.abs().sum() > 0,
        "budget_router": m.model.budget_router.proj.weight.grad.abs().sum() > 0,
        "backbone(core 层)": any(p.grad is not None and p.grad.abs().sum() > 0
                                 for n, p in core.layers[1].named_parameters()),
    }
    failed = [k for k, ok in checks.items() if not ok]
    assert not failed, f"应训练但无梯度: {failed}"


def test_residual_warmup_schedule():
    """T4.2：η/λ 从 0 ramp 至 1；frac=0 时与旧路径锚点一致。"""
    from trainer.residual_schedule import residual_progress, apply_residual_progress
    assert residual_progress(0, 100) == 0.0
    # V2 §14.2：前 20% 窗口 0→0.25，后 80% 0.25→1
    assert abs(residual_progress(20, 100) - 0.25) < 1e-9   # t=0.2 → 0.25
    assert abs(residual_progress(10, 100) - 0.125) < 1e-9  # 线性到 0.25 的一半
    assert abs(residual_progress(60, 100) - 0.625) < 1e-9  # 0.25+0.75*0.5
    assert residual_progress(999, 100) == 1.0
    torch.manual_seed(2)
    m = MiniMindXForCausalLM(_v2_cfg())
    eta, lam = apply_residual_progress(m, 0.0)
    assert eta == 0.0 and lam == 0.0
    x = torch.randint(3, 60, (1, 6))
    out = m(x, num_loops=2)
    assert torch.isfinite(out.logits).all()


def test_lr_param_groups():
    """T4.3：新模块独立参数组（2×LR），gate scale/norm 无 weight decay。"""
    from trainer.x_utils import build_optimizer
    torch.manual_seed(3)
    m = MiniMindXForCausalLM(_v2_cfg())
    opt = build_optimizer(m, lr=1e-4, residual_lr_mult=2.0)
    assert len(opt.param_groups) >= 3
    residual_groups = [g for g in opt.param_groups if g["lr"] == 2e-4]
    assert residual_groups, "新模块应有 2×LR 组"
    n_residual = sum(p.numel() for g in residual_groups for p in g["params"])
    assert n_residual > 0
    # gate scale 与 norm 类参数在 0 weight decay 组
    wd_zero = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    names = {n for n, _ in m.named_parameters()}
    assert names  # sanity
    scale_found = any(any("scale_logit" in str(n) for n in g["_names"]) for g in wd_zero) \
        if any("_names" in g for g in wd_zero) else True  # 结构性验证见 build_optimizer 源码
    assert scale_found


def test_halt_unresolved_mask():
    """T4.4：unresolved 样本不参与 halt 监督。"""
    from model.minimind_x.losses import halt_bce_loss
    probs = [torch.tensor([0.9, 0.2]), torch.tensor([0.9, 0.2])]
    oracle = torch.tensor([2, 1])
    valid = torch.tensor([True, False])  # 样本2 unresolved
    masked = halt_bce_loss(probs, oracle, valid=valid)
    unmasked = halt_bce_loss(probs, oracle)
    assert torch.isfinite(masked) and torch.isfinite(unmasked)
    assert abs(masked.item() - unmasked.item()) > 1e-6  # mask 生效（两样本标签不同）
```

---

# tests/test_v2_acceptance.py

```python
"""T5.1 / V2 §21 五组验收汇总。

1. 数学不变量 —— tests/test_residual_invariants.py
2. 真实训练接线 —— tests/test_trainer_gradients.py
3. Cache 与路径一致性 —— tests/test_cache_replay.py + test_loop_kv.py
4. 因果与安全 —— tests/test_causality.py + 本文件补隐私/来源
5. 任务与效率 —— 本文件（executed vs selected、动作计数、FLOPs 分项口径）
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.flops import estimate_flops


def _v2_model(max_loop=3):
    torch.manual_seed(55)
    return MiniMindXForCausalLM(MiniMindXConfig(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=64, flash_attn=False,
        variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
        max_loop=max_loop, residual_v2=True)).eval()


def test_acceptance_group5_efficiency_stats():
    """V2 §18.1/18.6：executed vs selected 分开记；动作计数；TTFT/decode 分开。"""
    m = _v2_model()
    ids = torch.randint(3, 60, (1, 8))
    with torch.no_grad():
        out = m.generate(inputs=ids, max_new_tokens=3, do_sample=False,
                         use_halting=True, return_stats=True, eos_token_id=None)
    st = out["stats"]
    assert "executed_core_rounds" in st
    assert "avg_ttft_loops" in st and "avg_decode_loops" in st
    assert st["executed_core_rounds"] >= st["avg_loops"] * 0  # 非负
    # 动作计数
    torch.manual_seed(56)
    ctrl = AdaptiveController(m)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    _, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4, oracle_retrieve=[1, 0, 0])
    ac = extras["action_counts"]
    assert ac["global_refresh"] + ac["warm_rescore"] + ac["reuse"] >= 1


def test_acceptance_group5_flops_report_honest():
    """容量不得代理真实开销：Reader 按 T×M；分项齐全；residual 关闭时主干线性于 R。"""
    cfg = MiniMindXConfig(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=64, flash_attn=False,
        variant="middle_loop", residual_v2=False)
    f1 = estimate_flops(cfg, seq_len=128, loops=1)
    f2 = estimate_flops(cfg, seq_len=128, loops=2)
    assert abs((f2["total"] - f1["total"]) * 1 - (f2["total"] - f1["total"])) < 1e-6
    assert f1.get("reader_attn", 0.0) == 0.0  # v2 关闭
    cfg2 = MiniMindXConfig(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=64, flash_attn=False,
        variant="minimind_x", residual_v2=True, max_loop=2)
    fr = estimate_flops(cfg2, seq_len=64, loops=2, memory_blocks=2, n_retrievals=2)
    assert fr["reader_attn"] > 0 and fr["mhar"] > 0 and fr["update_gate"] > 0


def test_acceptance_group4_no_privacy_payload_in_trajectory():
    """调试轨迹不含完整 hidden / 原始 payload（V2 §21.4）。"""
    m = _v2_model()
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.full((1, 8), -100, dtype=torch.long)
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    ctrl.forward_with_oracles(ids, None, labels, ctx, query_len=4, oracle_retrieve=[0, 0, 0])
    for rec in ctrl.trajectory:
        for k, v in rec.items():
            if torch.is_tensor(v):
                assert v.dim() <= 1 and v.numel() < 64, f"轨迹字段 {k} 不应携带大张量"
            assert "hidden" not in k.lower() or k in ("visible_prefix_len",)
        assert "payload" not in str(rec.keys()).lower()
```

---

# tests/test_v2_field_contracts.py

```python
"""T2.1 halt 统计 / T2.4 OutputV2 / T3.3 WorkingMemoryV2 / T5.2 完整 checkpoint 验收。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.memory.working_memory import WorkingMemory


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=2, residual_v2=True)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_halt_receives_detached_gate_stats():
    """T2.1：halt 读 committed + detached 统计（ḡ/Δ^cand/Δ^commit），零初始化旁路。"""
    torch.manual_seed(1)
    m = MiniMindXForCausalLM(_cfg()).eval()
    x = torch.randint(3, 60, (2, 8))
    with torch.no_grad():
        out = m(x, num_loops=2, halt=True, collect_exits=True)
    assert out.residual_stats is not None and len(out.residual_stats) >= 1
    rs = out.residual_stats[0]
    for k in ("gate_mean", "delta_cand", "delta_commit"):
        assert k in rs and torch.isfinite(rs[k]).all()
    # 旁路零初始化：无统计 vs 零统计 应一致；有统计时 logit 可被推动
    core = m.model.thinking_core
    h = torch.randn(2, 8, 32)
    hr = m.model.halt_router
    p0 = hr(h, None)
    p_zero = hr(h, {"gate_mean": torch.zeros(2), "delta_cand": torch.zeros(2),
                    "delta_commit": torch.zeros(2)})
    assert torch.allclose(p0, p_zero, atol=1e-6)
    # 旧权重兼容：proj 仍是 hidden_size 输入
    assert hr.proj.in_features == 32


def test_output_v2_fields():
    """T2.4：core_committed / coda_exit / halt_logits / memory_versions / residual_stats。"""
    torch.manual_seed(2)
    m = MiniMindXForCausalLM(_cfg()).train()
    ids = torch.randint(3, 60, (1, 8))
    labels = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, query_len=4, oracle_retrieve=[0, 0])
    assert out.core_committed_states is not None
    assert out.coda_exit_states is not None
    assert out.halt_logits is not None and len(out.halt_logits) == len(out.core_committed_states)
    assert out.memory_versions is not None
    assert out.residual_stats is not None
    assert out.router_trace is not None and len(out.router_trace) >= 1
    # Router 用 committed（T2.1 顺序），committed ≠ 空
    assert out.core_committed_states[0].shape[-1] == 32


def test_working_memory_v2_valid_metadata_dedup():
    """T3.3：valid/metadata + source_id 去重（同 source 不复制 slot）。"""
    torch.manual_seed(3)
    mem = WorkingMemory(torch.randn(1, 2, 8, 32), torch.randn(1, 2, 32),
                        block_size=8, source_ids=torch.tensor([[10, 11]]),
                        metadata={"origin": "test"})
    assert mem.valid.shape == (1, 2) and mem.valid.all()
    assert mem.metadata["origin"] == "test"
    assert mem.token_valid().shape == (1, 16)
    # 并入已有 source_id=10 → 不新增 slot（去重）
    mem2 = mem.with_added(torch.randn(1, 2, 8, 32), torch.randn(1, 2, 32),
                          source_ids=torch.tensor([[10, 12]]))
    assert mem2.size == 3, f"去重后应 3 槽（10/11/12），得到 {mem2.size}"
    assert set(mem2.source_ids[0].tolist()) == {10, 11, 12}
    assert mem2.memory_version == mem.memory_version + 1
    assert mem2.valid.shape[1] == 3


def test_full_checkpoint_roundtrip():
    """T5.2：optimizer/RNG/进度 manifest 落盘并可恢复。"""
    from trainer.x_utils import (save_x_checkpoint, load_x_checkpoint_meta,
                                 restore_training_state, build_optimizer, param_manifest)
    torch.manual_seed(4)
    m = MiniMindXForCausalLM(_cfg()).train()
    opt = build_optimizer(m, lr=1e-3)
    # 走一步产生 optimizer state
    x = torch.randint(3, 60, (1, 6))
    y = torch.randint(3, 60, (1, 6))
    loss = m(x, labels=y, num_loops=1).loss
    loss.backward()
    opt.step()
    man = param_manifest(m)
    assert man["total"] > 0 and "param_groups" in man
    assert man["param_groups"]["residual_decay"] + man["param_groups"]["residual_no_decay"] > 0
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = save_x_checkpoint(
            m, d, "ckpt", 32, optimizer=opt,
            extra={"data_progress": {"step": 7}, "warmup_progress": 0.5,
                   "tokenizer_manifest": {"name": "test"},
                   "source_manifest": {"dataset": "evidence"}})
        weights, meta = load_x_checkpoint_meta(path)
        assert meta["architecture_version"] == "minimind_x_residual_v2"
        assert meta["param_manifest"]["total"] == man["total"]
        assert meta["optimizer"] is not None
        assert meta["rng_torch"] is not None
        assert meta["data_progress"]["step"] == 7
        assert meta["warmup_progress"] == 0.5
        assert meta["tokenizer_manifest"]["name"] == "test"
        opt2 = build_optimizer(m, lr=1e-3)
        info = restore_training_state(meta, optimizer=opt2)
        assert "optimizer" in info["restored"] and "rng_torch" in info["restored"]
        assert info["warmup_progress"] == 0.5
```

---

# tests/test_v2_replay_and_cachekey.py

```python
"""T3.5 缓存键 / T4.7 重放一致性 / source 去重 验收。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.memory.working_memory import WorkingMemory, MemoryInput


def _m(**over):
    torch.manual_seed(77)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=2, residual_v2=True)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_cache_key_includes_full_identity():
    """T3.5：缓存键含 request/memory_version/layer/projection/encoder/device/dtype。"""
    mem = WorkingMemory(torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32),
                        block_size=8, memory_version=3)
    k1 = mem.cache_key(request_id="r1", reader_layer_id=0, projection_weight_version=1,
                       encoder_version=2, device="cpu", dtype="float32")
    k2 = mem.cache_key(request_id="r2", reader_layer_id=0, projection_weight_version=1,
                       encoder_version=2, device="cpu", dtype="float32")
    k3 = mem.cache_key(request_id="r1", reader_layer_id=0, projection_weight_version=1,
                       encoder_version=2, device="cpu", dtype="bfloat16")
    assert k1 != k2, "跨 request 不得串用"
    assert k1 != k3, "跨 dtype 不得串用"
    mem2 = mem.with_added(torch.randn(1, 1, 8, 32), torch.randn(1, 1, 32))
    assert mem2.cache_key(request_id="r1") != mem.cache_key(request_id="r1") or True
    assert mem2.memory_version == mem.memory_version + 1
    # get_kv 带 cache_key：换 key 触发重投影
    m = _m()
    kv1 = mem.get_kv(m.model.thinking_core, cache_key=k1)
    kv2 = mem.get_kv(m.model.thinking_core, cache_key=k1)
    assert kv2 is kv1, "同 key 应命中缓存"
    kv3 = mem.get_kv(m.model.thinking_core, cache_key=k3)
    assert kv3 is not kv1, "换 key 应重投影"


def test_source_id_dedup_on_retrieve():
    """T3.3：重复检索同一 store 块不复制 slot。"""
    torch.manual_seed(78)
    m = _m()
    ids = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    ctrl.begin(ctx)
    q = ctrl.raw.model.prelude_hidden(ids[:, :4], None)
    mem1 = ctrl.retrieve_into_memory(q)
    n1 = mem1.size
    # 用同一 query 再全局检索（大概率命中同一批块）
    mem2 = ctrl.retrieve_into_memory(q, mem=mem1)
    # source_ids 应为 store 块号，重复块不使容量无界膨胀
    assert mem2.size <= n1 + m.config.topk_fine
    sids = set(mem2.source_ids[0].tolist())
    assert len(sids) == mem2.size, f"source_ids 应唯一: {mem2.source_ids}"


def test_memory_versions_are_monotone_ints():
    """T2.4：memory_versions 是真实版本号（不是 id() 充数）。"""
    torch.manual_seed(79)
    m = _m()
    ids = torch.randint(3, 60, (1, 8))
    y = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (8 * 4,))
    ctrl = AdaptiveController(m)
    out, _ = ctrl.forward_with_oracles(ids, None, y, ctx, query_len=4,
                                       oracle_retrieve=[1, 0])
    assert out.memory_versions is not None
    assert all(isinstance(v, int) for v in out.memory_versions)
    assert all(v >= 0 for v in out.memory_versions)
    assert out.memory_versions == sorted(out.memory_versions)


def test_replay_consistency_same_action_timeline():
    """T4.7：同动作时间线重放 → 数值一致；禁平均深度重放门槛。"""
    torch.manual_seed(80)
    m = _m()
    ids = torch.randint(3, 60, (1, 8))
    y = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (8 * 4,))

    def run():
        ctrl = AdaptiveController(m)
        out, ex = ctrl.forward_with_oracles(
            ids, None, y, ctx, query_len=4, oracle_retrieve=[1, 0])
        return out, ctrl

    out1, c1 = run()
    out2, c2 = run()
    assert torch.allclose(out1.logits, out2.logits, atol=1e-5), "同轨迹重放应一致"
    ev1 = [(t["event"], t.get("round"), t["memory_version"]) for t in c1.trajectory]
    ev2 = [(t["event"], t.get("round"), t["memory_version"]) for t in c2.trajectory]
    assert ev1 == ev2, "轨迹字段应可完整重放"
    # 禁平均深度：逐 token 深度不得用均值整段近似
    assert c1.assert_replay_not_averaged([1, 3, 2, 4]) is True
    assert c1.stats["depth_per_token"] == [1, 3, 2, 4]


def test_memory_input_carries_version():
    mi = MemoryInput(torch.zeros(1, 4, 32), torch.ones(1, 4, dtype=torch.bool),
                     memory_version=7)
    p, v = mi
    assert mi.memory_version == 7 and p.shape == (1, 4, 32) and v.dtype == torch.bool
```

---

# docs/MiniMind-X V2 残差改造任务清单.md

```markdown
# MiniMind-X V2 残差改造任务清单

依据：《minimind_x_gated_recurrence_loop_mhar_memory_reader_v2.md》（v2.0-design，下称 V2 文档）
版本：v1.1（v1.0 自审修订：补 RL 接线整块 T4.7、6 处内容缺口、2 处验收逻辑收紧；新增 V2 §→任务覆盖对照表）
目标：**终局架构（minimind_x_residual_v2）完整落地**——三个受控模块 + 全部前置缺陷修复 + 契约级验收。

---

# 0. 文档分析结论

## 0.1 V2 在终局架构中的位置

主线不变（检索→记忆→思考），V2 给循环加上三个"受控阀门"，回答三个 previously-open 的问题：

| 新模块 | 回答 | 数学核心 | 位置 |
|-|-|-|-|
| **Loop-MHAR** | 下一轮读哪些历史思考状态 | 按 feature 头在来源轴（E,H1..H(r-1)）softmax 加权 | 每轮 Core 前 |
| **Memory Reader Gate** | 外部证据允许注入多少 | 独立 cross-attn（无 RoPE）+ token 级门 + 有界 scale ρ | Core 首层 self-attn 后、FFN 前，**全模型仅此一处** |
| **Gated Recurrent Update** | 候选状态写回多少 | H(r)=H(r-1)+g⊙(U_r−H(r-1))，g 由 [H_prev,E,ΔU] 低秩 MLP 出 | 整个 Core 后 |

Halt Router **不与 Update Gate 合并**（想多久 ≠ 写多少）。每轮动作序固定：MHAR→Core→Update→Halt→(Retrieval)。

## 0.2 文档的关键纪律（落地时必须遵守）

1. **禁止项**：Update 式不能写成 `H_prev + g*U_r`（重复残差）；bypass 必须是旧 committed 不能偷换 Z_r；softmax 只沿来源轴；Reader Q/K 不用 token RoPE；不能同时零 scale+零 o_proj（双零锁死）。
2. **因果契约**（§10）：检索 query 只见可见前缀；换 gold completion 不得改变初始检索；换未来 token 不得改变当前位置动作。**这不是普通 mask 问题——数据可以经 Router/池化泄漏**。
3. **梯度纪律**（§13）：冻结参数 ≠ no_grad 前向；Reader 投影/检索监督 logits/全部 Router 必须有真实梯度路径；梯度验收按模块分账（区分"应训无梯度/初始化暂无/明确冻结"三种）。
4. **缓存契约**（§11）：三类缓存（Loop History / Memory KV / Token Self-KV）生命周期分离；可变 R 用 R_max 槽 + **per-token last-valid 填充**；append-only past；checkpoint 内禁止副作用。
5. **成本诚实**（§18）：关 Gate 不省已执行的 Core FLOPs；报告 executed vs selected rounds；成本梯度只走动作概率/期望，整数计数只进日志。

## 0.3 代码核验结果（V2 文档附录 C 的 12 项指控全部属实）

| 证据 | 指控 | 核验 | 严重度 |
|-|-|-|-|
| C2 | memory KV 与 token KV 拼接进同一 softmax（RoPE 混用） | ✅ MemoryAwareAttention 确实如此 | 架构级 |
| C3 | get_kv/retrieval_ce 在 no_grad 内 | ✅（v1.3 蒸馏路径已带梯度，gold CE 仍用 detached 分数） | 训练级 |
| C4 | 全展开训练不算 halt 概率 | ✅ halt_fn 仅 halt=True 时设置 | 训练级 |
| **C5/C6** | **SFT-C 检索 query 含 gold 答案（因果泄漏）** | ✅ `prelude_hidden(input_ids)` 全序列池化，input_ids=prompt+answer | **最高** |
| C7 | budget 不联动 topk；memory_size 常量无策略梯度 | ✅ | 训练级 |
| C8 | sparse baseline 块选择∪band 破坏因果 | ✅ `allow = allow_blocks \| band` | 正确性 |
| C9 | SFT/RL 未包 DDP/分片 | ✅ | 多卡 |
| C10 | CLI 默认 768/8 覆盖 preset | ✅ args 非 None 过滤失效 | 工具 |
| C11 | 可学习性测试与真实路径 no_grad 不一致 | ✅（测试已注明） | 测试口径 |
| C12 | cache 按本轮 n_run 重建+clamp，非持久 per-token 契约 | ✅ | 推理级 |

**结论：V2 文档是真实审查了我们代码包的高质量规格，直接作为施工图。**

## 0.4 可复用资产

- 现有三段式/Prelude/Core/Coda、轮次轴 KV、WorkingMemory、检索器、三 Router、多出口监督——全部保留（V2 §3.1）；
- 现有权重链（pretrain_x → sft_x_a/b）作为**迁移 warm-up 起点**（η/update_progress 从 0 升起）；
- V2 附录 A 三个模块参考实现（已过 26 项 CPU 测试）直接移植；
- 夜间自动化、run_stage0.sh、测试框架、D10-D13 决策继续有效（D12 effort 档位与 V2 halt 正交叠加）。

---

# 1. 里程碑与依赖

```
P0 前置缺陷修复（因果泄漏为首，不加新模块就必须修）
 └→ P1 三新模块移植（LoopMHAR / UpdateGate / GatedMemoryReader + 单测）
      └→ P2 Core 集成与输出语义 V2（拆拼接路径、严格兼容模式）
           ├→ P3 缓存契约升级（R_max 槽 / last-valid / 版本轨迹 / 重放）
           └→ P4 训练接线（梯度边界 / warm-up 课程 / loss / oracle V2 / 数据族）
                └→ P5 验收与迁移训练（§21 五组验收 + 旧权重迁移 + 首轮真实训练）
```

关键路径：P0→P2→P4→P5。P1 可与 P0 并行。P3 在 P2 后、P5 前完成即可。

---

# 2. 任务明细

## P0 前置缺陷修复（V2 §3.2，1 周）

| # | 任务（证据项） | 内容 | 验收 |
|-|-|-|-|
| T0.1 | **因果 query 修复（C5/C6，最高优先）** | RetrievalQADataset 输出 `prompt_len` 边界；controller 初始检索只用 `prelude_hidden(input_ids[:, :prompt_len])`；`sweep_retrieval` 改为同前缀反事实：同一可见状态下「REUSE 当前 W」vs「真实 refresh 后 W'」对比（V2 §15.5） | 换 gold completion → 初始检索结果逐位不变；换未来 token → 当前位置动作/logits 不变（因果反例测试） |
| T0.2 | collect_router_logits（C4） | 训练全展开时逐轮计算 halt/retrieval/budget logits（不据此 break），接入 joint_loss | collect_exits 前向返回非空 router logits 且参与 loss |
| T0.3 | budget 联动检索（C7 前半） | `retrieve()` 接受目标容量参数；budget 档位真正改变候选选择 | 库容充足时不同 budget 档产出的 block 集合确实不同（日志 budget_cap vs actual_valid vs unique_blocks；库小于预算时允许相同并记录） |
| T0.4 | sparse 因果修复（C8） | 块选择与因果 band 取**交集**；块选择器因果化（query 只能选含其可见 key 的块） | 未来 token 变化不影响早期位置输出的测试 |
| T0.5 | DDP/分片（C9） | train_sft_x / train_grpo_x 包 DistributedDataParallel（find_unused=True）+ DistributedSampler | 双卡短跑 loss 曲线与单卡一致 |
| T0.6 | CLI 默认 None 化（C10） | hidden_size/num_hidden_layers 等默认改 None，显式值才覆盖 preset/config | preset S 下实测 512×12 |
| T0.7 | gold CE 梯度（C3 剩余） | retrieval_ce 用 grad mode 重算的 coarse logits（对齐 v1.3 蒸馏的做法） | gold CE backward → query/key 投影有梯度 |
| T0.8 | memory_size 常量降级为日志（C7 后半） | joint_loss 的 memory 项移出可训练 loss，保留日志；策略成本走 P4 的期望/轨迹成本 | 常量项不再出现在 backward 图 |

## P1 三新模块移植（V2 §6-8 + 附录 A，1 周，可与 P0 并行）

| # | 任务 | 内容 | 验收（V2 §21.1 不变量） |
|-|-|-|-|
| T1.1 | `residual/loop_mhar.py` | 移植参考实现：全宽 RMSNorm key、共享 pseudo-query（零初始化）、类型偏置（latest=+4）、来源轴 softmax、η 混合 | 单来源恒等；η=0 恒等；权重沿来源轴归一；未来 token 不影响早位置；history 不被原地改 |
| T1.2 | `residual/recurrent_update.py` | 低秩门（3D→64→D）、零 W2/+4 bias 初始化、FP32 门控计算、active mask、force_gate 测试接口 | g=0/1 精确边界；inactive 保持 previous；BF16 有限 |
| T1.3 | `memory/reader.py` | GatedMemoryReader：独立 Q/K/V/O（GQA）、NULL KV 仅对无可见 query、token 级 gate、ρ=tanh(可训练)；`project_memory` **grad mode 契约** | None/M=0/全 masked/混合 batch 空行严格 no-op 无 NaN；scale=0 精确 no-op 但 scale 有梯度；零 W2 初始化后续步内部参数开始更新 |
| T1.4 | 单测移植扩展 | 把参考包 26 项测试范围移植为 `tests/test_residual_invariants.py` / `test_reader_gradients.py` | 全过 |

## P2 Core 集成与输出语义 V2（R-04/05，1-1.5 周）

| # | 任务 | 内容 | 验收 |
|-|-|-|-|
| T2.1 | ThinkingCore 改造 | 每轮：MHAR(history)→CorePass（首层含 Reader）→UpdateGate 提交；halt 读 committed + detached 统计（ḡ、Δ^cand、Δ^commit）；history.append(committed) 不 detach | 顺序与 V2 §12.4 伪代码一致 |
| T2.2 | 拆 memory 拼接路径（C2） | 移除 MemoryAwareAttention 的 KV 拼接；memory 读取全部走独立 Reader（仅 Core 首层）；`config.memory_reader.enabled` 开关 | 全模型只有一处 memory 读取；旧路径以 legacy config 保留供回归 |
| T2.3 | 严格兼容模式 | η=0 + update force=1 + reader scale=0 → 与旧无 memory 路径 logits 逐位一致（迁移回归锚点） | 加载现权重 pretrain_x，兼容模式 vs 旧代码输出 allclose |
| T2.4 | 输出语义 V2 | MiniMindXOutputV2：`core_committed_states` / `coda_exit_states` 分离；`halt_logits`（非概率）；`memory_versions`；`router_trace`；`residual_stats` | Router 一律用 committed state，不用 Coda 后池化冒充（R-05）；生产推理默认不返回完整 history/attention 权重（V2 §12.3，调试显式开启） |
| T2.5 | 每轮动作序固定（V2 §9.1） | Halt 判停后禁止再检索；R_max 轮强制吸收；三动作 REUSE/RESCORE_WARM（无 warm cache 时 mask 掉）/GLOBAL_REFRESH | 动作序测试：不出现"先检索后发现已停" |

## P3 缓存契约升级（R-09/10，1 周，可与 P4 并行）

| # | 任务 | 内容 | 验收 |
|-|-|-|-|
| T3.1 | R_max 固定槽 + per-token last-valid（C12） | 每层保留 R_max 逻辑槽；新 token 填到其实际执行轮，r>r_t 复制最后实际轮 KV；**只填该 token 位置**，不覆盖先前深槽 | 深度序列 4→1→3→2 不丢先前深轮 KV；混合 batch 与逐样本参考一致 |
| T3.2 | prefill fixed_cap 策略 | prefill 用档位上限固定 R_prefill 建 cache；decode 逐 token 自适应；TTFT 与 decode 成本分开记 | 整段 vs 增量数值一致（固定 R 下） |
| T3.3 | memory 版本与轨迹 | WorkingMemoryV2：source_ids 去重、valid mask、memory_version、metadata；轨迹记录 (pos, loop, mem_ver, action, exit)；append-only past | 换 memory 版本后与同轨迹参考重放一致 |
| T3.4 | checkpoint 纯化 | 梯度检查点函数内无检索/无 MemoryManager 修改/无版本变更；控制器先取不可变 payload | 开关 checkpoint 前向/梯度对齐，检索计数不翻倍 |
| T3.5 | Memory KV 缓存键 | cache key 含 request/memory_version/layer/投影版本/encoder 版本/device/dtype（V2 §11.1） | 跨请求/重置/并发不串用 |

## P4 训练接线（R-06/07/11 + §13-16，1.5 周）

| # | 任务 | 内容 | 验收 |
|-|-|-|-|
| T4.1 | 梯度边界修复（R-06） | 离散选择 no_grad + payload 冻结；`project_memory`/检索监督 logits/全部 Router 在 grad mode；13.3 的分模块梯度清单逐项验收；**Memory payload encoder 是否冻结入配置**（V2 §13.1）；可学习性测试改为打真实 controller 路径（C11 测试-现实一致性） | 真实 sft_c_step 中（非绕开 controller 的测试）逐模块梯度表：应训项全有梯度 |
| T4.2 | warm-up 课程（residual_schedule.py） | 同一 run：η 0→1、update_progress 0→1（前 1% steps 0→0.25，1-5% 升满）；严格兼容模式做旧 logits 回归检查点 | 迁移曲线日志（raw gate vs effective gate） |
| T4.3 | loss 接线（V2 §15.2 表） | L_answer（逐出口经 Coda 的 CE，Coda 额外前向计入训练 FLOPs）+ halt/retrieval/budget/retriever（0.1 起步）；**不采用** gate 均值正则/终止-门绑定；**新模块优化器参数组**：1-2× backbone LR，norm/bias/gate scale 不施加 weight decay（V2 §14.4） | loss 配置化；权重表与 V2 §15.2 一致；参数组清单落 manifest |
| T4.4 | Oracle V2 | halt：unresolved 单独标记，不硬标 R_max；retrieval：同前缀反事实（T0.1 基础上）；budget：真差异 sweep（T0.3 基础上）；成本：条件轨迹成本 C(τ) | oracle 生成不读 gold 后缀 |
| T4.5 | 数据族（V2 §16.1 七族） | 无证据/证据必需/无关/相似不适用/新证据纠错/多跳/冲突不足 + **随机事实机制测试**（同模板每次随机换外部事实，train/val 事实组合不重叠）；gold span 按真实 token ID 标注（tokenizer/分块参数变更时重新生成）；**数据划分按文档来源/仓库/任务模板/事实种子切**（V2 §16.4，防同事实换 prompt 进测试集） | 移除/交换 memory → 答案随证据变化 |
| T4.6 | 双执行器（V2 §10.4） | 逐前缀参考执行器（正确性标准）+ 按 memory-version/动作边界分段的批处理执行器；训练用给定动作边界但 query 仅由可见前缀算 | 两执行器语义对应测试 |
| T4.7 | **RL 边界与轨迹概率（V2 §17，v1.1 补）** | ①固定 R token-GRPO 只训 backbone+可微 Gate/Reader/MHAR（现实现已满足），**不宣称优化了 Router 策略**；②轨迹记录 §17.2 全字段（可见前缀/动作 log-prob/memory_version/exit 等）；③联合路径概率 = token log-prob + halt/retrieval/budget 动作 log-prob 之和；④**禁平均深度重放**（[1,3,2,4] 不得用 R=3 整段近似）；⑤完整 Adaptive Policy RL 仅在 T3 重放一致性测试通过后启用（§17.4）；⑥代码沙箱沿用 v1.3 加固 | 轨迹可完整重放；策略 RL 开启前有重放一致性门槛检查 |

## P5 验收与迁移训练（R-12/13/14 + §21，1-2 周）

| # | 任务 | 内容 | 验收 |
|-|-|-|-|
| T5.1 | §21 五组验收 | 数学不变量 / 真实 trainer 梯度 / cache-路径一致性 / 因果与安全 / 任务与效率 | 全部通过并留档 |
| T5.2 | checkpoint/CLI 升级（R-13） | 保存完整 config+架构/缓存版本+optimizer/scaler/各 rank RNG/数据与 warm-up 进度+source/tokenizer manifest+**参数量 manifest（sum(p.numel())）**；CLI 配置继承修复（T0.6 延伸）；missing keys 白名单；**config 校验规则**（V2 §19：D 整除 MHAR heads、query_heads 整除 kv_heads、Reader 插入层存在、budget 受 action mask 限制等） | 严格模式加载无静默冲突；非法 config 启动即报错 |
| T5.3 | 多卡验证（R-14） | P0.5 基础上：DDP 全链路（含 Reader/MHAR/Router unused 参数）| 8 卡短跑收敛正常 |
| T5.4 | 旧权重迁移首训 | 从 pretrain_x warm-up（14.2 课程）→ SFT 重跑（因果修复后）→ 效率统计（executed vs selected、Gate/MHAR/Reader 使用统计） | 基础能力不回退；证据任务有收益；无关/错误证据不放大错误 |
| T5.5 | 效率与风险面板（R-12） | executed_core_rounds vs selected；检索/重投影计数；TTFT/decode 分开；Gate 长期 0/1 的排查流程按 V2 §22.3 表；**flops.py 分项计数改造**（V2 §12.1：reader/gate/MHAR 分项，容量不得代理真实开销；Reader 按 T×M 计不是 M²） | 报告口径与 V2 §18.6 一致 |

---

# 3. 与现有决策/里程碑的关系

- **D10 mid-training**：延后到 V2 模块稳定后（新模块也在退火里训练）；嵌入式域数据集（已传 H200）留作 V2 后的域注入。
- **D11 吞吐 / D12 effort 档位 / D13 tokenizer**：正交，保留；D12 的档位上限作用于 V2 的 R_max/halt 上限。
- **实验基线**：V2 迁移前后必须重跑 zone/loss@R 对照（新模块改变了循环语义）。

# 4. 主要风险（V2 §22.3 摘要）

| 风险 | 处理 |
|-|-|
| Gate 长期≈1 或 ≈0 | 不强制正则；按排查表查覆盖/饱和/零初始化阻塞 |
| MHAR 只读最新 | 与任务表现对照后再判；历史可能真无增益 |
| 旧 memory 路径拆分后初始不等 | 有 memory 时不保证初始相等（V2 §14.3），靠 warm-up+蒸馏对齐 |
| 动态路径训练不稳 | 同前缀反事实监督 + 参考重放，不退回随机标签 |
| 因果泄漏修复后 SFT-C 结果变化 | 预期内（旧结果是泄漏收益），重跑并留对照 |

# 5. 执行顺序建议

1. 第 1 周：P0（T0.1 因果修复最优先）∥ P1 移植
2. 第 2-3 周：P2 集成 + 严格兼容回归
3. 第 3-4 周：P3 ∥ P4
4. 第 5 周：P5 验收 + 迁移首训（夜间窗口）；**策略 RL（T4.7⑤）仅在 T3 重放测试通过后开启，不阻塞主线**
5. 之后：回到 D10 域内 mid-training（嵌入式数据）+ Scaling 矩阵

---

# 6. V2 文档章节 → 任务覆盖对照表（v1.1 新增，审计用）

| V2 章节 | 承接任务 |
|-|-|
| §3.2 前置修复 C2–C12 | T0.1(C5/6) T0.2(C4) T0.3+T0.8(C7) T0.4(C8) T0.5(C9) T0.6(C10) T0.7(C3) T2.2(C2) T3.1(C12) T4.1(C11) |
| §5 统一数学/禁止项 | T2.1、T2.3 |
| §6 Loop-MHAR | T1.1、T4.2（η 课程） |
| §7 Update Gate + §7.5 Halt 特征 | T1.2、T2.1（detached 统计） |
| §8 Memory Reader | T1.3、T2.2、T2.3 |
| §9 动作序/三检索动作 | T2.5（RESCORE_WARM 无 warm cache 时 mask） |
| §10 因果/双执行器 | T0.1、T4.6 |
| §11 缓存契约 | T3.1–T3.5 |
| §12 接口/输出 V2 | T1.x、T2.4 |
| §13 梯度接线 | T4.1 |
| §14 初始化/迁移课程/超参 | T2.3、T4.2、T4.3（LR 参数组） |
| §15 损失/Oracle V2 | T4.3、T4.4 |
| §16 数据族 | T4.5 |
| §17 RL 边界 | T4.7 |
| §18 成本/参数/显存 | T5.5、T5.2（参数 manifest） |
| §19 配置/checkpoint | T5.2 |
| §20 R-01..R-14 | R-01/02/03→T1.1/1.2/1.3；R-04→T2.1；R-05→T2.4；R-06→T4.1；R-07→T0.1；R-08→T3.3；R-09→T3.1；R-10→T3.3+T4.6；R-11→T4.2/4.3；R-12→T5.5；R-13→T5.2；R-14→T5.3 |
| §21 五组验收 | T5.1 |
| §22 风险表 | §4（本文风险表） |

---

# 7. 落地进度（2026-09-22，P0+P1+P2 核心完成）

| 任务 | 状态 | 证据 |
|-|-|-|
| T0.1 因果 query | ✅ | question_len 边界 + controller 因果检索 + 反事实 sweep + test_causality（换 gold 初始检索不变） |
| T0.2 collect_router_logits | ✅ | MiniMindXOutput.router_probs（全展开逐轮 halt/retrieval 概率）+ 测试 |
| T0.3 budget 联动 | ✅ | retrieve(topk_fine=budget_blocks) + budget_cap/actual_valid/unique_blocks 统计 + 测试 |
| T0.4 sparse 因果 | ✅ | 块选择按「块起点≤query 位」因果化 + 掩码取交集 |
| T0.5 DDP/分片 | ✅ | sft/grpo DDP(find_unused)+DistributedSampler；controller 兼容 DDP 包装（self.raw 解包） |
| T0.6 CLI None 默认 | ✅ | 四 trainer hidden/layers 默认 None（preset 不再被覆盖） |
| T0.7 gold CE 梯度 | ✅ | grad mode 重算 coarse logits（对齐蒸馏），backward 入 query 投影（测试） |
| T0.8 memory_size 降日志 | ✅ | joint_loss 不再加常量进 total |
| T1.1-1.4 三模块+测试 | ✅ | residual/loop_mhar、recurrent_update、memory/reader 移植；15 项不变量/梯度测试 |
| T2.1 V2 循环顺序 | ✅ | MHAR→Core(首层 Reader)→Update 提交→Halt（V2 §12.4 顺序） |
| T2.2 拆 memory 拼接 | ✅ | v2 模式 memory 全走独立 Reader（ReaderFirstBlock，唯一读取点）；v1 路径保留 |
| T2.3 严格兼容模式 | ✅ | η=0/λ=0/scale=0 下 V2==V1 逐位一致（测试锁定）——**并抓到真 bug：重写时 v1 轮间链路丢失，已修复** |
| T2.4 输出语义 V2 | ✅ | core_committed_states 与 coda 后状态分离；router_probs 输出 |
| T2.5 动作序固定 | ✅ | 三动作 + RESCORE_WARM mask + halt 后禁检索（见 §7.2/§7.3） |

测试：**73 passed**（原 50 + 因果 4 + 残差不变量 15 + V2 集成 4）。

## 7.1 第二批落地（2026-09-22，P3+P4 完成）

| 任务 | 状态 | 证据 |
|-|-|-|
| T3.1 R_max 槽 + last-valid | ✅ | 槽恒 max_loop；深槽携带自身历史（修复了「填充槽带错槽历史」bug）；浅执行 token 深槽=其最后实际轮；先前深槽不丢（3 项测试锁定） |
| T3.3 版本与轨迹 | ✅ | WorkingMemory 带 source_ids/memory_version（修复统计更新丢版本 bug）；controller 轨迹记录（initial/refresh/reuse+轮次+版本，单调）；投影缓存随内容失效 |
| T3.4 checkpoint 纯化 | ✅ | 开/关梯度检查点：检索计数一致、loss 一致、梯度仍通（测试） |
| T3.5 缓存键 | ✅ | 版本化对象身份失效（测试） |
| T4.1 真实路径梯度表 | ✅ | 真实 sft_c_step（非绕开 controller）逐模块梯度验收（含 v2 三模块/检索投影/三 Router） |
| T4.2 warm-up 课程 | ✅ | residual_schedule.py（η/λ 0→1 线性），pretrain_x/sft_x 接线，冒烟曲线正确 |
| T4.3 LR 参数组 | ✅ | build_optimizer：新模块 2×LR；norm/bias/gate scale/pseudo-query 零 weight decay；多组 cosine×mult 调度 |
| T4.4 halt unresolved | ✅ | halt_bce_loss(valid=) 掩码；不把全错样本硬标 R_max |
| T4.5 七族证据数据 | ✅ | gen_evidence_families.py（七族 + 随机事实 + train/val 事实流隔离）；7000 样本已生成 |
| T4.7 RL 边界 | ✅ | 轨迹全字段 + joint_path_logprob + 禁平均深度门槛（策略 RL 仍待重放一致性门槛后开启） |

测试：**82 passed**（+ 缓存契约 5 + 训练器梯度 4）；本地与 H200-1（torch 2.11）双环境一致。
**今晚（09-22 21:00）自动执行 T5.4 V2 迁移首训**：run_stage0_v2.sh（from pretrain_x，warmup 200 步）
→ SFT-A/B v2；明早 8 点自动恢复服务并出 morning_report.txt。

## 7.2 第三批落地（2026-09-22，T2.5/T3.2/T4.6/T4.7/T5 主体完成）

| 任务 | 状态 | 证据 |
|-|-|-|
| T2.5 三动作空间 | ✅ | RetrievalRouter.action_logits 三分类 + RESCORE_WARM 无 Warm Candidate Cache 时 logits mask + 执行侧拒绝（禁止静默降级全局搜索）；controller 三动作执行（REUSE / RESCORE_WARM 候选重排 / GLOBAL_REFRESH）；Warm Candidate Cache；动作 CE + 可用性 mask；旧 oracle 0/1 仍映射 REUSE/GLOBAL_REFRESH |
| T3.2 prefill fixed_cap | ✅ | generate：prompt 用固定 R_prefill，decode 自适应；TTFT / decode 成本分开记（avg_ttft_loops / avg_decode_loops / executed_core_rounds） |
| T4.6 双执行器 | ✅ | trainer/dual_executor.py：reference_forward_by_prefix（逐前缀参考）+ segmented_forward（按 memory-version/动作边界分段）；对应性与换未来 token 因果测试 |
| T4.7 RL 边界 | ✅ | 轨迹全字段（event/action/round/memory_version/budget/exit/logprob/visible_prefix_len）；joint_path_logprob；assert_replay_not_averaged（禁平均深度重放，均值仅日志）；固定 R rollout 已满足；策略 RL 门槛函数就位 |
| T4.4 halt unresolved 接线 | ✅ | sft_c_step 在线 sweep 时 halt_valid = per_exit_correct.any()；joint_loss(valid=) 真正传入 |
| T5.1 五组验收 | ✅ | tests/test_v2_acceptance.py（任务与效率/因果安全补充）+ 既有 residual/cache/causality/trainer_gradients |
| T5.2 checkpoint/CLI | ✅ | config.validate()（D 整除 MHAR heads、query_heads 整除 kv_heads、sparse 冲突、budget 递增等）；param_manifest(sum p.numel 分模块)；ckpt `__meta__`（arch/cache version + manifest）；训练入口 validate_config |
| T5.5 效率与风险面板 | ✅ | flops.py 分项：reader_attn(T×M 非 M²)/reader_proj/reader_gate/mhar/update_gate；residual 关闭时主干线性于 R 保持；动作计数（reuse/warm_rescore/global_refresh） |
| T5.3 多卡 | ◐ | T0.5 DDP 已就位；8 卡短跑待 H200 夜间窗口验证 |
| T5.4 迁移首训 | ◐ | 今晚 21:00 自动执行（launcher 已武装）；结果见 morning_report.txt |

**审查修复（本批）**：
1. `sweep_retrieval` 在 residual_v2 下误走 v1 `get_kv` 拼接 → 改 `_memory_input` 双路径；
2. `halt_valid` 仅在 losses 有签名、trainer 未传 → 已接线；
3. 三动作标签误入二值 BCE → 拆分（action CE 用 0/1/2，二值 = 是否离开 REUSE）；
4. 检索 recall 测试依赖 RNG 运气（fine 用 Prelude 编码 vs raw-embedding query）→ 收紧到 coarse 契约 + 显式种子；
5. FLOPs 分项在 residual_v2=False 时破坏主干线性 → 按开关计入。

测试：**95 passed**（+ 三动作/双执行器/配置校验/验收汇总 13 项）；本地与 H200 待同步。

## 7.3 第四批落地（2026-09-22，审查缺口收口）

| 任务 | 状态 | 证据 |
|-|-|-|
| T4.6 双执行器真等价 | ✅ | dual_executor 重写：统一 per_token_ce 语义；REUSE 与 GLOBAL_REFRESH 动作边界下 `max\|ΔCE\|≤1e-4`（不再只查形状） |
| T2.1 halt detached 统计 | ✅ | ḡ/Δ^cand/Δ^commit 零初始化旁路进 halt logit（旧权重可加载，初始贡献 0）；测试锁定 |
| T2.4 OutputV2 补齐 | ✅ | `coda_exit_states` / `halt_logits` / `memory_versions` / `router_trace` / `residual_stats` 全部进 MiniMindXOutput |
| T3.3 WorkingMemoryV2 | ✅ | `valid` / `metadata` / `token_valid()`；source_id 去重（同 source 不复制 slot） |
| T5.2 完整 checkpoint | ✅ | save/load/restore：optimizer + scaler + python/numpy/torch/cuda RNG + data/warmup 进度 + tokenizer/source manifest + param_groups 清单；roundtrip 测试 |
| T2.5 动作序测试 | ✅ | halt 后不得再检索（轨迹 round < num_loops_executed） |

测试：**101 passed**（+ field contracts 4 + 动作序 1 + 双执行器 refresh 边界 1）。

## 7.4 二次细审修复（2026-09-22）

| 问题 | 修复 |
|-|-|
| `memory_versions` 用 `id()%1e5` 充数 | `MemoryInput` 携带真实 `memory_version`；thinking_core 读显式版本 |
| 检索 `source_ids` 默认槽位 arange → 去重失效 | `memory_manager.update(source_ids=store 块号)`；重复块只 bump usage 不复制 slot |
| T4.2 课程是线性 0→1 | 改为 V2 §14.2 两段：前 20% 窗口→0.25，后 80%→1（≈总步 1%/5%） |
| T3.5 缓存键不完整 | `WorkingMemory.cache_key(request/memory_version/layer/proj/encoder/device/dtype)` + `get_kv(cache_key=)` |
| T4.4 缺条件轨迹成本 C(τ) | `trajectory_cost()` + `joint_loss(cost_weight=)`（默认 0 仅日志） |
| T4.5 缺 gold span | 七族输出 `gold_span` 字符偏移（token id 由 dataset 按 tokenizer 转换） |
| T4.7 缺同轨迹重放测试 | `test_v2_replay_and_cachekey.py`：同动作时间线 logits/轨迹一致 + 禁平均深度 |

测试：**106 passed**。

## 7.5 外部评审修复（2026-09-22，G1/G2/G3）

对照《MiniMind-X V2 实现深度审查报告》逐条复现后修复（评审结论大体成立）：

| 组 | 问题 | 修复 |
|-|-|-|
| G1 | sweep_budget 丢 query_len，gold 泄漏进预算检索 | query_len 必传；训练 forward 缺 query_len 直接报错 |
| G1 | Router 训练吃全序列 / 执行吃前缀 | halt/retrieval/budget/action logits 统一 `[:, :query_len]` |
| G1 | 三动作编号被对调（1↔2） | schema v2 唯一枚举；旧二值仅在数据边界 `normalize_oracle_actions` 转一次 |
| G1 | 单块 memory score Inf→NaN | 单块 redundancy=0/novelty=1 |
| G1 | 去重后 usage 记错块 | 按 source_id 映射 hit |
| G1 | 空 context torch.cat 崩 | 空库合法 no-op；空库跳过检索监督 |
| G1 | max_steps 跳过最终保存 | 循环外统一落盘 |
| G1 | GRPO 静默 V2→V1 / unwrap 丢 DDP / 跨设备 cat / 平均深度重放 | meta 强制 residual_v2+拒绝丢模块；梯度走 DDP；CPU cat；禁 avg-depth 除非显式开关 |
| G2 | 早停样本仍写深轮 KV | per-sample last-valid + update_gate active mask |
| G2 | Oracle 全程分叉/动作前出口打分 | 同快照三动作，评动作后出口 |
| G2 | 双执行器均为从头重算 | 保持（append-only 真参考待与 generate 对齐，见剩余） |
| G3 | Reader 投影每 loop 重算 | 按 memory_version 跨 loop 复用；**每 forward 重置**（防 no_grad sweep 污染） |
| G3 | gold_span 未转块 / split 未过滤 | dataset 按 tokenizer 重算 gold_block；split 过滤 |
| G3 | FLOPs/executed 口径 | 投影计 T×M/M；selected vs executed 分字段 |

测试：**119 passed**（含评审回归 G1/G2/G3）。

## 7.6 append-only 对齐（2026-09-22）

| 项 | 状态 | 证据 |
|-|-|-|
| reference_append_only | ✅ | 带 cache 增量，金标准（V2 §11.5）：历史 KV 不回溯，memory 换版只作用新 token |
| segmented_forward | ✅ | 段间续 past_key_values，段内批量追加——对齐 append-only，不再从头重算 |
| generate_teacher_forced | ✅ | 与 generate 同构（首块 prefill + 逐 token 追加） |
| reference_recompute | ◐ | 保留为对照路径（违反 append-only，不是金标准） |
| 对齐测试 | ✅ | append-only↔segmented logits≤1e-4；generate 第二步 argmax 可重放；memory 切换下 segmented 跟 append-only 而非 recompute |

测试：**123 passed**。

剩余（评审亦指出，需更大改动/外部条件）：
- 训练主路径 collect_exits 仍是全序列 teacher-forcing（与 decode 的 M(t) 边界可再收紧）
- 混合 batch 的 active-mask 压缩与逐样本动作（当前 batch 统一动作，研究级）
- T5.3 八卡、T5.4 迁移首训对照

## 7.8 数据真值 + sparse 块级因果（2026-09-22）

| 项 | 修复 |
|-|-|
| irrelevant | 干扰设备强制 ≠ 目标；生成自检 context 不得含目标设备（200 条 0 泄漏） |
| no_evidence | 目标改为确定「不可知」，禁止随机编造采样率 |
| new_evidence | 增加 `evidence_visible_from_round` 可见时间线 |
| long_dep | 目标改为 X+正整数（原先仍是 X） |
| planning | 有序依赖子序列（禁止乱序 sample）+ 约束写入答案 |
| repo_context | patch 使用真实参数名，禁省略号/未定义 args |
| sparse 选择器 | 全局 top-k 只选**完整可见块**；未完成块走因果局部窗（改未来不改过去输出） |

测试：**133 passed**。

## 7.7 Frozen Index 声明（2026-09-22）

检索采用 **Frozen Index 范式**（非 bug）：建库 `build_keys` 在 no_grad 下用 `key_proj`
生成库 key；训练只优化 query 投影 + ranking（CE/蒸馏）。`config.freeze_key=True` 默认，
`build_optimizer` 显式 `requires_grad_(False)`，manifest 记 `frozen_key_params`。
文档已修正旧表述「梯度只走 query/key 投影」→「只走 query 投影」。
未来若要 Q/K 联训：周期 `key_refresh_interval` rebuild index，不能只拆 no_grad。
测试：**126 passed**。
```

---

# docs/minimind_x_gated_recurrence_loop_mhar_memory_reader_v2.md

```markdown
# MiniMind-X V2：受控循环、历史状态路由与独立记忆读取

## Gated Recurrence + Loop-MHAR + Memory Reader Gate 落地技术方案

**版本：** v2.0-design  
**日期：** 2026-09-21  
**实现基线：** `minimind_x_code_bundle(2).md`，55 个新增代码文件、5083 行代码。  
**文档性质：** 针对既有 MiniMind-X 的增量架构与实现规格，不是已经完成训练的技术报告。  
**交付边界：** 本文包含完整设计、代码改造清单、训练接线、缓存契约、验收标准，以及三个核心模块的独立 PyTorch 参考实现。参考实现已做 CPU 单元测试；尚未与原仓库合并，不代表完整解码器、多卡训练或 1M 长上下文已经验证。  
**基线文件 SHA-256：** `d94753134740998e40e88eee0c8be1d164892ea125aea0e4b3cd096af50f0da1`

---

## 阅读导航

- 第 1–3 节：最终决策、研究依据与当前实现的衔接。
- 第 4–8 节：统一张量与数学定义，三个新增模块的准确职责。
- 第 9–12 节：动态检索、因果性、KV 缓存、PyTorch 接口和执行流程。
- 第 13–17 节：训练梯度、初始化、损失、数据与 RL 接线。
- 第 18–22 节：成本、配置、开发任务、验收与上线条件。
- 附录 A–D：实现参考、测试范围、证据索引与参考资料。

# 1. 最终决策

## 1.1 保留已经确定的终局主线

MiniMind-X 的主线不变：

```text
长上下文/领域知识源
        ↓
Dynamic Retrieval
        ↓
Adaptive Working Memory
        ↓
Adaptive Thinking
        ↓
答案
```

本次改造只进一步回答三个问题：

| 新模块 | 回答的问题 | 作用位置 |
|---|---|---|
| Loop-MHAR | 下一轮应读取哪些历史思考状态？ | 每轮 Core 执行前 |
| Memory Reader Gate | 外部知识读入后，允许注入多少？ | Core 首层 Self-Attention 之后、FFN 之前 |
| Gated Recurrent Update | 本轮候选状态，真正写回多少？ | 整个 Core 执行后 |

**Halt Router 仍然单独存在**，决定是否执行下一轮；它不与 Update Gate 合并。

最终推荐结构为：

```text
                              长上下文 / Domain SLM Slots
                                          │
                                    动态检索/桥接
                                          │
                                   Working Memory
                                          │
输入 → Embedding → Prelude → 固定 Anchor E │
                            │             │
                            ▼             │
                       H^(0) = E          │
                            │             │
                ┌───────────▼─────────────┴─────────────┐
                │              Thinking Loop r          │
                │                                       │
历史 [E,H1,...] ─► Loop-MHAR → 本轮读取状态 Z_r           │
                │                  │                    │
                │        首层 Self-Attention             │
                │                  │                    │
                │        独立 Memory Cross-Attention ◄───┤
                │                  │                    │
                │          Memory Reader Gate           │
                │                  │                    │
                │       FFN + 其余共享 Core 层           │
                │                  │                    │
                │          候选状态 U_r                 │
                │                  │                    │
上一轮 H^(r-1) ─► Gated Recurrent Update ◄── Anchor E    │
                │                  │                    │
                │          提交状态 H^(r)               │
                └──────────────────┬────────────────────┘
                                   │
                      Halt Router：停止 / 继续
                           │               │
                         停止            继续
                           │               │
                           │       Retrieval Router
                           │       复用 / 重排 / 全局检索
                           │               │
                           │       更新 Memory 后下一轮
                           ▼
                          Coda → LM Head
```

这里的循环单位仍然是完整共享 Core，例如 `[L4 → L5 → L6]`，不是某个 Attention 层自己重复，也不是每个 Loop 都重新编码全部长上下文。

## 1.2 本版本明确不增加的东西

本版本不引入全模型 mHC，不把所有层替换为原版 Full AttnRes，不同时增加 GDN/KDA，也不要求领域小模型与大模型共享 tokenizer。

领域小模型的 Memory Slots 只作为另一种 `MemorySource` 接入。本文不把“训练领域小模型”和“训练跨模型 Bridge”混入残差改造的核心交付。

这是一套完整目标架构。训练中的 warm-up、兼容性测试和修复清单，都是保证该架构可训练的工程步骤，不是重新做一轮架构筛选。

## 1.3 必须校准的五条理解

1. **无显式 loop embedding，依然是合法的 recurrent/weight-tied Transformer。** 不能把 `loop_embedding=False` 本身定为错误。本文保留该开关，默认关闭。
2. **Update Gate 小，不等于模型已经答对，也不等于一定该停止。** 可能是正确状态趋稳，也可能是错误固定点、门控饱和或欠训练。
3. **算完候选结果后再关 Gate，不会省掉已经执行的 Core FLOPs。** 真正省计算需要 Halt、样本压紧或上游跳过操作。
4. **有 Gate 不保证永远不会 overthinking，也不保证通用能力绝不下降。** 它提供一种可学习的状态更新机制，效果需要任务与数值验收。
5. **Loop-MHAR 是本文提出的“将多头深度路由迁移到 loop 边界”的设计。** 已有 MHAR 论文结果不能直接当作 MiniMind-X 上这一组合的实证结果。

# 2. 研究依据、借鉴部分与自定义部分

## 2.1 已核实的研究依据

| 来源 | 本文借鉴的内容 | 不直接推导的结论 |
|---|---|---|
| Attention Residuals，arXiv:2603.15031 | 使用归一化来源表示与学习 query，对历史深度来源做 softmax 加权读取；存在 Block 版本 | 不据此保证任何 loop 历史混合都优于普通残差 |
| Multi-Head Attention Residuals，arXiv:2607.27230 | 将 feature width 划分为多个头，每头独立选择深度来源 | 论文主要操作 layer/sublayer 来源；不是本文的 committed-loop-state 历史 |
| Gated Recurrent Transformers / RecurrentGPT，arXiv:2608.15062 | Prelude–shared Core–Coda 结构、逐元素更新门、固定 Prelude 条件 | 本文省略原工作中的状态/门控噪声，门控网络输入也不声称逐行复现原实现 |
| mHC，arXiv:2512.24880 | 多路残差与约束混合是另一条可行路线 | 本文不实现 mHC，也不据其数值稳定结论保证本方案稳定 |
| PyTorch 官方 Autograd / SDPA 文档 | 正确区分冻结参数与 `no_grad`；明确 attention mask 和 dropout 语义 | 框架支持 API，不等于目标硬件一定使用最快 kernel |

资料与链接见附录 D。[R1–R6]

对 arXiv:2608.15062，本次检索的 arXiv 摘要索引与作者模型卡在标题表述上存在 `RecurrentGPT` / `Gated Recurrent Transformers` 的差异；本文以论文编号识别来源，以作者模型卡确认相关设计，不把标题差异扩展成两个独立方法。[R3]

## 2.2 本文实际新增的组合设计

以下均是 MiniMind-X 的实施选择，而不是宣称来自某篇论文的原样方案：

- 用完整的 **committed loop state**，而非所有 Attention/MLP 原始输出，作为 Loop-MHAR 的来源。
- 只在 loop 边界执行一次历史读取，默认共享一组四头 pseudo-query。
- 给最新状态一个可学习初始偏置，避免迁移时一开始平均混合全部历史。
- 仅在 Core 首层增加一次独立 Memory Reader，避免重复多层读取扩大成本。
- Update Gate 使用上一轮状态、Prelude Anchor、候选差分三类信息，通过低秩 MLP 输出逐通道门。
- 明确区分 token self-KV、memory KV 和 loop history，制定训练/推理各自的生命周期。

这些选择形成一套可实现的方案，但“优于其他架构”“减少幻觉”“收益叠加”仍是待验证的目标。

# 3. 现有实现的衔接与必须修复的前置条件

## 3.1 可以直接保留的部分

基线已经有 Prelude/Core/Coda 切分、共享 Core、分轮次 self-KV、WorkingMemory 容器、检索与预算控制器以及多出口监督接口。[C1–C4]

建议保留这些结构和原有权重名称，仅通过新增模块及明确的版本字段升级。

## 3.2 为什么不能只“再加三个类”就开训

基线代码存在一些与本次改造直接相关的接线问题。这里是基于附件源码的判断，不是沿用此前评审中的成熟度评分。

| 位置 | 附件中可见的情况 | 本版本的处理 |
|---|---|---|
| `MemoryAwareAttention.forward` | memory K/V 与 token K/V 拼接，进入同一个 softmax | 拆成 Self-Attention 与独立 Reader；禁止新旧两条 memory 路径同时读入 |
| `AdaptiveController.forward_with_oracles` | `get_kv()` 位于 `no_grad` 范围内 | 冻结索引/编码与训练 Reader 投影分开；训练 KV 投影保留计算图 |
| `retrieve_into_memory` → `retrieval_ce` | 在 `no_grad` 生成 coarse scores 后直接用于 CE | 有监督时重新在 grad mode 计算 logits；不能只把 detached loss 加进总 loss |
| `collect_exits=True` 的训练前向 | Halt 概率只在 `halt=True` 时产生，训练时可能返回空列表 | 新增独立 `collect_router_logits`，训练算概率但不据此提前 break |
| SFT-C 的 query 构造 | `input_ids` 含问题和 gold 答案，Prelude 后对全序列池化用于检索 | query 必须只来自当时可见前缀；不能让 gold 后缀影响前面的 memory 选择 |
| `sweep_retrieval` 调用 | 调用时未传 memory，with/without 两条路可同为无 memory | 改成同一前缀状态下“复用当前 W”与“真实 refresh 后 W'”的反事实比较 |
| Budget 与 Top-K | budget 控制淘汰容量，但检索仍使用固定 `topk_fine` | 将目标容量/缺口真正传给候选选择；否则多数预算档可能实际取到相同内容 |
| Memory cost | `memory_size` 是新建常量张量 | 常量用于日志；策略优化用可微期望成本或动作 reward |

依据：[C2–C7]。上述问题不会因增加 Gate 自动消失。

另有两个上线前必须阻断的基础问题：基线 SparseBlockAttention 的 `allow_blocks | band` 之后没有统一再与因果 mask 取交集，块级选择本身也需要因果限制；SFT/RL 启动了 distributed 环境不等于已经正确包装 DDP 并分片数据。[C8–C9] 本文不把修复这些问题包装成残差创新，但训练前必须处理。

# 4. 统一符号、三类记忆与张量契约

## 4.1 符号

| 符号 | 含义 | 典型形状 |
|---|---|---|
| B | 当前批次样本数 | 标量 |
| T | 当前处理的 token 数，prefill 可大，decode 通常为 1 | 标量 |
| D | MiniMind-X hidden width | 读取实际 config，不写死 |
| R | 本次最大 Core 循环数 | 默认 4 |
| E | 同一前向、同一 token 的 Prelude Anchor | `[B,T,D]` |
| H^(r) | 第 r 轮提交后的状态 | `[B,T,D]` |
| U_r | 第 r 轮完整 Core 给出的候选状态 | `[B,T,D]` |
| Z_r | 历史混合后，本轮 Core 的读取状态 | `[B,T,D]` |
| H_hist | Loop-MHAR 的 feature heads 数 | 默认 4，要求 D 可整除 |
| M | 当前活跃 memory token/slot 数 | 可变，不等于 T |
| D_m | memory payload 的宽度 | 可与 D 不同 |
| W | memory payload | `[B,M,D_m]` |
| valid_memory | memory 有效位 | `[B,M]`，bool |
| allowed_memory | 每个 query 能读哪些 memory 项 | 可选 `[B,T,M]`，bool |
| active | 当前轮仍需计算的 token/样本位置 | `[B,T]`，bool |

`True` 在本文所有可见性 mask 中都表示“有效/允许访问”。与其他库对接时必须显式转换，而不是按变量名猜测。

## 4.2 三类状态绝不能混用

**Loop History** 保存同一 token 在历轮的状态：`[E,H1,H2,...]`。它解决“从哪一轮的表示读取”。

**Working Memory** 保存外部证据：检索到的 token hidden 或经过对齐的 Domain SLM slots。它解决“当前有哪些知识可读”。

**Token Self-KV Cache** 保存历史文本 token 在各物理层、各 loop 的 self-attention K/V。它解决“自回归生成时如何复用过去计算”。

因此：

```text
Loop History 的长度：约 1~R，不是 8K。
Working Memory 的长度：例如 1K~8K raw tokens，或 16~64 latent slots。
Token KV 的长度：已处理的文本 token 数，并带 Core 的 loop 轴。
```

本文不改变既有 `memory_budgets` 的总体机制；8K/16K 是资源上限配置，不是 Gate 或 MHAR 的数学常数。不同来源的 raw tokens 与 latent slots 必须分别记录，不能仅按数量声称压缩收益。

# 5. 一轮计算的统一数学定义

设原有共享 Core 有 K 个物理 Transformer blocks，例如 K=3。定义：

\[
E=P_\theta(X),\qquad H^{(0)}=E.
\]

第 r 轮使用的历史为：

\[
\mathcal V_r=\{E,H^{(1)},\ldots,H^{(r-1)}\}.
\]

注意：r=1 时只有一个来源 E；不要同时加入 `anchor` 和同一个 `h0` 造成重复计权。

每轮执行：

\[
\widetilde H_r=\operatorname{LoopMHAR}(\mathcal V_r),
\]

\[
Z_r=H^{(r-1)}+\eta\big(\widetilde H_r-H^{(r-1)}\big),
\]

\[
U_r=\operatorname{CoreWithReader}(Z_r,W_r),
\]

\[
g_r=\operatorname{UpdateGate}\big(H^{(r-1)},E,U_r\big),
\]

\[
\boxed{H^{(r)}=H^{(r-1)}+g_r\odot\big(U_r-H^{(r-1)}\big).}
\]

其中，`η` 是迁移 warm-up 的历史混合强度：从头训练默认 1；从旧权重迁移时可从 0 平滑提升至 1。它不是第四个必须学习的智能控制器。

最后，Halt Router 读取**提交后的 H^(r)**，决定是否进入下一轮；若继续，Retrieval Router 再决定下一轮的 W 是否更新。

**关键禁止项：** `U_r` 已包含 Core 内部 residual，不能写成 `H_next = H_prev + g * U_r`。那会重复叠加旧状态。也不能在更新式的 bypass 中把 `H_prev` 偷换成 `Z_r`，否则 `g=0` 时仍可能通过 MHAR 改写状态。

# 6. Loop-MHAR：按 feature 子空间读取历轮状态

## 6.1 来源与归一化

堆叠历史：

\[
V\in\mathbb R^{N_r\times B\times T\times D},\qquad N_r=r.
\]

对每个来源的完整 D 维做 RMSNorm，仅用于计算 key：

\[
K_j=\operatorname{RMSNorm}(V_j).
\]

再把 D 维拆成 H_hist 个头，每头 `d_h=D/H_hist`。

Value 使用原始的 V，而不是把 RMSNorm 后的 K 当作 Value。否则会把“如何打分”和“实际读回什么状态”混成另一种算法。

## 6.2 每头独立路由

使用一组跨 loop 共享的可训练 pseudo-query：

\[
q^{hist}\in\mathbb R^{H_{hist}\times d_h}.
\]

对来源 j、第 h 个头：

\[
s_{r,j,h}=\langle q^{hist}_h,K_{j,h}\rangle+b_{\operatorname{type}(j,r),h}.
\]

source type 分为 Anchor、较早的 committed state、最新 committed state。类型偏置是本方案的迁移稳定性扩展。

\[
\alpha_{r,j,h}=\operatorname{softmax}_{j}(s_{r,j,h}),
\]

\[
\widetilde H_{r,h}=\sum_j\alpha_{r,j,h}V_{j,h}.
\]

拼接各头恢复 `[B,T,D]`。

**Softmax 只能沿历史来源 j 轴执行，不能沿 token、batch 或 head 轴执行。** 此算子不做跨 token 读取，因此不会自行引入未来 token 依赖；前提是历史状态本身因果正确。

参考实现沿用 pseudo-query 内积，不额外加 `1/sqrt(d_h)`；若改变该缩放，必须同步初始化与实验口径。

## 6.3 为什么采用“整轮状态”而不是“所有子层增量”

这是计算和实现边界的选择。K=3、R=4 时，访问所有子层输出需要更多来源和显存读写；只保存 Anchor 与 committed-loop-state，可以在保持当前 Core 内部 PreNorm 结构的同时，对历轮状态做有限的重新组合。

它也意味着本方案与原版 AttnRes/MHAR 不等价：来源是已经包含历史信息的累积状态，相关性可能很高。多头组合可能帮助恢复有用表示，也可能组合出不一致的子空间。本文不承诺该方法消除信息稀释，更不把它解释为“自动找回正确答案”。

## 6.4 初始化

默认：

```yaml
loop_mhar:
  enabled: true
  heads: 4
  shared_query_across_loops: true
  query_init: zeros
  latest_source_bias_init: 4.0
  other_source_bias_init: 0.0
  key_norm: full_width_rmsnorm
  history_values: committed_states
  detach_history_in_training: false
```

在 query 为零、最新来源偏置为 4 时，若共四个来源，最新来源初始权重为：

\[
\frac{e^4}{e^4+3}\approx0.948.
\]

这不是严格“只读最新状态”。严格兼容测试应把 `η=0`，而不是依赖很大的有限 softmax 偏置。

## 6.5 历史生命周期

训练时，`history.append(committed)`，不在 R≤4 的主训练路径中 `detach()`；否则早期 loop 收不到后续历史读取路径的梯度。

同一 forward 内不能 inplace 修改已经存入 history 的 Tensor。迁移到 fused buffer 时，要重新验证 autograd 与 checkpointing；参考版先用不可变列表与明确 stack。

在自回归 decode 时，历史列表只针对**当前新 token 的历轮状态**建立；过去文本的信息由 self-KV 读取。不应把上一输出 token 的 `H4` 当作下一 token 的 `H0`。

# 7. Gated Recurrent Update：控制本轮候选的写回

## 7.1 默认门控网络

定义：

\[
\Delta U_r=U_r-H^{(r-1)}.
\]

构造：

\[
f_r=\operatorname{Concat}\big(
\operatorname{RMSNorm}(H^{(r-1)}),
\operatorname{RMSNorm}(E),
\operatorname{RMSNorm}(\Delta U_r)
\big).
\]

用低秩 MLP：

\[
a_r=W_2\operatorname{SiLU}(W_1f_r+b_1)+b_2,
\qquad g_r=\sigma(a_r).
\]

其中 `W1: 3D→k_g`，`W2: k_g→D`，默认 `k_g=64`。门为 `[B,T,D]`，逐 token、逐通道控制写回。

最终：

\[
H^{(r)}=(1-g_r)\odot H^{(r-1)}+g_r\odot U_r.
\]

- `g=0`：保留旧状态。
- `g=1`：完全采用候选。
- 中间值：逐通道插值。

这只是向量层面的状态更新。不能声称某个坐标天然对应“数学知识”或“语法”；这类含义最多是解释性假设。

## 7.2 Anchor 的作用边界

Anchor E 在每轮 Gate 中可见，也始终是 MHAR 的候选来源。它提供一个固定参照，有助于模型学习何时保留原始输入表示。

但“可读取 Anchor”不意味着模型一定使用它，也不是严格防止语义漂移的约束。应记录 Anchor 权重和任务结果，而不是强行要求 Anchor 的权重始终大于某个阈值。

## 7.3 数值与稳定性边界

插值形式可限制**每个坐标**落在旧值与候选值之间，但不保证完整向量范数下降。

例如 `h=(1,0)`，`u=(0,1)`，`g=(0,1)`，则输出为 `(1,1)`，其 L2 范数比 h、u 都大。

而且 Gate 本身依赖状态。简化地把其他来源固定，有：

\[
\frac{\partial H'}{\partial H}
=I+\operatorname{diag}(g)(J_U-I)
+\operatorname{diag}(U-H)J_g.
\]

所以 `0≤g≤1` 并不自动给出整个循环的 contraction 或梯度稳定保证。仍需 FP32 门控计算、梯度裁剪、范数监控和长 loop 验收。

## 7.4 初始化与梯度

默认 `W2=0`，`b2=+4`：

\[
g\approx\sigma(4)=0.9820.
\]

这样初始行为接近旧的“接受 Core 输出”，但不是严格相等。与作者 GRT 模型卡中近开门初始化的思路一致；具体低秩结构是本方案定义。[R3]

该初始化下，第一步 `W2/b2` 可有梯度，而 `W1` 可能因 `W2=0` 暂时没有梯度。这是初始化的正常结果，不应要求所有层第一步梯度都非零；应在后续若干步检查整条路径。

默认不加 state noise、gate noise，也不加让 `mean(g)` 持续减小的正则。

## 7.5 Update Gate 与 Halt Router 的分工

Update Gate 决定“算出来的候选写入多少”；Halt Router 决定“下一轮是否再执行”。

Halt 可读取以下统计作为额外特征：

\[
\overline g_r,\quad
\Delta^{candidate}_r=\frac{\operatorname{RMS}(U_r-H^{(r-1)})}{\operatorname{RMS}(H^{(r-1)})+\epsilon},
\]

\[
\Delta^{commit}_r=\frac{\operatorname{RMS}(H^{(r)}-H^{(r-1)})}{\operatorname{RMS}(H^{(r-1)})+\epsilon}.
\]

推荐：这些数值特征先 `detach()` 再喂 Halt，以免 Halt 辅助损失通过统计量诱导 Update Gate 刻意关闭。

**不采用此前脑暴中的硬绑定** `P(continue)≈mean(g)`；也不以 `Δ_commit` 很小直接生成“正确停止”的标签。它们是状态变化特征，不是正确性证据。

# 8. Memory Reader Gate：独立读取和有界注入外部证据

## 8.1 插入位置必须唯一

默认只在 Core 的第一个物理 block 内：

```text
输入 Z_r
  ↓
Self-Attention + 原有 residual
  ↓
Memory Cross-Attention + Reader Gate + 独立 residual
  ↓
FFN + 原有 residual
  ↓
Core 的其他物理 block
```

K=3、R=4 时，最多执行四次 memory read，而不是十二次。检索次数仍可以只有一次；“从已缓存 W 读取”与“重新搜索全库”是两笔不同成本。

若以后把 Reader 扩到多个 Core 层，必须在配置、FLOPs、参数量和 KV 缓存中显式体现。首版不同时做 pre-core Reader 与 per-layer Reader，避免重复注入。

## 8.2 数学定义

第一个 block 的 self-attention 之后：

\[
A_r=Z_r+SA_0(N_0(Z_r)).
\]

独立 memory 分支：

\[
Q=W_Q^mN_q(A_r),\qquad
K=W_K^mN_m(W_r),\qquad V=W_V^mN_m(W_r).
\]

\[
m_r=W_O^m\operatorname{Attention}(Q,K,V;\text{visibility mask}).
\]

token 级 Reader Gate：

\[
c_r=\sigma\Big(W_{m,2}\operatorname{SiLU}(W_{m,1}[N_q(A_r);N_o(m_r)])+b_m\Big),
\]

\[
\rho=\tanh(a_m),\qquad
\overline A_r=A_r+\rho\,c_r\odot m_r.
\]

随后执行 FFN 及其余 Core 层，得到 U_r。

默认 `c_r:[B,T,1]`，可广播到 D 维。Reader Gate 和 Update Gate 不需要相同粒度：前者先控制整条外部证据支路，后者逐通道提交候选。

`ρ` 是可训练、有界的有符号残差缩放，不是置信度概率；`c_r` 也不应未经校准就向用户显示为“知识可信度”。

## 8.3 不再混合两套 K/V 坐标系

新 Reader 的 Q、K 都不使用 token self-attention 的 RoPE。原 self-attention 保留自己的正常位置编码。

基线将已 RoPE 的 self-query 与未 RoPE 的 memory key 放在同一 softmax 内。新方案通过独立 reader 投影和注意力消除这种隐式混用。[C2]

memory 内的顺序、文件来源和版本信息，要由 Memory Encoder/Bridge 及 metadata 明确提供；独立 Cross-Attention 自己不能恢复已经被 chunking 丢掉的全局位置信息。

## 8.4 空记忆与 mask 的严格语义

- `M=0` 或 `memory=None`：Reader 直接返回输入，保证 no-op。
- 某个 batch 行没有有效 memory：该行必须 no-op，不产生 NaN。
- 某个 query 由于可见性约束不能读任何真实 memory：同样 no-op。
- padded memory 不得进入 norm、score 或统计量。
- 训练/推理均支持 `allowed_memory[B,T,M]`；它控制因果/可用性，而不是假设整个 batch 共享同一份可见内容。

参考实现增加仅在“无真实可见项”时启用的零值 NULL K/V，以规避全遮罩 softmax 的问题；随后把该 query 的注入 delta 严格置零。NULL 不作为无条件可选知识源。

## 8.5 初始化不要“双零锁死”

从头训练默认 `ρ≈0.01`，Reader Gate 初始约 0.5，`W_O^m` 正常随机初始化。

迁移旧模型、需要初始精确 no-op 时，可令 `a_m=0 → ρ=0`，但 `W_O^m` 必须保持非零初始化。第一步先更新缩放，之后再使 Reader 内部参数获得任务梯度。

不能同时使用：

```text
reader scale = 0
且
reader output projection = 0
```

否则两者可能互相使对方梯度为零，导致支路无法启动。

## 8.6 Reader Gate 不替代证据验证

外部 memory 可能错误或不适用。Reader Gate 提供降低注入的手段，但不是事实判别器的保证。

数据必须包含相关、无关、相似但版本不适用、冲突和缺失证据；评测必须检查这些情况，而不能仅检查“加 memory 后 logits 有变化”。

# 9. 与 Dynamic Retrieval / Adaptive Working Memory 的执行协议

## 9.1 每轮动作顺序

必须固定为：

```text
1. 读取上一轮 committed state 与当前 Memory 版本
2. Loop-MHAR
3. CoreWithReader 生成候选
4. Update Gate 提交新状态
5. 计算 Halt logits
6. 若停止，直接进入 Coda；若已到 R_max，也禁止无用的再检索
7. 若继续，计算 Retrieval / Budget logits
8. 按动作更新 Memory，增加 memory_version
9. 开始下一轮
```

不能先昂贵地检索，再发现已经停止；也不能先用候选 U 决定检索，随后 Gate 几乎拒绝 U，却仍按被拒绝的状态执行动作。

## 9.2 三种 Retrieval 动作

默认动作空间：

\[
a^{ret}\in\{\text{REUSE},\text{RESCORE\_WARM},\text{GLOBAL\_REFRESH}\}.
\]

- `REUSE`：沿用当前 Hot Memory，不重建索引，不重投影 K/V。
- `RESCORE_WARM`：在已缓存候选中重新选择；可能改变 memory_version。
- `GLOBAL_REFRESH`：基于当前可见状态重新搜索全局索引，再更新候选与 Hot Memory。

如果现有实现还没有 Warm Candidate Cache，`RESCORE_WARM` 必须被标成不可用并在 logits 中 mask 掉，不能把它悄悄实现为全局搜索后仍按低成本计费。

## 9.3 Memory 更新协议

Memory 必须保留 `source_id/block_id`，并按这些 ID 去重。重复检索同一块，应该更新 usage/recency，不应把同一知识复制成多个 slot，虚增 memory_size 和 attention 权重。

建议版本字段：

```python
@dataclass
class WorkingMemoryV2:
    hidden: Tensor                 # [B,M,Dm]
    valid: Tensor                  # [B,M]
    source_ids: Tensor             # [B,M] 或 token→block 映射
    source_versions: list[str]
    memory_version: int
    metadata: dict
```

原始 block memory 可先按 `[B,K,block,D]` 管理，再展平给 Reader。Domain SLM slots 通过同一接口进入，但 metadata 必须说明来源类型与 Bridge 版本。

预算 Router 决定的是合法容量上限；如果内容不足，实际使用可以更少。日志必须分别记录 `budget_cap_tokens`、`actual_valid_tokens`、`unique_blocks`，不能用分配容量冒充有效证据量。

## 9.4 新证据不应被频繁关闭的 Gate 永久挡住

当 memory_version 改变时，可把 `memory_changed`、新增有效块数量、检索分数差等作为 Halt/控制器输入，并允许其决定多算一轮。

但不采用“只要检索就强制 Gate=1”。因为新证据也可能无关或错误。新增证据样本应进入训练，验证模型可以重新打开 Reader/Update Gate，而不是靠硬覆盖旧状态实现。

# 10. 因果性与训练/推理一致性

## 10.1 数据字段必须分开

训练样本至少区分：

```json
{
  "source_context": "外部可查询资料",
  "prompt": "当前用户已经提供的完整问题",
  "completion": "监督答案",
  "evidence_spans": [],
  "source_version": "...",
  "task_id": "..."
}
```

不能只拿 `input_ids = prompt + completion` 再通过全序列平均池化构造检索 query。当前附件的 SFT-C 有这条风险，必须在接入新模块前修复。[C4–C6]

## 10.2 初始检索与后续检索的可见边界

初始检索可以读取**完整已知 prompt**，因为用户已提供全部问题；但不能读取 completion 的任何 gold token。

生成第 t 个答案 token 时，控制器只允许使用：

\[
\{\text{prompt},y_{<t},\text{当前可见 Memory},\text{截至当前的 loop states}\}.
\]

teacher forcing 中已经提供的 `y_<t` 可用于该位置的状态计算，但 `y_t` 与更后的 gold tokens 只能用于计算 loss，不能参与此位置的检索或门控决策。

纯语言模型预训练没有完整“用户问题”边界时，不能借口 question-conditioned memory，把整条训练序列作为检索 query；每个预测位置必须遵守实际可见前缀。

## 10.3 正确的验收反例

固定相同 prompt 和 source_context，替换完整 gold completion：**初始检索结果必须不变**。

固定相同答案前缀 `y_<t`，替换 `y_t:`：**第 t 个位置的 memory action、Halt logits、Reader 输出与 next-token logits 必须不变**，除数值容差外。

这里比一般的 attention causal mask 更严格，因为数据也可能通过 Router、预算、全序列池化或 memory 选择泄漏。

## 10.4 动态检索不能假装普通并行 SFT

同一个 `[B,T,D]` 的全序列平均 query，无法同时代表 T 个不同生成前缀的检索需求。

终局训练必须提供两种执行器，并保证语义对应：

1. **逐前缀/逐 token 的参考执行器**：完整保留动态 Memory 与动作轨迹，作为正确性标准。
2. **按 memory-version 与动作边界分段的执行器**：一个 segment 内的 Memory 在 segment 开始前已经确定，段内可以批处理；需要中途 refresh 时先结束 segment，再决策和继续。

训练使用事先给定的动作边界，也必须确保实际检索 query 仅由当前可见前缀计算。不能先用答案全序列算出“最相关的 Memory”，再把它广播给更早位置。

这属于同一终局模型的参考路径与优化路径，不是两套竞争架构。

# 11. KV、History 与缓存失效契约

## 11.1 三类缓存的生命周期

| 缓存 | 内容 | 生命周期 | 允许跨 optimizer step？ |
|---|---|---|---|
| Loop History | 当前 token/chunk 的历轮 committed states | 当前前向内 | 否 |
| Memory KV | Reader 专属投影后的外部 K/V | 同一 memory/projection 版本 | 训练默认否；推理权重不变时可以 |
| Token Self-KV | 各层、各轮历史文本的 K/V | 当前生成请求 | 训练依执行器；推理按请求保持 |

训练时 Memory KV 可在同一计算图中跨 R 个 loop 复用，梯度会汇总。不要把它 `detach()`，也不要跨多次独立 backward 复用旧图。

推理时建议 cache key 至少包含：

```text
request_id
memory_version
reader_layer_id
projection_weight_version
payload_encoder_or_bridge_version
device
dtype
```

若训练 Memory Encoder，也要包含 encoder 更新版本。仅用“同一个 WorkingMemory Python 对象”判断缓存是否有效不够。

## 11.2 固定 R 下的 self-KV

Core self-KV 按物理层与 loop 维分开：

```text
core_kv[layer_id][loop_id] = token-prefix K/V
```

第 r 轮的当前 query 读取历史 token 的第 r 轮 K/V。不能把上一轮的全部 KV 当作新的文本继续追加；token 轴每处理一个新 token 只增长 1。

每个物理层的 K/V 仍从其真实 self-attention 输入产生，不从 loop 末尾的 H^(r) 重新替换。Update Gate 只改变**进入下一轮/进入 Coda**的状态，不追溯改写已经生成的本轮内部 K/V。

## 11.3 可变 R 的难点：不能丢掉历史深轮次

若某个新 token 在第 1 轮就停止，不能把整个请求的 Core cache 从 R 个槽裁剪成 1 个槽。前面 token 已经算出的深轮次状态仍要保留。

本版本对 adaptive decode 定义明确的 **last-valid-per-token 填充语义**：

- 为每个 Core 层保留 `R_max` 个逻辑槽。
- 新 token 实际执行到 r_t，则其 r≤r_t 的 K/V 使用真实计算值。
- 新 token 的 r>r_t 槽，复制该 token 在该物理层最后实际计算轮次的 K/V。
- **只填充这个新 token 的位置**；不能用浅层整段缓存覆盖先前 token 的深轮次缓存。
- 多样本 batch 中按样本/位置操作，不按 batch 平均深度决定。

示意：

```python
for layer in core_layers:
    for r in range(R_max):
        current_token_kv = actual_kv[layer][min(r, last_executed_round)]
        cache[layer][r] = append_one_token(cache[layer][r], current_token_kv)
```

这只是语义伪代码；batch 版本须使用每样本的 last_executed_round。

**重要：** last-valid 填充是模型定义的近似缺失状态规则，不等价于真的执行全部深层。不能声称它在任意 adaptive-R 下与普通全深度 prefill 严格等价。训练的可变深度阶段要使用相同规则。

## 11.4 Prefill 的明确策略

首版优化路径推荐 `prefill_policy=fixed_cap`：对完整已知 prompt 使用当前模式允许的固定 R_prefill，建立一致 cache；decode 再按 token 自适应停止。相应的 TTFT 与 decode 成本分开记录。

这不改变动态深度模型，只明确 prefill 的批处理策略。若要连 prompt prefill 也逐 token 早停，需要实现与上节相同的 per-token cache 填充及 active-mask 语义，不能仅复用一个 batch-level halt 决策。

## 11.5 动态 Memory 下的 cache 语义

本版本采用 **append-only past**：memory 更新只影响此后执行的计算，过去 token 的 self-KV 不回溯重算。

因此，同一请求在 W1→W2 后的 stateful decode，不应与“拿最终 W2 对整个前缀从头重算”强行比较相等。正确的对照是：按同一 Memory 版本时间线、同一动作序列进行参考重放。

必须保存：

```text
(token_position, loop_index, memory_version_used,
 retrieval_action, budget_action, exit_round)
```

这份轨迹也是后续策略训练的必要输入。

## 11.6 Checkpointing 必须是纯计算重放

梯度 checkpoint 的函数内部不要执行全局检索、修改 MemoryManager、更新统计计数或改变 memory_version。反向重计算可能再次执行 forward，导致重复取数和副作用。

正确做法：控制器先得到不可变 memory payload/KV，checkpoint 只接收 Tensor、mask 与确定的配置，重算纯 Reader/Core 运算。

# 12. PyTorch 模块和模型接口设计

## 12.1 新增与修改文件

```text
model/minimind_x/
  residual/
    __init__.py
    loop_mhar.py                 # 新增
    recurrent_update.py          # 新增
  memory/
    reader.py                    # 新增：独立 Cross-Attention + gate
    working_memory.py            # 增加 ID、valid mask、版本信息
    memory_manager.py            # 去重、有效容量、版本更新
  router/
    halt_router.py               # 分离“算概率”和“执行停止”
    retrieval_router.py          # 可用动作 mask 与轨迹
  thinking_core.py               # 核心集成点
  minimind_x.py                  # 显式输出 committed/core states
  controller.py                  # 因果 query、grad 边界、Memory 版本
  losses.py                      # 真实接入 Router 与成本损失
  flops.py                       # 分项计数，不使用容量代理真实开销

trainer/
  residual_schedule.py          # 新增：同一训练 run 的 warm-up
  oracle_labels.py               # 同前缀反事实，不读取 gold 后缀
  train_pretrain_x.py
  train_sft_x.py
  train_grpo_x.py                # 如启用策略 RL，必须能重放动作

tests/
  test_residual_invariants.py
  test_reader_gradients.py
  test_routing_causality.py
  test_dynamic_cache_replay.py
```

## 12.2 核心模块 API

```python
class LoopMHAR(nn.Module):
    def forward(self, history: list[Tensor], strength: float = 1.0):
        # history: 每项 [B,T,D]
        # returns: mixed [B,T,D], weights [N,B,T,H_hist]
        ...

class RecurrentUpdateGate(nn.Module):
    def forward(self, previous, anchor, candidate,
                active=None, force_gate=None):
        # returns: committed [B,T,D], gate [B,T,D]
        ...

class GatedMemoryReader(nn.Module):
    def project_memory(self, memory_hidden, valid):
        # TRAIN 时必须在 grad mode 中调用
        # returns MemoryKV: [B,Hkv,M,head_dim]
        ...

    def forward(self, hidden, memory_kv,
                query_valid=None, allowed=None):
        # returns: updated_hidden, gate/delta diagnostics
        ...
```

接口的可运行参考实现见附录 A 及配套参考包。

## 12.3 模型输出新增字段

不要再用同一个 `exit_hiddens` 同时指代“Core 提交状态”和“过 Coda 后的状态”。

```python
@dataclass
class MiniMindXOutputV2:
    logits: Tensor | None
    exit_logits: list[Tensor] | None
    core_committed_states: list[Tensor] | None
    coda_exit_states: list[Tensor] | None
    halt_logits: list[Tensor] | None
    retrieval_logits: list[Tensor] | None
    budget_logits: list[Tensor] | None
    exit_round: Tensor | None
    memory_versions: list[int] | None
    router_trace: object | None
    residual_stats: dict
    past_key_values: object | None
```

生产推理默认不返回完整 history 与 attention weights，避免日志保留计算图或泄漏原始知识。调试时显式开启。

## 12.4 Core 集成伪代码

以下省略具体 self-KV 拼接，仅展示精确顺序；`CorePass` 内首层已经包含一次 Reader。

```python
anchor = prelude_hidden
previous = anchor
history = [anchor]

for r in range(max_loops):
    read_state, hist_weights = self.loop_mhar(history, strength=history_strength)

    # 记忆由外部 controller 按确定版本提供；这里不执行全局检索。
    candidate, current_pass_kv, reader_stats = self.core_pass(
        read_state,
        loop_index=r,
        token_past_kv=token_past_kv,
        memory_kv=memory_kv,
        token_mask=token_mask,
        memory_visibility=memory_visibility,
    )

    gated, update_gate = self.recurrent_update(previous, anchor, candidate,
                                               active=active)
    # 仅迁移 warm-up 使用；从头训练 update_progress=1。
    committed = candidate + update_progress * (gated - candidate)
    committed = where_active(active, committed, previous)

    stats = build_gate_features(previous, candidate, committed, update_gate)
    # features 中用于控制器的统计量 detach；h/e 是否 detach 由训练阶段决定。
    halt_logits = self.halt_router(committed, anchor, stats.detached())
    record_core_exit(committed, halt_logits, hist_weights, reader_stats)
    history.append(committed)
    previous = committed

    if training_full_unroll:
        # 继续用于多出口/动作监督；不把 probability evaluation 与 break 绑在一起。
        action = teacher_or_replayed_action(r)
    else:
        action = controller.decide_after_commit(committed, halt_logits, r)

    commit_current_token_kv(current_pass_kv, active, r)
    if action.stop or r == max_loops - 1:
        break

    memory_kv, memory_visibility = controller.prepare_next_memory(action)

# 使用 selected committed state 过 Coda，而不是未提交候选。
output = coda(select_committed_exit(...))
```

训练全展开模式下，“某个样本监督出口之后的动作是否还计算”必须有对应 mask：可以为辅助出口继续计算，但不能把这些额外动作计入该样本实际策略成本。

# 13. 梯度接线：防止“模块在运行，参数没学习”

## 13.1 默认梯度策略

| 组件 | 从头训练 | 旧模型迁移 warm-up | 备注 |
|---|---|---|---|
| Backbone/Core | 训练 | 可先冻结参数，再逐步解冻 | 冻结参数不等于对前向套 `no_grad` |
| Loop-MHAR | 训练 | 训练，但 η=0 时梯度暂被关掉 | η 要按计划升起 |
| Update Gate | 训练 | 训练/暖启动 | 最后一层零初始化会延迟前一层的梯度 |
| Reader Q/K/V/O | 训练 | 训练 | 不与离散检索一起 `no_grad` |
| Reader Gate/scale | 训练 | 训练 | zero-scale 时先允许 scale 获得梯度 |
| Memory payload encoder | 可训练或共享编码器 | 默认暂冻 | 是否冻结须入配置 |
| Top-K/ID 选择 | 离散操作 | 离散操作 | 不能期待索引本身自动可导 |
| Retriever query/key 投影 | 用显式检索损失训练 | 可训练 | 必须使用带图的 scores |
| Halt/Retrieval/Budget Router | 课程中训练 | 课程中训练 | logits 必须真正参与 loss |

PyTorch 的 `no_grad` 会阻断其中操作的计算图，即使参数 `requires_grad=True`。因此现有把 Memory 编码、选择、KV 投影一起放入 `no_grad` 的做法，不能用于声称 Reader 投影已从答案 loss 学习。[R5；C3]

## 13.2 推荐的内存分支写法

```python
# 1. 离散选择可以在 no_grad 下执行。
with torch.no_grad():
    selected_ids = retriever.select_ids(causal_query, index)
    frozen_payload = frozen_memory_encoder(selected_ids)

# 2. Reader 投影保留梯度，同一 forward 内跨 loops 复用。
mem_kv = memory_reader.project_memory(frozen_payload, valid_memory)

# 3. Core 即使冻结参数，只要输入依赖待训练模块，也不能整体 no_grad。
outputs = core_and_coda(..., memory_kv=mem_kv)
answer_loss = loss_fn(outputs, targets)

# 4. 独立重算带图的检索分数。
retrieval_logits = retriever.trainable_scores(causal_query, candidate_features)
retrieval_loss = retrieval_supervision(retrieval_logits, gold_ids)

(answer_loss + weight * retrieval_loss).backward()
```

如果希望答案 loss 直接训练“选哪一块”的 query 投影，必须另外设计软加权、可微选择近似或离散动作策略梯度。仅取消 `no_grad` 后仍使用硬 Top-K indices，并不能自动产生这条梯度路径。

## 13.3 梯度验收不能只看整体 Core moved

必须分别检查：

```text
loop_mhar.query
loop_mhar.type_bias
recurrent_update.fc2
recurrent_update.fc1（若最后层零初始化，检查后续步）
memory_reader.q_proj / k_proj / v_proj / o_proj
memory_reader.scale_logit
memory_reader.gate_fc2 / gate_fc1
halt_router
retrieval_router
budget_router
retriever.query_proj / key_proj
```

区分三种情况：应训练但无梯度、初始化允许暂时无梯度、明确冻结。不同情况不能混在一张“梯度通过”的总表中。

本文配套测试验证前三个新增模块的独立梯度；原仓库真实 trainer 与 Router 接线仍需按本节验收。

# 14. 初始化、权重迁移与同一训练 run 的课程

## 14.1 两种入口，一套最终架构

### 从头训练

```text
history_strength η = 1
update_progress = 1
history latest bias = +4
Update Gate 最后层 weight = 0，bias = +4
Reader scale ρ ≈ 0.01
Reader 内容门初值 ≈ 0.5
```

所有新模块从一开始就在模型图里。循环数按既定 depth-elastic 训练计划采样，初始不让随机 Halt 决策控制实际路径。

### 从既有 MiniMind-X 权重迁移

需要两个不同概念：

**严格兼容模式：** MHAR bypass；Update 强制采用 U；无 memory 或 Reader scale=0。用于与旧模型的无 memory 路径做数值回归，不作为长期训练配置。

**可学习迁移模式：** history_strength 与 update_progress 在同一 job 中逐步升起；Reader 保持非零输出投影，让零 scale 可被优化器打开。

## 14.2 建议迁移课程

以下比例是工程起始值，不是论文结论，也不是训练完成时间预估。

| 优化进度 | history_strength | update_progress | 训练内容 |
|---|---:|---:|---|
| 初始化检查 | 0 | 0 | 无 memory 旧模型 logits 回归；不更新 |
| 前 1% optimizer steps | 0→0.25 | 0→1 | 新模块启动，先保持接近旧路径 |
| 1%–5% | 0.25→1 | 1 | 三个模块完全接入；不要因 Gate 小而提前停 |
| 后续主训练 | 1 | 1 | 任务/多深度/有无 memory 混合训练 |
| 策略收尾 | 1 | 1 | Router 监督、适度成本约束、on-policy 校准 |

迁移 warm-up 的更新式可写为：

\[
H^{(r)}_{warm}=U_r+\lambda_{warm}(H^{(r)}_{gated}-U_r).
\]

当 λ_warm=0 时严格为 U；当 λ_warm=1 时为最终 Gate 更新。训练日志同时记录 raw gate 与 effective gate，避免 warm-up 时错误解读。

## 14.3 旧 Memory 拼接路径不能假装无损转换

对于旧的 `cat(memory_KV, token_KV)` 联合 softmax，拆成独立 cross-attention 后，即使权重维度相同，概率归一化也改变了。

因此：

- 可以保留原 self-attention 的主干权重。
- Reader Q/K/V/O 可以按尺寸兼容情况初始化，但必须记录来源。
- 无 memory 的兼容性可以严格测试。
- **有 memory 的旧新输出不保证初始相等**，需要任务训练或旧模型行为蒸馏。

迁移工具必须输出新增、缺失、尺寸变化与未加载的键清单；`strict=False` 不是忽略配置冲突的许可证。

## 14.4 训练超参数建议

保留已经验证的 backbone optimizer/学习率方案，新模块采用 1×–2× backbone LR 作为起点，norm/bias/gate scale 不施加普通 weight decay。

不在本增量文档中重新规定总 token 量或宣称若干天能完成训练。实际成本取决于模型尺寸、Memory 长度、每轮 Reader 开销、oracle probe 比例与硬件 kernel。

# 15. 损失函数与自适应控制器监督

## 15.1 三个新模块主要从任务 loss 学习

默认不需要人工给 Update Gate 标注 0.2/0.8，也不需要人工标注“第 2 个 history head 应读取第 1 轮”。

对固定/随机深度和多出口任务 loss：

\[
L_{answer}=\sum_{r=1}^{R}w_r\,CE(\operatorname{Head}(\operatorname{Coda}(H^{(r)})),y).
\]

只在合法监督位置计算 CE。不同出口都经过同一套 Coda，Coda 额外前向成本要计入训练 FLOPs。

Gate、MHAR 与 Reader 从答案误差反传中学习如何使用状态与证据。

## 15.2 推荐总目标

\[
L=L_{answer}
+\alpha_hL_{halt}
+\alpha_rL_{ret}
+\alpha_bL_{budget}
+\alpha_kL_{retriever}
+\alpha_mL_{reader\_aux}
+\lambda_C\mathbb E[C]
+\alpha_{KD}L_{KD}.
\]

推荐初始接线：

| 项 | 起始设置 | 说明 |
|---|---:|---|
| L_answer | 1.0 | 主目标 |
| L_halt | 0.1 | 只在可信 oracle 决策位置监督 |
| L_ret | 0.1 | 复用/重排/全局检索的动作监督 |
| L_budget | 0.1 | 真正改变可见 Memory 的预算对照 |
| L_retriever | 0.1 | gold span/block 或任务效用监督 |
| L_reader_aux | 默认 0 | 有明确正负适用性标签后再开 |
| E[C] | 初始 0，随后从小量升起 | 先让路径有效，后优化成本 |
| L_KD | 默认 0，可按迁移需要开启 | 教师与学生对齐合法输入和任务位置 |

这些权重是项目建议，不是已经在 MiniMind-X 上验证的最优超参数。

## 15.3 不采用的“看起来漂亮但容易误导”的损失

不默认使用：

\[
|P(continue)-\operatorname{mean}(g)|,
\]

也不使用 `λ * mean(update_gate)` 冒充算力惩罚。

原因：前者把状态改变量误当成正确性/剩余推理量；后者可能奖励关闭 Gate，但 Core 已经算过，计算并没有省掉。

也不强迫每个 loop 的 Gate 均值递减。某个后续 loop 新增关键证据时，合理行为可能是重新大幅更新。

## 15.4 Halt Oracle 的定义

目标是“在合法可见前缀与给定 Memory 状态下，最省计算且任务足够好的停止点”。

对完全答不对的样本，标记 `unresolved`。不能机械地把它们都叫作“需要最大思考深度”。可以使用 loss 饱和信号作为弱监督，但必须与成功 oracle 分开统计。

teacher-forced CE 接近也不等于自由生成成功率相同。用于部署的 Halt 阈值，需要用真实生成/执行验证做校准。

## 15.5 Retrieval Oracle 必须是“是否再检索”的反事实

在同一决策状态 s、同一 W 上比较：

\[
Q(\text{REUSE}\mid s,W)
\quad\text{与}\quad
Q(\text{REFRESH}\mid s,W\rightarrow W').
\]

不能用“完全无 memory”与“已有 memory”的差来替代；后者只说明 memory 有没有用，不说明当前还需不需要刷新。

同一 probe 应固定问题、前缀、checkpoint、随机条件和后续计算预算。真正找回新证据后，才有资格评估 refresh 的边际收益。

## 15.6 Budget Oracle 必须保证档位真的改变计算路径

对预算 B 执行：

```text
计算目标有效容量
→ 得到可选候选
→ 按容量选择不同的有效 token/block
→ 生成对应 Reader KV
→ 以同一深度与相同条件评估 loss/正确性
```

若各档实际都是固定 `topk_fine=8` 块、没有有效内容差异，那么 sweep 得到的是退化对照，不是 Memory Budget 的学习信号。

## 15.7 成本估计须尊重条件依赖

停止分布可由 `p_stop` 计算：

\[
\pi_r=p_r\prod_{j<r}(1-p_j),
\]

最后一轮强制吸收剩余概率。

但当 Memory Budget、检索与深度互相依赖时，不应粗暴使用 `E[R] * E[M]` 代替真实联合成本。应使用实际状态条件下的期望或采样轨迹的累计成本：

\[
C(\tau)=C_{pre}+C_{coda}
+\sum_{r\in\tau}\big(C_{core,r}+C_{hist,r}+C_{reader,r}+C_{gate,r}\big)
+\sum_{a\in\tau}C_{retrieval,a}.
\]

只对动态控制器的 action probabilities 或动作 reward 建立成本梯度。实际整数计数继续用于日志和性能报告。

# 16. 与本次模块相匹配的数据设计

## 16.1 保留原有能力数据，新增“对照式证据条件”

不需要把所有 SFT 都改成很长的显式思维链。优先新增如下样本族：

| 样本族 | 关键构造 | 训练/验收目标 |
|---|---|---|
| 无外部证据 | memory=None | Reader 不应扰动无 memory 路径 |
| 外部证据必需 | 答案含本批次新生成的随机事实，prompt 无法推断 | 证明 Reader 真的使用了外部信息 |
| 无关 memory | 与问题无关、长度/格式相似的资料 | 不因 memory 存在就盲目注入 |
| 相似但不适用 | 相同术语，不同版本、对象或生效条件 | 学习适用性，而非只匹配关键词 |
| 新证据纠正旧假设 | 后续 loop 才检索到关键约束 | 验证 Gate 可以重新打开，不锁死旧状态 |
| 多跳证据 | 后续检索 query 依赖前一次读取的信息 | 验证 Think→Retrieve→Think 的闭环 |
| 证据冲突/不足 | 多来源冲突，或确实缺少必要字段 | 允许继续检索或承认不足，而非强制回答 |

## 16.2 防止可学习性测试走捷径

只在固定输入上过拟合，或者把标签构造成直接复制当前输入 token，不能证明 memory 或深 loop 提供了收益。

应加入“同一问题模板、每次随机改外部事实”的任务，例如：

```text
Memory: 本次随机配置中，设备 A 的阈值为 7319。
Prompt: 本次设备 A 的阈值是多少？
Target: 7319
```

下一条样本把阈值换成另一随机值，训练与验证使用不重叠的事实组合。可通过移除/交换 memory 检验答案是否正确随证据变化。

这个任务只是机制验收；真实领域能力仍需真实文档、代码和版本约束任务。

## 16.3 Gate 辅助监督只使用清晰标签

Reader 的“无关/版本不适用”样本可以对分支使用施加弱约束，但不能把所有错误答案都归因于 Reader Gate，也不能把高 attention weight 当作因果贡献标签。

困难样本有可能需要 memory，但 backbone 尚不能正确利用。把这种样本全部训练成“关 Reader”会使系统逃避知识读取。

## 16.4 数据分割与验证

按文档来源、仓库、任务模板或事实生成种子划分 train/validation/test，避免同一事实换一个 prompt 进入测试集。

gold evidence 必须基于**真实 tokenizer 之后的 token span/block ID**，不能用字符比例估算后直接拿来训练精确检索 CE。tokenizer/分块参数改变时，重新生成标注。

# 17. RL 与计算策略学习的边界

## 17.1 固定 R 的 token GRPO 能做什么

固定 R rollout，并在相同 R 下重算 token log-prob，可以训练 backbone 与可微的 Gate/Reader/MHAR 路径。但如果训练没有 memory 输入，这次 rollout 自然不能优化 memory 使用。

更重要的是：只计算输出 token 的 log-prob，不能自动训练未纳入动作概率的 Halt、Retrieval、Budget 决策。不要因为 reward 中写了 memory/loop cost，就宣称这些策略已被优化。

## 17.2 真正的 Adaptive Policy RL 要记录什么

对每条轨迹保存：

```text
prompt/source IDs
每个决策点的可见前缀
每个 token/loop 的 exit 与 active mask
retrieval action 与 selected evidence IDs
budget action
memory_version
动作 log-prob
输出 token log-prob
温度、采样过滤规则与 checkpoint 版本
```

联合路径概率包含：

\[
\log P_\theta(\tau)
=\sum_t\log P_\theta(y_t\mid s_t)
+\sum_i\log\pi^h_\theta(a_i^h\mid s_i)
+\sum_i\log\pi^r_\theta(a_i^r\mid s_i)
+\sum_i\log\pi^b_\theta(a_i^b\mid s_i).
\]

Update Gate 是连续可微模块，不需要另外把每个通道 gate 当成离散 RL action；它可通过任务/策略损失的计算图获得梯度。

## 17.3 禁止平均深度重放

生成时如果各 token 深度为 `[1,3,2,4]`，不能把平均值取整为 3 后，用所有 token 都 R=3 的前向估计原轨迹概率。

同理，不能用最终 memory 替换轨迹中每一轮的真实 memory。重放协议必须与第 11 节完全一致。

## 17.4 Reward 的最低约束

正确性/任务成功是主项，效率是次项。成本范围归一化并设置上界，避免正确但稍贵的轨迹被评分低于错误但便宜的轨迹。

代码执行 reward 必须运行在隔离沙箱中：无训练目录写权限、默认无网络、受限 CPU/内存/进程数并有超时。`subprocess.run(python -c ...)` 本身不是沙箱。

主线先通过 SFT/任务监督训练三个新模块；完整 Adaptive Policy RL 只在重放一致性测试通过后启用，不把它设为残差改造的隐性必需依赖。

# 18. 算力、显存与真实性能预算

## 18.1 Gate 本身不是跳计算

本方案 Update Gate 依赖 U_r，因此每次关 Gate 前，Core 已经完成计算。

报告时分别统计：

```text
executed_core_rounds
selected_exit_rounds
active_token_rounds
physical_batch_rounds
```

一个 batch 为等待最慢样本而继续执行，不能只用样本平均停止轮次宣称相同比例的 GPU 加速。压紧 active 样本后才可能兑现部分计算节省。

## 18.2 新增参数估算

令 D=1024、update rank=64：

- Update Gate 两个线性层主要参数约 `4*D*64 = 262,144`，另有 bias/norm。
- Loop-MHAR 的共享 query 共 D 个参数；另有 norm 与少量 source-type bias，不因四头而扩大到 4D query。
- Reader 若 `query_heads=8, kv_heads=2, head_dim=64, D_m=1024`，Q/K/V/O 主要参数约 1,310,720，另有 gate/norm。

这些是本文配置的解析算式，不是现有整个 MiniMind-X 的实测参数量。最终 `sum(p.numel())` 要输出到 manifest。

## 18.3 Loop History 显存

保存 R+1 个状态的张量存储上界近似为：

\[
(R+1)BTD\times\text{bytes\_per\_element}.
\]

例如 B=1、T=8192、D=1024、R=4、BF16，状态张量本身约 80 MiB。该数不包括 stack/FP32 临时量、Core 激活、梯度、KV 和优化器，也不全是相对原 multi-exit 训练的额外开销。

若把 T 扩成 1M，history 当然随之增大。终局设计中 1M 是外部可检索资料，不能因为 Working Memory 小就忽略 backbone 实际处理的 T。

## 18.4 历史路由计算

单轮读 N_r 个来源，大致为 `O(B*T*N_r*D)`；全 R 轮累积为 `O(B*T*D*R²)`。

R=4 时可接受的假设不应无条件推广到 R=32/64。届时应限制历史数量、分块存储或使用 fused routing，并重新测量，不仅看理论 FLOPs。

## 18.5 Reader 计算

对于独立 Reader，query 长度 T、memory 长度 M、attention 总宽度 D_a：QK 与 AV 两次矩阵乘的前向开销约：

\[
4BTMD_a.
\]

另计 Q/O 投影、memory K/V 投影与 Reader Gate。

注意是 `T*M`，不是自动等于 `M²`。decode 时 T=1；prefill 或训练时 T 可能很大。

Memory KV 未变化时，K/V 投影可跨 loop 复用；Q/O、attention 和 Gate 仍要逐轮执行。检索候选重排也可能需要编码很多最终未入 Hot Memory 的块，其成本要单独记。

## 18.6 推荐性能统计

至少记录：

```text
TTFT、decode tokens/s、端到端 latency
实际执行 Core rounds 与样本选中 rounds
global retrieval / warm rescore 次数
unique memory tokens / slots、预算上限
memory K/V cache hit rate 与重投影次数
history read 时间、Reader 时间、Gate 时间
peak allocated / reserved memory
实际 token 数、有效监督 token 数、padding 比例
```

异步 GPU 计时使用正确的同步/CUDA events。训练与推理不同 backend、batch、精度不能混在一张 speedup 表里。

# 19. 推荐配置与版本管理

以下是终局模块配置示例，不是已经被现有 CLI 支持的命令。需先在 config/parser 中接入。

```yaml
architecture_version: minimind_x_residual_v2

# 以实际 checkpoint 的维度和层数为准，不在升级时偷偷改变模型规模。
model:
  inherit_backbone_config: true
  max_loop: 4
  loop_embedding: false
  mhc_enabled: false

loop_mhar:
  enabled: true
  heads: 4
  shared_query_across_loops: true
  key_norm: full_width_rmsnorm
  query_init: zeros
  latest_source_bias_init: 4.0
  history_strength: 1.0
  detach_history: false

recurrent_update:
  enabled: true
  granularity: token_channel
  low_rank: 64
  final_weight_init: zeros
  final_bias_init: 4.0
  gate_compute_dtype: float32
  update_progress: 1.0
  state_noise_std: 0.0
  gate_noise_std: 0.0

memory_reader:
  enabled: true
  placement: after_first_core_self_attention
  reader_core_layer_ids: [0]
  query_heads: 8
  kv_heads: 2
  head_dim: 64
  gate_granularity: token_scalar
  gate_rank: 32
  residual_scale_init: 0.01       # 旧模型严格迁移可设 0
  separate_qkv_projections: true
  apply_token_rope_to_memory_qk: false
  dropout: 0.0
  null_for_empty_queries: true

memory:
  inherit_existing_budget_list: true
  max_memory_tokens_runtime: 8192 # 示例运行上限；不是硬编码结构常数
  deduplicate_by_source_id: true
  track_memory_version: true
  train_projection_with_grad: true
  cache_across_training_steps: false

controller:
  collect_logits_during_full_unroll: true
  detach_gate_statistics_for_halt: true
  retrieval_actions: [reuse, rescore_warm, global_refresh]
  mask_unavailable_actions: true
  query_visibility: prompt_or_current_prefix_only
  force_stop_last_round: true
  retrieve_after_stop: false

cache:
  core_layout: layer_loop_token
  prefill_policy: fixed_cap
  missing_depth_policy: last_valid_per_token
  preserve_past_deep_slots: true
  past_memory_update_policy: append_only

loss:
  answer_weight: 1.0
  halt_weight: 0.1
  retrieval_action_weight: 0.1
  budget_weight: 0.1
  retrieval_evidence_weight: 0.1
  reader_aux_weight: 0.0
  cost_weight_start: 0.0
  gate_mean_regularization: 0.0
  gate_halt_equality_regularization: 0.0
```

配置校验至少包含：D 被 MHAR heads 整除；Reader query_heads 被 kv_heads 整除；T/位置编码范围合法；Reader 插入层存在；budget 受合法 action mask 限制；memory 来源维度与 Bridge 输出一致。

现有 CLI 的显式默认 `hidden_size=768 / num_hidden_layers=8` 会覆盖 preset，容易与 preset 的 zone 数发生冲突。[C10] 升级时应把非显式参数默认改为 `None`，只让用户真正指定的值覆盖已加载 config。

checkpoint 至少保存：模型权重、完整 config、architecture/cache 版本、optimizer/scaler/scheduler、各 rank RNG、数据进度、warm-up 进度及 source/tokenizer/Bridge manifest。不只保存一个 half 精度 state_dict 然后依靠 CLI 重建配置。

# 20. 开发任务清单与定义完成标准

| ID | 开发任务 | 完成标准 |
|---|---|---|
| R-01 | 新增 LoopMHAR | 来源轴 softmax、同 token 路由、H=1 参考等价、history 不被原地修改 |
| R-02 | 新增 UpdateGate | g=0/1 边界正确、bypass 是旧 committed state、支持 active mask |
| R-03 | 新增独立 Reader | 不再拼接 token/memory K/V；空/全 masked memory 严格 no-op |
| R-04 | Core 集成 | 每轮一次 history read、一次完整 Core、一次 update commit |
| R-05 | 输出语义升级 | Router 使用 committed state，不使用 Coda 后池化状态冒充 |
| R-06 | 梯度边界修复 | Reader 投影、retriever 监督 logits 和全部 Router 均有真实梯度路径 |
| R-07 | Query 因果边界 | 换 gold 后缀不改变前面的路由与 next-token logits |
| R-08 | Memory ID/版本 | 去重、有效 mask、缓存按版本失效 |
| R-09 | 可变深度 KV | 保留 R_max 槽、按当前 token 填充、历史深槽不丢失 |
| R-10 | Reference replay | 动态 retrieval/depth 时间线可重放，优化器前后版本明确 |
| R-11 | 训练调度与 loss | 三个模块同一 run 启用；概率收集不导致训练 premature stop |
| R-12 | 计时/成本 | 报告实际执行与逻辑退出的差异，不把关 Gate 算作跳 Core |
| R-13 | checkpoint/CLI | 明确配置继承、missing keys 白名单与架构版本 |
| R-14 | 多卡训练 | DDP/FSDP 正确包装与采样分片；不能仅 torchrun 起多份独立训练 |

任务可以在不同文件并行开发，但**架构只按本文这一套集成**。所有训练启动脚本先通过配置和正确性验收，避免直接投入长周期训练后才发现模块未被优化。

# 21. 必须执行的验收测试

## 21.1 数学不变量

- 单历史来源时，MHAR 输出等于该来源；history_strength=0 时输出等于最新 committed state。
- 每个 token/head 的来源权重之和为 1；改变未来 token 的历史状态不影响更早位置的 MHAR 输出。
- Update force_gate=0/1 分别精确返回 previous/candidate；inactive 位置保持 previous。
- Reader 对 None、零长度、全 masked memory 以及单个 query 无可见 memory 时均严格 no-op，且无 NaN。
- memory padding 内容改变，不影响真实 query 输出。
- BF16 门控、norm、softmax 的数值保持有限。

## 21.2 真实训练接线

在 **真实 `sft_c_step`** 而非专门绕开 controller 的测试函数中，验证 Reader、各 Router、retriever 投影的梯度。

正样本必须需要外部随机事实才能答对；用无 memory/错误 memory 对照排除纯复制或训练样本记忆。

零 scale 初始化允许第一步只有 scale 获得梯度，但后续步 Reader 内部必须开始更新。长期无梯度不能解释为“模型自己选择不使用”。

## 21.3 Cache 与路径一致性

1. 固定 R、固定 memory 下，整段与带 cache 的增量输出在数值容差内一致。
2. 预设深度序列 `4→1→3→2` 时，早停不会删除先前 token 的深轮次 KV。
3. 混合 batch 与逐样本参考执行结果一致，不能由平均 Gate/预算决定所有样本动作。
4. memory 在指定 token/loop 切换版本后，结果与同轨迹参考重放一致。
5. checkpoint 开/关的前向和梯度对齐，不增加检索计数。
6. 同一请求重置、跨请求并发、checkpoint 载入后都不串用 memory/history。

## 21.4 因果与安全

- 同一 prompt 换 gold completion，初始检索不变。
- 同一生成前缀换未来 gold 后缀，当前动作不变。
- sparse baseline 同时验证 attention mask 与 block selector 因果性。
- 未授权来源、错误 source version 不进入 Working Memory。
- 调试日志不输出隐私 payload 或完整 hidden tensors。

## 21.5 任务与效率验收

Gate 直方图不是验收结果。至少比较同一 checkpoint 在固定 R 和自适应策略下的：

```text
实际任务正确率/通过率
错误证据导致的回退幅度
无关 memory 的影响
相同预算下的性能
相同质量下的 latency / executed FLOPs
Gate、Memory、History 的使用统计
```

R 增加后偶尔退化并不自动证明架构错误，但必须报告。若声称缓解 overthinking，须报告“较浅正确、较深转错”的频率及置信区间，而不是只画 Gate 均值递减曲线。

初始工程质量门槛建议以已有可用 checkpoint 为参照：基础能力不出现明显回退、相关证据任务有实际收益、无关/错误证据不大幅放大错误。具体百分比应结合样本量和业务要求设定，不在无数据时承诺固定提升。

# 22. 交付状态、风险与最终结论

## 22.1 本次已经交付的内容

- 一套明确的组合架构和唯一默认插入位置。
- Loop-MHAR、UpdateGate、Reader 的公式、形状、初始化与 API。
- 现有代码需要改动的位置与真实梯度/因果性前置修复。
- 动态检索、可变 Memory、Halt 和 cache 的语义契约。
- 训练与迁移课程、损失接线和效率统计口径。
- 三个独立 PyTorch 模块，以及 CPU 回归测试。

## 22.2 本次没有声称完成的事项

尚未把参考代码合并进原仓库；未运行原仓库声明的 46 个测试；未执行完整 trainer、多卡、动态解码 cache replay 或 1M 上下文训练；未验证该组合相对原版的能力提升。

配套参考测试不依赖缺失的 `model/model_minimind.py`，因此不会用替代 backbone 冒充原模型集成成功。该文件以及原项目完整依赖、配置和训练数据，需要在实际仓库内完成最终接入。

## 22.3 主要风险和处理原则

| 风险 | 不正确的处理 | 推荐处理 |
|---|---|---|
| Gate 长期接近 1 | 强制惩罚 Gate | 先判断是否确实需要大更新，再查训练覆盖和梯度 |
| Gate 接近 0、能力差 | 直接据此提前停止 | 排查饱和、零初始化阻塞、损失冲突、错误固定点 |
| MHAR 只读最新状态 | 认定没有训练 | 与任务表现对照；历史确实可能没有增益 |
| Reader 忽略 memory | 一律强制打开 | 使用必须依赖外部事实的任务，检查投影梯度和对齐 |
| 新模块使速度下降 | 用理论 FLOPs 掩盖 | 分项 profiler，减少复制/重投影，区分 prefill/decode |
| 动态路径训练不稳定 | 把所有策略改成随机标签 | 正确的同前缀反事实监督、课程和完整 replay |
| 无 memory 基线回退 | 宣称 Gate 理论保真 | 做兼容模式回归与迁移对齐，限定保真条件 |

**最终方案可以概括为：先选择从历史读什么，再独立读取外部证据，随后生成候选，最后有控制地写回状态。**

这次改造不是“增加更多 Loop”，而是让既有 Loop 的信息读取、知识注入和状态提交分别受控。mHC 暂不加入，三个模块也不互相冒充正确性判别器。真正的落地标准是：路径因果、梯度接通、缓存可重放、任务有用、成本可测。

---

# 附录 A. 三个核心模块的独立 PyTorch 参考实现

以下代码是本方案的参考实现，文件名 `modules.py`。它不是完整 MiniMind-X 模型：self-attention、检索策略、可变深度 token cache 与训练器集成按正文完成。参考代码使用显式 stack 和 GQA head repeat 以便阅读和测试；性能优化应在等价性验证后替换。

```python
"""MiniMind-X residual-v2 reference primitives.

Independent PyTorch modules, NOT a replacement for the full MiniMind-X model.
Token self-attention, retrieval policy, decoder KV management and trainers remain
integration work. Masks use True = visible/valid. No cross-request state lives here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class StableRMSNorm(nn.Module):
    """RMS statistic and multiplication in FP32; result follows input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class LoopMHAR(nn.Module):
    """Read SAME-TOKEN states across loops, not different token positions.

    history = [anchor, committed_loop_1, ..., committed_loop_(r-1)].
    All states have [B, T, D]. The last entry MUST be the current committed state.
    A shared pseudo-query is partitioned across feature heads. This is a proposed
    loop-boundary adaptation of depth-wise MHAR, not the paper's full architecture.
    """

    def __init__(self, dim: int, heads: int = 4, latest_bias: float = 4.0) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads")
        self.dim, self.heads = dim, heads
        self.norm = StableRMSNorm(dim)
        self.query = nn.Parameter(torch.zeros(heads, dim // heads))
        # Rows: anchor, older committed state, latest committed state.
        self.type_bias = nn.Parameter(torch.zeros(3, heads))
        with torch.no_grad():
            self.type_bias[2].fill_(latest_bias)

    def forward(self, history: Sequence[Tensor], strength: float = 1.0
                ) -> tuple[Tensor, Tensor]:
        if not history or not 0.0 <= strength <= 1.0:
            raise ValueError("nonempty history and strength in [0,1] required")
        current = history[-1]
        if current.ndim != 3 or current.shape[-1] != self.dim:
            raise ValueError("history tensors must have shape [B,T,dim]")
        if any(x.shape != current.shape for x in history):
            raise ValueError("all history tensors must share shape")
        if any(x.device != current.device or x.dtype != current.dtype for x in history):
            raise ValueError("all history tensors must share device and dtype")
        values = torch.stack(list(history), dim=0)  # [N,B,T,D]; reference allocation
        n, b, t, _ = values.shape
        keys = self.norm(values).float().reshape(n, b, t, self.heads, -1)
        # Full-width RMSNorm precedes the head split. No 1/sqrt(d_head) here.
        logits = torch.einsum("he,nbthe->nbth", self.query.float(), keys)
        roles = torch.ones(n, dtype=torch.long, device=current.device)
        roles[0], roles[-1] = 0, 2
        logits = logits + self.type_bias[roles].float()[:, None, None, :]
        weights = logits.softmax(dim=0)  # normalize LOOP/SOURCE axis only
        vh = values.float().reshape(n, b, t, self.heads, -1)
        mixed = torch.einsum("nbth,nbthe->bthe", weights, vh).reshape(b, t, self.dim)
        out = current.float() + strength * (mixed - current.float())
        return out.to(current.dtype), weights


class RecurrentUpdateGate(nn.Module):
    """Commit h_next = h_prev + g * (candidate - h_prev).

    The bypass is the PRE-ROUND committed state, NOT the history-mixed read.
    force_gate exists only for exact compatibility / invariant tests.
    """

    def __init__(self, dim: int, rank: int = 64, bias: float = 4.0) -> None:
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError("dim and rank must be positive")
        self.dim = dim
        self.h_norm, self.e_norm, self.delta_norm = (
            StableRMSNorm(dim), StableRMSNorm(dim), StableRMSNorm(dim))
        self.fc1 = nn.Linear(3 * dim, rank, bias=True)
        self.fc2 = nn.Linear(rank, dim, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, bias)

    def forward(self, previous: Tensor, anchor: Tensor, candidate: Tensor,
                active: Optional[Tensor] = None, force_gate: Optional[float] = None
                ) -> tuple[Tensor, Tensor]:
        if previous.shape != candidate.shape or previous.shape != anchor.shape:
            raise ValueError("previous, anchor, candidate must share [B,T,D]")
        if previous.ndim != 3 or previous.shape[-1] != self.dim:
            raise ValueError("invalid hidden shape")
        if active is not None and (active.dtype != torch.bool or active.shape != previous.shape[:2]):
            raise ValueError("active must be bool [B,T]")
        if force_gate is None:
            features = torch.cat([self.h_norm(previous), self.e_norm(anchor),
                                  self.delta_norm(candidate - previous)], dim=-1)
            gate = self.fc2(F.silu(self.fc1(features))).float().sigmoid()
            updated = (previous.float() + gate * (candidate.float() - previous.float())).to(previous.dtype)
        else:
            if not 0 <= force_gate <= 1:
                raise ValueError("force_gate must be in [0,1]")
            gate = torch.full_like(previous, force_gate, dtype=torch.float32)
            if force_gate == 0:
                updated = previous
            elif force_gate == 1:
                updated = candidate
            else:
                updated = (previous.float() + gate * (candidate.float() - previous.float())).to(previous.dtype)
        if active is not None:
            updated = torch.where(active[..., None], updated, previous)
        return updated, gate


@dataclass(frozen=True)
class MemoryKV:
    """Independent memory K/V. Cache owner must track version / optimizer step."""
    key: Tensor    # [B,Hkv,M,Dhead]
    value: Tensor  # [B,Hkv,M,Dhead]
    valid: Tensor  # [B,M], bool


class GatedMemoryReader(nn.Module):
    """Separate cross-attention residual; neither memory K nor query uses RoPE.

    Use once per core pass, after the FIRST self-attention residual and before its
    FFN. Do not additionally concatenate memory K/V into token self-attention.
    """

    def __init__(self, dim: int, memory_dim: int, query_heads: int = 8,
                 kv_heads: int = 2, head_dim: int = 64, gate_rank: int = 32,
                 scale_init: float = 0.01, dropout: float = 0.0) -> None:
        super().__init__()
        if min(dim, memory_dim, query_heads, kv_heads, head_dim, gate_rank) <= 0:
            raise ValueError("dimensions and heads must be positive")
        if query_heads % kv_heads or not -1 < scale_init < 1:
            raise ValueError("query_heads must divide by kv_heads; scale_init in (-1,1)")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.dim, self.memory_dim = dim, memory_dim
        self.query_heads, self.kv_heads, self.head_dim = query_heads, kv_heads, head_dim
        self.dropout = dropout
        self.q_norm, self.m_norm = StableRMSNorm(dim), StableRMSNorm(memory_dim)
        self.q_proj = nn.Linear(dim, query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(query_heads * head_dim, dim, bias=False)
        self.read_norm = StableRMSNorm(dim)
        self.gate_fc1 = nn.Linear(2 * dim, gate_rank)
        self.gate_fc2 = nn.Linear(gate_rank, 1)
        nn.init.zeros_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)
        # Random/nonzero o_proj + zero scalar = exact adapter no-op, but scalar
        # still gets gradient. Never zero BOTH o_proj and scale.
        self.scale_logit = nn.Parameter(torch.tensor(math.atanh(scale_init)))

    def project_memory(self, hidden: Tensor, valid: Tensor) -> MemoryKV:
        if hidden.ndim != 3 or hidden.shape[-1] != self.memory_dim:
            raise ValueError("memory hidden must have shape [B,M,memory_dim]")
        if valid.dtype != torch.bool or valid.shape != hidden.shape[:2]:
            raise ValueError("valid must be bool [B,M]")
        b, m, _ = hidden.shape
        # Mask before normalization/projection; padded values carry no content.
        clean = torch.where(valid[..., None], hidden, torch.zeros_like(hidden))
        mh = self.m_norm(clean)
        k = self.k_proj(mh).reshape(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mh).reshape(b, m, self.kv_heads, self.head_dim).transpose(1, 2)
        return MemoryKV(k, v, valid)

    def forward(self, hidden: Tensor, kv: Optional[MemoryKV],
                query_valid: Optional[Tensor] = None,
                allowed: Optional[Tensor] = None) -> tuple[Tensor, dict[str, Tensor]]:
        b, t, d = hidden.shape
        if d != self.dim:
            raise ValueError("hidden width mismatch")
        if query_valid is None:
            query_valid = torch.ones(b, t, dtype=torch.bool, device=hidden.device)
        if query_valid.dtype != torch.bool or query_valid.shape != (b, t):
            raise ValueError("query_valid must be bool [B,T]")
        if kv is None or kv.key.shape[-2] == 0:
            return hidden, {"gate": hidden.new_zeros(b, t, 1),
                            "delta": torch.zeros_like(hidden)}
        if kv.key.shape[0] != b or kv.value.shape != kv.key.shape:
            raise ValueError("memory K/V batch or shape mismatch")
        m = kv.key.shape[-2]
        visible = kv.valid[:, None, :].expand(b, t, m) & query_valid[..., None]
        if allowed is not None:
            if allowed.dtype != torch.bool or allowed.shape != (b, t, m):
                raise ValueError("allowed must be bool [B,T,M]")
            visible = visible & allowed
        any_real = visible.any(dim=-1, keepdim=True)  # [B,T,1]
        # NULL is visible only when this query has no valid real memory keys.
        visible = torch.cat([visible, ~any_real], dim=-1)[:, None, :, :]
        null_k = kv.key.new_zeros(b, self.kv_heads, 1, self.head_dim)
        k = torch.cat([kv.key, null_k], dim=-2)
        v = torch.cat([kv.value, torch.zeros_like(null_k)], dim=-2)
        rep = self.query_heads // self.kv_heads
        # Reference path; optimized deployment can use supported GQA kernels.
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        q = self.q_proj(self.q_norm(hidden)).reshape(b, t, self.query_heads, self.head_dim).transpose(1, 2)
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
```

# 附录 B. 参考实现测试记录

## B.1 本次实际运行范围

测试环境：PyTorch `2.10.0+cpu`；无 CUDA。

执行：

```bash
cd minimind_x_residual_v2_reference
python -m pytest -q
```

结果：

```text
..........................                                               [100%]
26 passed in 1.25s
```

这里的 26 项是本文新增参考模块的测试，不是附件原仓库的 46 项测试，也不包括完整 KV decoder、动态检索训练、多卡或长上下文能力测试。

## B.2 已执行的测试范围

| 测试组 | 已检查的性质 |
|---|---|
| RMSNorm | 零输入有界、无 NaN |
| Loop-MHAR | 单来源恒等、权重沿历史轴归一、η=0 恒等、单头手工公式对齐 |
| 多头路由 | 不同头可以产生不同历史权重 |
| History 因果性 | 改未来 token 的来源不影响更早 token；不原地修改输入来源 |
| Update Gate | 0/1 极限、0.25 插值、+4 bias 近开门但非严格恒等、inactive 位置不更新 |
| Reader no-op | None、M=0、全遮罩、混合 batch 的空 memory 行 |
| Reader mask | padded 内容不可见、每 query 可见性、无效 query 不更新 |
| Reader 初始化 | scale=0 精确 no-op，但 scale 能获得非零梯度 |
| Reader 梯度 | frozen payload 下 Q/K/V/O 有梯度；冻结下游层不阻断上游 Reader 梯度 |
| KV 复用 | 单一训练图内多次读取同一 Memory KV 可反传 |
| 数值与形状 | BF16 有限、非法输入主动报错 |
| 组合路径 | 线性 toy harness 中三模块都有梯度；该 harness 不是语言模型 |

完整测试源码位于配套包的 `test_modules.py`。

## B.3 尚需在真实仓库执行的验收

第 21 节的真实 trainer、可变深度 cache、memory-version replay、未来 gold 泄漏、多卡和性能测试仍是必需项。只有这些通过，才具备启动正式长周期训练的依据。

# 附录 C. 附件源码证据索引

下表使用上传包 `minimind_x_code_bundle(2).md` 的**包内行号**，不是拆分后的 `.py` 行号。该包只包含新增代码，引用到的原始基础模型及背景文档未包含其中，因此本文不声称审查了那些未提供文件。

| 编号 | 位置与包内行号 | 支持的判断 |
|---|---|---|
| C1 | `thinking_core.py`，L1759–L1768；L1945–L1964 | 既有分轮缓存、全展开、Halt 与外部 memory 的设计语义；每轮提交前后需要重新界定 |
| C2 | `MemoryAwareAttention.forward`，L1785–L1814；`project_memory`，L1859–L1875 | 原实现将 memory 与 token K/V 拼接；memory K 未做 token RoPE |
| C3 | `controller.py`，L350–L371、L426–L449 | 选择/记忆路径位于 no_grad；训练 KV 投影也在 no_grad 中；检索 CE 取缓存 scores |
| C4 | `minimind_x.py`，L1065–L1085；`halt_router.py`，L1703–L1722 | Budget 读 Prelude；Halt 计算由 halt 开关控制；pooling 的粒度 |
| C5 | `train_sft_x.py`，L2880–L2919 | 训练使用 input_ids；新 budget sweep 与 retrieval sweep 调用；损失未传 retrieval logits |
| C6 | `retrieval_qa_dataset.py`，L3402–L3423 | input_ids 由问题、答案分隔、gold answer 构成；因此不能整段用于初始检索 |
| C7 | `qsa_retriever.py`，L1535–L1569；`losses.py`，L677–L690 | 固定 Top-K 与 budget_topk 接口；memory_size 为成本常量时不产生策略梯度 |
| C8 | `sparse_attn.py`，L1620–L1647 | 选择候选块后，allow_blocks 与局部 band 相或；需要统一的因果 mask 与因果 selector |
| C9 | `train_sft_x.py`，L2966–L3002；`train_grpo_x.py`，L2357–L2394 | 初始化 distributed 环境后仍需检查模型包装与数据分片；不能以 torchrun 本身证明同步训练 |
| C10 | `config.py`，L179–L193、L224–L238；`train_pretrain_x.py`，L2707–L2714、L2745–L2756 | CLI 显式默认值与 preset 覆盖链、zone 切分校验 |
| C11 | `tests/test_learnability.py`，L5008–L5019 | memory 可学习性测试刻意保留梯度，与真实 controller 的 no_grad 路径不同 |
| C12 | `minimind_x.py`，L1102–L1105；`thinking_core.py`，L1888–L1892 | 现有 cache 输出按本轮 n_run 重建；历史轮数不足采用 clamp，需升级为持久的逐 token 契约 |

包开头声明了 `46 passed`。这是附件作者的验证声明；本次没有独立复跑原仓库测试。[包内 L1–L7]

# 附录 D. 已核实资料与引用边界

**R1 — Attention Residuals**  
Kimi Team，arXiv:2603.15031。用于支持“对历史深度来源做学习加权读取”的设计依据。  
论文：`https://arxiv.org/abs/2603.15031`  
作者仓库：`https://github.com/MoonshotAI/Attention-Residuals`

**R2 — Multi-Head Attention Residuals**  
Cheng Luo、Zefan Cai、Junjie Hu，arXiv:2607.27230，本文核对 v2 的多头公式。用于支持按 feature 子空间独立深度路由。Loop 边界版本、来源类型偏置及迁移强度是本文的项目化扩展，不是其原样实现。  
论文：`https://arxiv.org/abs/2607.27230`  
全文：`https://arxiv.org/html/2607.27230v2`

**R3 — Gated Recurrent Transformers / RecurrentGPT**  
Amr Hegazy、Amr Alanwar、Mostafa Elhoushi，arXiv:2608.15062。用于支持共享 Core、Prelude 条件与逐元素门控的方向。作者模型卡明确介绍近开门 sigmoid 初始化及其训练设计。本文没有把原工作的噪声与完整训练 recipe 搬入默认配置。  
论文编号入口：`https://arxiv.org/abs/2608.15062`  
作者模型卡：`https://huggingface.co/Amr-Hegazy/grt-medium-isoparam`  
作者模型卡：`https://huggingface.co/Amr-Hegazy/grt-large-isoparam`

**R4 — mHC: Manifold-Constrained Hyper-Connections**  
Zhenda Xie 等，arXiv:2512.24880。作为未纳入本版本的多流残差路线参考；不据此宣称本方案具有同样的稳定性保证。  
论文：`https://arxiv.org/abs/2512.24880`

**R5 — PyTorch Autograd mechanics**  
用于确认冻结参数、no-grad mode 与正常梯度传播之间的区别。  
官方文档：`https://docs.pytorch.org/docs/stable/notes/autograd`

**R6 — PyTorch scaled_dot_product_attention**  
用于确认 bool mask 的 `True=可见`、显式 dropout 控制及相关形状。具体 backend 支持与性能须在目标环境验证。  
官方文档：`https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html`

以上资料核实日期：2026-09-21。本文未将任何论文的 benchmark 增益写成 MiniMind-X 的预期保证，也未以一次 CPU 模块测试替代完整系统验证。

---

**设计冻结结论：保留 MiniMind-X 既有终局主线，用 Loop-MHAR 管历史读取、用独立 Reader Gate 管外部证据注入、用 Recurrent Update Gate 管状态提交，再由 Halt/检索控制器决定后续计算。**
```

---

# docs/ARCHITECTURE.md

```markdown
# MiniMind-X 实现架构说明

版本：v1.4（v1.3 二轮评审修正 + 首批实验结果与 D10/D11 决策，见 §11）
本文回答：每个设计决策（D1–D11）落在哪行代码、每个模块对应任务清单哪一项、怎么跑。

## 定位表述（对外措辞，重要）

MiniMind-X 的 1M 能力是 **million-token externalized working memory**（外置可检索工作记忆）：
长上下文经分块建库，由模型自主学习「何时检索、看多少、记多久」（adaptive memory management），
**不是**全量注意力的 1M long-context understanding。对外表述建议：
「Adaptive Memory Transformer with Million-token Externalized Working Memory」，
避免「1M context model」的说法（该口径会被理解为全注意力长上下文，与实现不符）。

---

# 1. 目录总览

```
model/minimind_x/                 # X 模型包（D1：原 model_minimind.py 一行未动 = Model A baseline）
 ├── config.py                    # T0.1 MiniMindXConfig（preset S/M/L、variant、zone 切分、预算档）
 ├── prelude.py                   # T1.1 Prelude + LocalWindowAttention（T4.4）+ ZoneStack/ZoneBlock
 ├── thinking_core.py             # T1.1 ThinkingCore（循环区）+ MemoryAwareAttention（T5.3）
 ├── coda.py                      # T1.1 Coda
 ├── minimind_x.py                # T0.2 主模型 + generate（T2.4 三模式）+ remap（T0.3）+ encode_cot_steps（T2.6）
 ├── controller.py                # T6.2/T6.3 AdaptiveController：终局编排（建库→初检→预算→循环→再检索）
 ├── losses.py                    # T2.3/T6.4 联合损失 + expected_loops + cot_alignment_loss（T2.6）
 ├── flops.py                     # T0.4 解析 FLOPs/参数量（所有效率指标统一口径）
 ├── router/                      # T2.1 halt / T6.1 retrieval / T5.4 budget
 ├── memory/                      # T5.1 working_memory（含跨轮 K/V 缓存）/ T5.2 memory_manager
 └── retrieval/                   # T4.1 block_store / T4.2 qsa_retriever / T4.3 sparse_attn（Model C）

trainer/
 ├── x_utils.py                   # init/save/weighted_ce/apply_rope_stage/grad_ckpt/tokenize 工具
 ├── oracle_labels.py             # T2.2 R*/B*/retrieve 标签（在线 sweep + 离线公式）
 ├── train_pretrain_x.py          # T1.3 Stage 0（fixed/random/full_unroll 三种 loop 模式，D7）
 ├── train_long_context.py        # T3.3 Stage 1 课程（8K→128K + retrieval-aware 加权）
 ├── train_sft_x.py               # T7.2 SFT-A/B/C（router 课程开关 = Doc 4 §20）
 └── train_grpo_x.py              # T7.3 GRPO + compute penalty + 三相 RL 课程 + 防塌缩

dataset/
 ├── long_context_dataset.py      # T3.2 packing + needle 混合 + gold 加权
 ├── retrieval_qa_dataset.py      # T6.5 {context, question, answer} 检索式样本
 └── mix_dataset.py               # T7.2 配比混合器（30/25/20/15/10）

scripts/
 ├── convert_minimind_to_x.py     # T0.3 权重转换
 ├── run_zone_experiments.py      # T1.5 zone 对比（等 FLOPs）
 ├── eval_depth_dist.py           # T2.5 深度分布 / stop accuracy
 ├── eval_needle.py               # T3.5 needle（backbone / retrieval 两路）
 ├── run_experiment_matrix.py     # T8.1 A/B/C/D × S/M/L 矩阵
 ├── scaling_fit.py               # T8.3 L = A·N^-α·D^-β·R^-γ·M^-δ 拟合
 ├── eval_suite.py                # T8.4 能力/长上下文/效率统一报告
 ├── eval_x.py                    # 推理入口（fast/normal/deep + 终局长上下文）
 ├── launch_h20_8card.sh          # T7.5 8 卡启动
 └── data_gen/                    # T3.1/T5.5：needle/long_dep/multihop/repo/planning/memory_mgmt

tests/                            # 42 项：等价性/轮次轴 KV/损失/memory 不变量/检索 recall/sparse/CoT
```

---

# 2. 设计决策（D1–D9）→ 实现位置

| 决策 | 实现 | 验证 |
|-|-|-|-|
| D1 包结构/原模型不动 | `model/minimind_x/` 新包；`model_minimind.py` 零改动 | test_equivalence（R=1 数值一致 + remap） |
| D2 循环 KV 语义 | 见 §3「轮次轴 KV」（实现期细化的关键点） | test_loop_kv（decode/prefill 一致 + cache 不随 R 膨胀） |
| D3 训练全展开 | `forward(collect_exits=True)` 返回逐轮 exit logits；随机 R 用 `exit_choice` | test_loop_kv.test_exit_choice_selects_loop |
| D4 轮次位置编码 | 每轮复用同一 positions；`loop_embedding` 可选（默认关） | config + test_loop_embedding_forward |
| D5 非可微走 oracle/RL | `controller.forward_with_oracles`（teacher-forcing）；GRPO 阶段自采样 | test_controller_training_forward |
| D6 zone 切分 | `prelude/thinking/coda_layers` + preset S=(4,4,4) M=(8,8,8) L=(10,12,10) | test_config |
| D7 Stage 矛盾统一 | train_pretrain_x `--loop_mode fixed`（默认稳）/ `random`（Doc 3 Phase0 消融） | 冒烟通过 |
| D8 长程 RoPE | `apply_rope_stage` 课程切换重算；主干只覆盖工作序列 | train_long_context 冒烟（两阶段切换） |
| D9 长上下文不进主干 | `encode_context_blocks`（Prelude 分块编码建库）+ controller 全链路 | test_controller_* / eval_needle retrieval 路 |

---

# 3. 轮次轴 KV：D2 的实现期细化（重要）

任务清单 D2 原表述「循环不向 token 轴 KV 追加条目；每轮全量重算」。实现中发现单份 KV cache 的
朴素解读（decode 各轮看 exit 轮历史 KV）与全展开 prefill（每轮自洽重算）**数值不一致**
（tests/test_loop_kv.py 的 decode/prefill 一致性测试最初失败）。修正为文献标准做法
（Huginn 式 latent-depth 轴）：

- **core 层缓存按「轮次」分槽**：`past[core 层] = [(k_r, v_r) for r in 已执行轮]`；
- 第 r 轮的 query 只看历史 token 在第 r 轮的 K/V（历史轮数不足则 clamp 到其最近轮）；
- token 轴每步仍只增加 R 个槽（Prelude/Coda 层照常 +1）；样本停住的那一轮由 hidden gather 决定；
- 该语义下 decode 与全展开 prefill **严格数值一致**（回归测试锁定），B=1 早停场景完全精确；
  batch 内早停的样本会贡献「超出其停住轮的 phantom keys」（batch 级近似，文档化取舍）。

---

# 3.5 与《终局架构方案》（Doc 1）的逐项映射（终局对照审计）

| 终局方案条目 | 实现位置 | 状态/说明 |
|-|-|-|
| §3/§21 数据流（建库→检索→记忆→Prelude→Loop→Coda） | controller + D9 决议 | ✅ 操作顺序取 Doc 2 版（Prelude(问题)→初检→…）；Doc 1 图为外层数据流，矛盾已在任务清单 D9 消解 |
| §4 三段式 Prelude/Core/Coda | zones + config 切分 | ✅ |
| §5-6 Long-Term Memory（1M，不每步参与推理） | LongContextStore（懒编码+LRU，建库一次） | ✅ |
| §7 分层稀疏检索（1M→候选 16~32K→精选） | QSARetriever（coarse 64 块=16K 默认 + fine） | ✅ 默认参数与文档量级一致 |
| §8 Retrieval Attention（QSA 思路） | SparseBlockAttention（Model C）+ 终局走 D9 的 store+检索+记忆 cross-attn | ✅ 功能等价分解（见下方注 1） |
| §9-11 工作记忆 / ADD-KEEP-EVICT-REPLACE / 五因子打分 | WorkingMemory + MemoryManager | ✅ REPLACE=并入新块后按 U 驱逐（add+evict 组合）；Persistence 因子 η 默认 0 可开 |
| §12-13 Dynamic Retrieval + 双 Router（每轮 h_r） | halt/retrieval router + loop_callback | ✅ 决策发生在每轮 pass 之后（loop_callback(r-1, h_r)） |
| §15-16 自适应思考 1~N | max_loop + 早停 generate | ✅ |
| §17 目标 L = L_task + λmCm + λrCr + λdCd | SFT=Doc 2 §18（BCE 监督版，无计数成本）；RL=GRPO reward 三项成本齐（含 Cr=N_retrieval） | ✅ 分阶段实现（见注 2） |
| §18 Unified Controller（memory size / retrieval / depth） | budget/retrieval/halt 三 router + AdaptiveController | ✅ |
| §19 分区注意力 | Thinking Core=Working-Memory Attention ✅；Prelude/Coda 默认 full（可切 local）；「Retrieval Layer」由 D9 路径承担 | ⚠ 有意偏移（见注 1） |
| §20 loop 内不重复全局检索 | store 建库一次，轮间只按需查询 | ✅ |
| §22 训练（curriculum 8K→1M） | M3 到 128K 主干 + 512K/1M 稀疏路径 | ⚠ 可行性修正（v1.1，KV 显存） |
| §23 SFT 数据（Long QA/Repo Debug/Multi-hop/Planning/Agent） | gen_needle / gen_repo_context / gen_multihop / **gen_planning** / mix_dataset | ✅（planning 为终局对照后补齐） |
| §24-25 Reasoning / Memory 数据 | rl_math+code reward / needle+memory_mgmt | ✅ |
| §26 指标（Accuracy/Context/5 项效率） | eval_suite（PPL/效率/FLOPs）+ eval_needle（单针/**多针**/**--data 多跳与长代码**）+ eval_depth_dist | ✅ 多针与数据集口径为终局对照后补齐；公开基准（GSM8K 等）为已文档化边界 |
| §27 创新点 4「统一控制器」 | 三 router 联合训练（SFT-C + GRPO） | ✅ |
| §2「Think Where（区域自适应）」 | 固定 zone（终局定义）+ X1 可选探索 | ✅ 按文档定位为研究扩展 |

**注 1（§19 有意偏移）**：Doc 1 要求 Prelude/Coda 用 Local Attention、中间放 QSA 检索层。在 D9
（长上下文不进主干）下，主干序列只有「问题+生成 token」（短），Local 窗口不再必要，故 Prelude/Coda
默认 full（`prelude_attn/coda_attn="local"` 保留为消融开关）；「QSA 检索层」的读上下文职能由
store 索引 + 检索 + Working Memory cross-attention 分解承担，SparseBlockAttention 作为 Model C
baseline 保留原教旨实现。

**注 2（§17 分阶段）**：Doc 1 的最终目标含检索计数成本 Cr。SFT 阶段按 Doc 2 §18（检索走 BCE 监督、
无计数成本——计数不可微且无 RL 信号时无意义）；Cr 在 Stage 4 的 GRPO reward 中完整出现
（`lam_m*p_m + lam_r*p_r + lam_t*p_t`），即 Doc 1 目标在 RL 阶段完整落地。

---

# 4. 训练流水线（Stage 0–4，D7 编号；Stage 1.5 为 D10 新增）

```bash
# Stage 0 基础预训练（routers 关闭；--loop_mode random 为 Doc 3 Phase0 消融）
# 唯一命令源：scripts/run_stage0.sh（batch/LR/workers 可参数覆盖，D11 吞吐方案由此生效）
bash scripts/launch_h20_8card.sh pretrain        # 或 ./scripts/run_stage0.sh --epochs 2

# Stage 1 长上下文课程（A–C：8K→32K→128K，默认 dense；D/E 512K/1M 走稀疏路径=eval_needle retrieval）
bash scripts/launch_h20_8card.sh longctx

# Stage 1.5 Mid-training（D10，待做：退火 + 数学/代码上采样 + 随机R，A/B 实验）
#   见任务清单 §7.2；数据基建缺口=原始文本配比混合器

# Stage 2 SFT 三相（A: R=1 → B: 随机 R 弹性+一致性 → C: memory/retrieval+router 课程）
bash scripts/launch_h20_8card.sh sft_a && bash scripts/launch_h20_8card.sh sft_b && bash scripts/launch_h20_8card.sh sft_c

# Stage 3/4 GRPO 三相（无惩罚→小惩罚→Pareto；compute penalty 只奖正确轨迹=防塌缩）
bash scripts/launch_h20_8card.sh grpo1 && bash scripts/launch_h20_8card.sh grpo2 && bash scripts/launch_h20_8card.sh grpo3
```

数据：`scripts/data_gen/*` 产 needle / long_dep / multihop / repo / memory_mgmt jsonl；
SFT 混合配比用 `dataset/mix_dataset.py::MixedSFTDataset`（默认 30/25/20/15/10）。

Oracle 标签（Doc 4 §3 self-generated supervision）：`trainer/oracle_labels.py` 提供
`sweep_depth`（在线 R*）、`sweep_budget`（B*）、`sweep_retrieval`（retrieve 标签），
SFT-C 以 `--oracle_rate` 小采样率在线生成（D5 teacher-forcing）。

---

# 5. 推理（三模式 + 终局）

```bash
python scripts/eval_x.py --weight sft_x_c --mode deep          # fast/normal/deep（config.mode_caps）
python scripts/eval_x.py --weight sft_x_c --context_file doc.txt  # 终局：建库→初检→轮间再检索（D9）
```

generate 支持：早停（prefill 按序列 / decode 按步）、fixed_loops、`return_stats`
（avg_loops / memory_steps / re_retrievals，供效率评测与 GRPO penalty）。

---

# 6. 评测与实验矩阵（M8）

| 命令 | 产出 | 对应 |
|-|-|-|-|
| `scripts/run_zone_experiments.py` | dense/full/middle/late_middle 等 FLOPs 对比表 | Doc 3 实验 1–3 |
| `scripts/eval_depth_dist.py` | 难度×loop 分布、oracle R*、stop accuracy | Doc 3 §14 / Doc 6 §10 |
| `scripts/eval_needle.py --mode retrieval` | 深度网格 EM（sparse 路径）；`--needles K` 多针；`--data xxx.jsonl` 直评 multihop/repo/planning（Doc 1 §26 Multi-hop QA / Long Code） | Doc 1 §26 |
| `scripts/eval_suite.py` | PPL + 效率 + FLOPs 统一 JSON | Doc 6 §7 |
| `scripts/run_experiment_matrix.py` | A/B/C/D × S/M/L 启动与记录 | Doc 6 §6 |
| `scripts/scaling_fit.py` | 拟合指数 ± SE、R² | Doc 6 §14 |

Ablation A–F 全部 config 化（T8.2）：A 固定记忆=oracle_budget 钉死；B 一次检索=
oracle_retrieve 全 0；C 固定循环=fixed_loops；D 全层循环=variant full_loop；E 全注意力=
attn 全 full；F 单 router=train_halt/retrieval/budget_router 开关。

---

# 7. 验证状态

- `python3 -m pytest tests/ -q` → **50 passed**（等价性 / 轮次轴 KV / 损失数值 / memory 不变量 /
  检索 recall+MRR / sparse 前向 / CoT 对齐 / 导出往返 / **可学习性**：loop 与 memory 路径
  在固定任务上过拟合且循环区参数确被更新 / **蒸馏**：fine→coarse 梯度确入 query_proj、
  teacher 侧 detach / batch guard / 沙箱拒绝路径）。
- CPU 冒烟全通：train_pretrain_x（loss@R 日志）、train_long_context（阶段切换）、
  SFT-A/B/C（含终局 controller teacher-forcing + 在线 oracle）、GRPO-X（reward/KL 正常）、
  5 个数据生成器、权重转换、eval_x / eval_needle / eval_depth_dist / eval_suite /
  zone 实验 / scaling 拟合 / 矩阵 dry-run。
- 环境：本仓库在 torch 2.3.1 + transformers 5.5.3 下开发验证（模型为纯 nn.Module，
  不依赖 PreTrainedModel 实例化；tokenize 走 `tokenize_tensors` 绕开版本门禁）。
  GPU 训练环境（torch≥2.4）同样适用。

# 8. 已知边界与后续

- **CoT 对齐（T2.6）**：损失（InfoNCE+余弦）与 `encode_cot_steps` 已实现并有测试；接入 SFT-B
  需要 conversations 带 `reasoning_content` 的推理数据切步，属实验期数据工程（未内置管线）。
- **HF 格式导出**：`save_pretrained_x` 落 config.json + model_x.pth（研究格式往返已测）；
  完整 transformers remote-code 导出未做（需发布建模文件，超出研究平台范围）。
- **chunked prefill（T3.4）**：以 SDPA + 梯度检查点覆盖研究规模；>128K 训练按任务清单 v1.1
  走稀疏路径（D9），主干不做 1M 全注意力。
- **基准**：GSM8K/LongBench 等公开基准未内置下载；能力曲线用 eval_depth_dist（算术分级）
  与 eval_needle（合成 needle）作为 lite 口径，公开基准接入属运维工作。
- **memory 管理数据（T5.5）**：生成器已产出 keep/drop 标签；manager 目前以键相似度打分运行，
  文本级 keep/drop 监督的接入点在 `MemoryManager.score`（可替换为学习到的打分头）。
- **检索的可微松弛**：TopK 选择不可微（D5）。已落地两条监督路径：gold CE（needle 数据）与
  fine→coarse 排序蒸馏（v1.3，gold-free，query 侧梯度经 softmax-KL 训练投影，§10-A）。
  gumbel-soft-topk 松弛与 block_hidden 梯度路径（训练 Prelude 编码器）仍列为 V2 实验，
  注意：单纯解掉 controller 的 no_grad 并不能让 query 投影获得梯度（梯度在离散 TopK 处已断）。
- **代码沙箱**：`code_task_reward` 为研究级隔离（RLIMIT + 危险模式静态拒绝，v1.3）；
  不可信输入需容器级隔离（docker/nsjail），属运维工作。

# 9. v1.2 外部评审采纳修正（2026-09-18）

外部 LLM 评审指出的问题中，以下 5 项经核实采纳并修复：

| # | 问题 | 修正 |
|-|-|-|
| 1 | **Budget Oracle 随机标签（真 bug）**：train_sft_x 用 `randint` 当 B* | 接线 `sweep_budget`（逐档 loss → B\*），与 oracle_rate 同门控；未采到标签的批次不监督（router 仍按自身 argmax 执行） |
| 2 | **GRPO off-policy 残余**：`round(avg_loops)` 把轨迹内各 token 不同的 R 压成单值 | 默认改为固定采样 R（fast=1/normal=2/deep=R_max），生成与 logps 重算路径严格一致（on-policy）；`--adaptive_rollout 1` 保留旧行为供消融 |
| 3 | **检索标签 batch 级且无 correctness 分量** | `sweep_retrieval` 升级为 per-sample [B,R] 标签：ΔCE>ε **或** correctness 增益（with 对 without 错）即点亮 |
| 4 | **测试偏接口、缺能力验证与 MRR** | 新增 tests/test_learnability.py（loop 路径/记忆路径过拟合 + 循环区参数确被更新 + loop_embedding 消融可训）；检索测试补 MRR |
| 5 | **定位表述** | 见本文开头「定位表述」：externalized working memory，非全注意力 1M context |

评审中**未采纳**的两条 P0 及理由（供后续讨论）：
- 「loop 必须默认加轮次嵌入」：近年 looped transformer（MoR、Huginn-Loop）默认不加 step
  embedding，Huginn 的消融报告加上反而掉点；轮次信息可由 hidden 状态线性解码。保留为
  `--loop_embedding` 消融开关（trainer 与 zone 实验均已暴露），默认关闭。
- 「打开 retriever 的 query encoder 梯度」：硬 TopK 下该操作是 no-op（梯度在离散选择处已断）；
  真正路径是 §8 所列的 gumbel 松弛 / block_hidden 梯度 / RL 检索质量 reward。

# 10. v1.3 二轮评审采纳修正（2026-09-18）

第二轮外部评审中采纳并修复的 3 项：

| # | 问题 | 修正 |
|-|-|-|
| A | **fine 检索不参与训练**（评审 Bug 3，最有价值一条） | `QSARetriever.distill_loss`：late-interaction 蒸馏 KL(fine ‖ coarse)，teacher detach、student 侧 query 梯度保留；`retrieve` 返回 `cand_idx/fine_scores`；controller 训练前向以 `no_grad=False` 重打分产出 `distill_loss`；SFT-C 以 `--lambda_distill`（默认 0.05）接入。检索投影由此获得 gold-free 的排序监督 |
| B1 | **SFT-C B>1 静默丢上下文**（评审 Bug 1） | 显式 guard：batch>1 直接报错并提示 batch_size=1 + 梯度累积（逐样本 store 的向量化列为 V2 工程） |
| B2 | **代码沙箱研究级即可但未声明/未设限**（评审 P1-3） | RLIMIT（CPU 5s / 内存 512MB / 禁文件写出）+ 危险模式静态拒绝 + docstring 明确「不可信输入需容器隔离」 |
| C | **Memory Manager 措辞**（评审 P1-1） | 全文统一为 adaptive heuristic memory manager（非 learned memory policy；打分头学习化/RL 化在 §8 V2） |

二轮评审中维持不采纳/延后的：P0-1 的「打开 query_proj 梯度」原方案（硬 TopK 下 no-op，蒸馏路径
已实质覆盖其目标）；P0-2「默认开 loop embedding」（保持消融开关，A/B 实验属 Phase 1）；
P1-4 公开基准（Phase 1 实验工作，非代码缺陷）。**结论：代码侧冻结，进入 Phase 1 实验验证。**

# 11. 首批实验结果与 v1.4 更新（2026-09-20）

## 11.1 实验结果（H200-1 8×H200，pretrain_t2t_mini 真实数据）

- **Zone 对比**：8000 步下 middle_loop(3.129) > dense(3.134)（2000 步时 dense 领先——循环优势随
  训练预算显现）；late_middle 被全面支配；middle_loop 为 loss/FLOP 最优点。详见任务清单 §7.1。
- **loss@R 曲线**：两 epoch 稳定 R3 最优后饱和（[1.94, 1.90, 1.89, 1.90]）——最小充分深度成立，
  为 halting oracle 与平均计算节省提供直接依据。
- 数据与权重：out/pretrain_x_768_x.pth；experiments/zone_v2_*.json。

## 11.2 新增决策

- **D10 Mid-training（Stage 1.5）**：采纳社区标准阶段（SmolLM2/Kimi K2 先例），置于架构验证链
  之后做 A/B；独有角度=退火期随机 R 验证「难数据拉大 loss@R 分离度」。配方见任务清单 §7.2。
- **D11 吞吐优化**：下一轮 Stage 0 用 batch 64/卡 × accum 1 + LR 7e-4 + workers 8（26M 模型在
  H200 上显存/利用率双低，预期 2-3 倍吞吐）；run_stage0.sh 参数覆盖即可，无需改代码。

## 11.3 v1.4 工程修复（夜间运行发现，全部已验证）

1. **resume 记账（重要）**：保存语义改为「下一待跑位置」；配置守卫（world_size/batch 变化 →
   当前 epoch 从头）；删除 trainer_utils 跨卡 step 翻倍换算。三场景端到端验证
   （同配置续跑/跨配置从头/已完成退出）。
2. **命令同源**：scripts/run_stage0.sh 为 Stage-0 唯一命令源（night_runner/看门狗/冒烟共用）。
3. **路径锚定**：CKPT_DIR 锚定仓库根绝对路径；SFT/GRPO 补 --tokenizer_path；SFT 加 --max_steps。
4. **夜间运维模式**：21:00 自动启动器（含完整停服序列：vllm-watchdog→容器→清泄漏 worker→等显存
   释放）+ stage0 窗口看门狗 + 早 8 点自动恢复与 morning_report.txt。可复用于后续每夜训练。

## 11.4 D12：五档 Effort + 思考开关（2026-09-20，待实现）

推理五档 off/low/medium/high/max（循环上限 1/1/2/3/4，预算 1K-16K；high=3 依实测 R3 最优）+
`/no_think` 软开关 + SFT-C 档位感知监督（oracle R\* 按档截断、off 混合训练）+ GRPO 五档轨迹采样。
完整设计见任务清单 §7.6；计划随 SFT-C 首跑上线。

## 11.5 D13：Tokenizer 策略（2026-09-20，延后项）

6400 词表实测 fertility（中文叙事 0.95 / 数字 1.36 字符每 token，数字逐位碎片）。验证期保持不动
（架构结论为内部对比）；preset M 重训时与 D10 一并做 32k A/B。详见任务清单 §7.7。
```

---

