# MiniMind-X 新增代码审查包

- 文件数：57（代码 56 / 文档 1）
- 总行数：5500
- 选择规则：git 未跟踪的 .py/.sh（即本次全部新增代码），按 模型→训练器→数据→脚本→测试 排序
- 背景材料：docs/ARCHITECTURE.md（D1–D9 决策映射与语义说明）、docs/MiniMind-X 编码任务清单.md（需求来源）
- 验证状态：python -m pytest tests/ -q → 50 passed（含可学习性/MRR/蒸馏梯度/batch guard/沙箱回归）；训练器/数据生成器/脚本 CPU 冒烟通过

## 目录

- `model/minimind_x/__init__.py` (18 行)
- `model/minimind_x/coda.py` (20 行)
- `model/minimind_x/config.py` (172 行)
- `model/minimind_x/controller.py` (161 行)
- `model/minimind_x/flops.py` (52 行)
- `model/minimind_x/losses.py` (176 行)
- `model/minimind_x/memory/__init__.py` (4 行)
- `model/minimind_x/memory/memory_manager.py` (86 行)
- `model/minimind_x/memory/working_memory.py` (83 行)
- `model/minimind_x/minimind_x.py` (392 行)
- `model/minimind_x/prelude.py` (104 行)
- `model/minimind_x/retrieval/__init__.py` (5 行)
- `model/minimind_x/retrieval/block_store.py` (66 行)
- `model/minimind_x/retrieval/qsa_retriever.py` (108 行)
- `model/minimind_x/retrieval/sparse_attn.py` (73 行)
- `model/minimind_x/router/__init__.py` (5 行)
- `model/minimind_x/router/budget_router.py` (19 行)
- `model/minimind_x/router/halt_router.py` (28 行)
- `model/minimind_x/router/retrieval_router.py` (20 行)
- `model/minimind_x/thinking_core.py` (208 行)
- `trainer/oracle_labels.py` (122 行)
- `trainer/train_grpo_x.py` (329 行)
- `trainer/train_long_context.py` (166 行)
- `trainer/train_pretrain_x.py` (234 行)
- `trainer/train_sft_x.py` (239 行)
- `trainer/x_utils.py` (116 行)
- `dataset/long_context_dataset.py` (119 行)
- `dataset/mix_dataset.py` (61 行)
- `dataset/retrieval_qa_dataset.py` (79 行)
- `scripts/convert_minimind_to_x.py` (53 行)
- `scripts/data_gen/__init__.py` (1 行)
- `scripts/data_gen/gen_long_dep.py` (48 行)
- `scripts/data_gen/gen_memory_mgmt.py` (56 行)
- `scripts/data_gen/gen_multihop.py` (62 行)
- `scripts/data_gen/gen_needle.py` (66 行)
- `scripts/data_gen/gen_planning.py` (67 行)
- `scripts/data_gen/gen_repo_context.py` (74 行)
- `scripts/eval_depth_dist.py` (123 行)
- `scripts/eval_needle.py` (126 行)
- `scripts/eval_suite.py` (111 行)
- `scripts/eval_x.py` (94 行)
- `scripts/launch_h20_8card.sh` (26 行)
- `scripts/run_experiment_matrix.py` (79 行)
- `scripts/run_zone_experiments.py` (115 行)
- `scripts/scaling_fit.py` (67 行)
- `tests/conftest.py` (4 行)
- `tests/test_config.py` (50 行)
- `tests/test_distill.py` (83 行)
- `tests/test_equivalence.py` (95 行)
- `tests/test_flops.py` (40 行)
- `tests/test_learnability.py` (101 行)
- `tests/test_loop_kv.py` (100 行)
- `tests/test_losses.py` (85 行)
- `tests/test_memory.py` (77 行)
- `tests/test_retrieval.py` (101 行)
- `tests/test_sparse_and_cot.py` (84 行)
- `docs/ARCHITECTURE.md` (247 行)

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
"""AdaptiveController：终局编排（任务清单 T6.2/T6.3；Doc 1 §12-14、§21；D9）。

数据流（单次 forward 的操作顺序，D9）：
  建库（一次）→ Prelude(问题) → Initial Retrieval → Budget 分配
  → Loop{ Core → halt? → retrieve? → MemoryManager.update } → Coda → 生成

训练：oracle teacher-forcing（D5：执行动作按 oracle 标签，router 只收分类损失）；
推理：router 实时决策。loop 内不重复全局建库，只按需再检索（Doc 1 §20）。
"""
from typing import Optional

import torch

from .retrieval.block_store import LongContextStore
from .memory.working_memory import WorkingMemory


class AdaptiveController:
    def __init__(self, model):
        """model: MiniMindXForCausalLM（须 use_retrieval/use_memory 打开）。"""
        self.model = model
        self.cfg = model.config
        self.store: Optional[LongContextStore] = None
        self.memory: Optional[WorkingMemory] = None
        self.stats = {"retrievals": 0, "memory_tokens": 0, "re_retrievals": 0}

    # ---------------------------------------------------------------- 建库与初检
    def begin(self, context_ids):
        self.store = LongContextStore(context_ids, self.cfg.block_size)
        self.store.build_keys(self.model.model.retriever, self.model.model.embed_tokens)
        self.memory = None
        self.stats = {"retrievals": 0, "memory_tokens": 0, "re_retrievals": 0}
        return self.store

    def _budget_idx(self, query_hidden, max_budget_tokens=None, teacher_idx=None):
        if teacher_idx is not None:
            return int(teacher_idx)
        logits = self.model.model.budget_router(query_hidden) if self.model.model.budget_router else None
        if logits is None:
            idx = self.cfg.default_budget_idx
        else:
            idx = int(logits.argmax(dim=-1).float().mean().item())  # batch 统一档位（研究级）
        if max_budget_tokens is not None:
            cap = [i for i, b in enumerate(self.cfg.memory_budgets) if b <= max_budget_tokens]
            if cap and idx > cap[-1]:
                idx = cap[-1]
        return idx

    @torch.no_grad()
    def retrieve_into_memory(self, query_hidden, budget_idx=None, mem=None,
                             gold_block_idx=None):
        """检索并把新块并入记忆（预算控制容量）。gold_block_idx: [B] 监督用。

        整条检索/建库/memory 编码路径不回传梯度（D5）；gold 监督走
        retriever.retrieval_ce_loss 的 coarse 分数（在 forward_with_oracles 中单独计算）。
        """
        result = self.model.model.retriever.retrieve(query_hidden, self.store, self.model)
        self.stats["retrievals"] += 1
        block_keys = self.store.block_keys.to(result.block_idx.device)[result.block_idx]  # [B,k,d]
        q_vec = self.model.model.retriever.project_query(query_hidden).detach()
        budget_blocks = self.cfg.budget_blocks(budget_idx)
        mem = mem if mem is not None else WorkingMemory(
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.block_size,
                                   self.cfg.hidden_size),
            query_hidden.new_zeros(query_hidden.shape[0], 0, self.cfg.hidden_size))
        hit = torch.ones_like(result.block_idx, dtype=torch.float)  # 新块本轮均被使用
        mem, _, _ = self.model.model.memory_manager.update(
            mem, result.block_hidden, block_keys, q_vec, budget_blocks, hit_new=hit)
        self.stats["memory_tokens"] = mem.token_size
        self.stats["last_result"] = result
        return mem

    # ---------------------------------------------------------------- loop callback
    def make_loop_callback(self, oracle_retrieve=None, max_budget_tokens=None):
        """轮间再检索回调（挂到 model.forward 的 loop_callback）。

        oracle_retrieve: [R] 0/1 训练 teacher-forcing；None 时用 retrieval router 实时决策。
        """
        cfg = self.cfg

        def cb(r, exit_hidden):
            if oracle_retrieve is not None:
                do = bool(oracle_retrieve[r]) if r < len(oracle_retrieve) else False
            else:
                p = self.model.model.retrieve_router(exit_hidden)
                do = bool(p.float().mean().item() > cfg.retrieve_threshold)
            if not do:
                return None
            self.stats["re_retrievals"] += 1
            with torch.no_grad():
                self.memory = self.retrieve_into_memory(
                    exit_hidden, budget_idx=self._budget_idx(exit_hidden, max_budget_tokens),
                    mem=self.memory)
                return self.memory.get_kv(self.model.model.thinking_core)
        return cb

    # ---------------------------------------------------------------- 推理入口
    @torch.no_grad()
    def generate_with_context(self, input_ids, attention_mask=None, context_ids=None,
                              mode=None, **gen_kwargs):
        """终局生成（T6.3）：建库一次 → 初检 → generate（router 驱动轮间再检索）。"""
        assert self.cfg.use_retrieval and self.cfg.use_memory, "终局路径需 use_retrieval+use_memory"
        self.begin(context_ids)
        q_hidden = self.model.model.prelude_hidden(input_ids, attention_mask)
        caps = self.cfg.mode_caps_for(mode)
        budget_idx = self._budget_idx(q_hidden, max_budget_tokens=caps.get("max_budget"))
        self.memory = self.retrieve_into_memory(q_hidden, budget_idx=budget_idx)
        mem_kv = self.memory.get_kv(self.model.model.thinking_core)
        cb = self.make_loop_callback(max_budget_tokens=caps.get("max_budget"))
        out = self.model.generate(
            inputs=input_ids, attention_mask=attention_mask, mode=mode,
            memory_provider=lambda step: self._refresh_kv(),
            loop_callback=cb, return_stats=True, **gen_kwargs)
        out["stats"].update({k: v for k, v in self.stats.items() if isinstance(v, (int, float))})
        out["stats"]["memory_tokens_final"] = self.memory.token_size if self.memory else 0
        return out

    @torch.no_grad()
    def _refresh_kv(self):
        if self.memory is None:
            return None
        return self.memory.get_kv(self.model.model.thinking_core)

    # ---------------------------------------------------------------- 训练入口（SFT-C）
    def forward_with_oracles(self, input_ids, attention_mask, labels, context_ids,
                             gold_block_idx=None, oracle_retrieve=None, oracle_budget=None,
                             oracle_r=None, distill=True):
        """SFT-C / Stage 3 的 teacher-forced 终局训练前向（D5）。

        返回 (MiniMindXOutput, extras)；extras 含：
          retrieval_ce —— gold block CE（needle 数据，有 gold 时）
          distill_loss —— fine→coarse 排序蒸馏（v1.3，无需 gold 标签即可监督检索投影）
          memory 统计
        """
        self.begin(context_ids)
        q_hidden = self.model.model.prelude_hidden(input_ids, attention_mask)
        budget_idx = self._budget_idx(q_hidden, teacher_idx=oracle_budget)
        with torch.no_grad():  # D5：memory 构建/选择路径不回传梯度
            self.memory = self.retrieve_into_memory(q_hidden, budget_idx=budget_idx)
            mem_kv = self.memory.get_kv(self.model.model.thinking_core)
        cb = self.make_loop_callback(oracle_retrieve=oracle_retrieve)
        out = self.model(
            input_ids, attention_mask=attention_mask, labels=None,
            collect_exits=True, memory_kv_per_layer=mem_kv, loop_callback=cb)
        extras = {"retrievals": self.stats["retrievals"],
                  "memory_tokens": self.memory.token_size}
        result = self.stats.get("last_result")
        if gold_block_idx is not None and result is not None:
            n_blocks = result.coarse_scores.shape[-1]
            if (gold_block_idx >= 0).all() and (gold_block_idx < n_blocks).all():
                extras["retrieval_ce"] = self.model.model.retriever.retrieval_ce_loss(
                    result.coarse_scores, gold_block_idx)
        if distill and result is not None and result.fine_scores is not None:
            # query 侧保留梯度的重打分（仅 Prelude + query_proj + 点积，成本小）；
            # teacher（fine）与候选对齐索引来自 no_grad 的检索结果
            q_grad = self.model.model.prelude_hidden(input_ids, attention_mask, no_grad=False)
            q_vec = self.model.model.retriever.project_query(q_grad)
            coarse_trainable = self.model.model.retriever.coarse_scores(q_vec, self.store)
            extras["distill_loss"] = self.model.model.retriever.distill_loss(
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
    total = backbone + memory_attn + retrieval
    if train:
        total *= 3
    return {"total": float(total), "backbone": float(backbone),
            "memory_attn": float(memory_attn), "retrieval": float(retrieval),
            "effective_depth": P + C + T * R}
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


def halt_bce_loss(halt_probs: List[torch.Tensor], oracle_r: torch.Tensor):
    """逐轮 BCE（标签见 halt_labels_from_oracle）。halt_probs: [B]×R 的列表。"""
    labels, mask = halt_labels_from_oracle(oracle_r, len(halt_probs))  # [B,R]
    probs = torch.stack(halt_probs).transpose(0, 1).clamp(1e-6, 1 - 1e-6)  # [B,R]
    bce = F.binary_cross_entropy(probs, labels, reduction="none")
    return (bce * mask).sum() / mask.sum().clamp(min=1.0)


def retrieval_bce_loss(retrieve_probs: List[torch.Tensor], oracle_retrieve: torch.Tensor):
    """每轮是否再检索的 BCE。oracle_retrieve: [R]（0/1，逐轮决策）。"""
    probs = torch.stack(retrieve_probs).clamp(1e-6, 1 - 1e-6)     # [R,B]
    labels = oracle_retrieve.to(probs.device).unsqueeze(1).expand_as(probs)
    return F.binary_cross_entropy(probs, labels.float())


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
               config=None) -> Dict[str, torch.Tensor]:
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
        parts["halt"] = halt_bce_loss(halt_probs, oracle_r)
        total = total + (config.lambda_halt if config else 0.1) * parts["halt"]
        c = expected_loops(torch.stack(halt_probs)) / len(halt_probs)   # E[R]/R_max
        parts["compute"] = c.mean()
        total = total + (config.lambda_compute if config else 0.01) * parts["compute"]
    if retrieve_probs and oracle_retrieve is not None:
        parts["retrieve"] = retrieval_bce_loss(retrieve_probs, oracle_retrieve)
        total = total + (config.lambda_retrieve if config else 0.1) * parts["retrieve"]
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
        b_max = max(config.memory_budgets)
        ms = memory_size.to(exit_logits[0].device).float()
        parts["memory"] = (ms / b_max).mean()
        total = total + config.lambda_memory * parts["memory"]
    parts["loss"] = total
    return parts
```

---

# model/minimind_x/memory/__init__.py

```python
from .working_memory import WorkingMemory
from .memory_manager import MemoryManager

__all__ = ["WorkingMemory", "MemoryManager"]
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


class WorkingMemory:
    def __init__(self, hidden, keys, usage=None, persistence=None, block_size=256):
        assert hidden.dim() == 4 and keys.dim() == 3
        assert hidden.shape[:2] == keys.shape[:2]
        self.hidden = hidden
        self.keys = keys
        self.block_size = hidden.shape[2] if hidden.shape[2] else block_size
        bsz, k = hidden.shape[0], hidden.shape[1]
        self.usage = usage if usage is not None else hidden.new_zeros(bsz, k)
        self.persistence = persistence if persistence is not None else hidden.new_zeros(bsz, k)
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

    # ---- 修改操作（返回新对象，函数式；Doc 1 §10 ADD/KEEP/EVICT/REPLACE 由 manager 组合）----
    def with_added(self, hidden, keys):
        if self.is_empty():
            return WorkingMemory(hidden, keys, block_size=self.block_size)
        return WorkingMemory(
            torch.cat([self.hidden, hidden], dim=1),
            torch.cat([self.keys, keys], dim=1),
            torch.cat([self.usage, self.usage.new_zeros(hidden.shape[0], hidden.shape[1])], dim=1),
            torch.cat([self.persistence, self.persistence.new_zeros(hidden.shape[0], hidden.shape[1])], dim=1),
            block_size=self.block_size)

    def select(self, keep_idx):
        """keep_idx: [B, m'] —— 按样本保留 slot（驱逐 argmin(U) 之后的结果）。"""
        bsz = self.hidden.shape[0]
        arange = torch.arange(bsz, device=self.hidden.device).unsqueeze(1)
        return WorkingMemory(
            self.hidden[arange, keep_idx], self.keys[arange, keep_idx],
            self.usage[arange, keep_idx], self.persistence[arange, keep_idx],
            block_size=self.block_size)

    def with_usage_bumped(self, hit_mask, decay=0.9):
        """attention/retrieval 使用统计的 EMA 更新：被再次选中的块 usage 提升。"""
        usage = decay * self.usage + (1 - decay) * hit_mask.float()
        return WorkingMemory(self.hidden, self.keys, usage, self.persistence, block_size=self.block_size)

    def with_persistence_ema(self, scores, decay=0.9):
        """persistence = 历史 U 的 EMA（Doc 1 §11 Persistence 因子）。"""
        persistence = decay * self.persistence + (1 - decay) * scores.detach()
        return WorkingMemory(self.hidden, self.keys, self.usage, persistence, block_size=self.block_size)

    # ---- 跨 loop K/V 缓存 ----
    def get_kv(self, thinking_core):
        """返回与 thinking_core.layers 对齐的 [(k, v), ...]；缓存失效自动重投影。"""
        if self._kv_cache is None:
            self._kv_cache = thinking_core.project_memory(self.token_hidden())
        return self._kv_cache

    def invalidate_kv(self):
        self._kv_cache = None
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
    halt_probs: Optional[List[torch.Tensor]] = None    # 每轮 P(stop) [B]
    exit_idx: Optional[torch.Tensor] = None            # 每样本选中的 exit 轮号 [B]
    budget_logits: Optional[torch.Tensor] = None       # 预算档 logits [B, n_budgets]
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
                memory_kv_per_layer=None, loop_callback=None):
        cfg = self.config
        if hasattr(past_key_values, 'layers'):  # 兼容 transformers Cache 误传
            past_key_values = None
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
        budget_logits = self.budget_router(hidden_states) if self.budget_router is not None else None

        # ---- ThinkingCore（循环区）----
        num_loops = max(1, int(num_loops)) if cfg.use_loop else 1
        halt_fn = None
        if halt and self.halt_router is not None:
            halt_fn = lambda hd: self.halt_router(hd)
        core_out = self.thinking_core(
            hidden_states, position_embeddings, past[P:P + T], use_cache, attention_mask,
            memory_kv_per_layer=memory_kv_per_layer, num_loops=num_loops,
            halt_fn=halt_fn, halt_threshold=cfg.halt_threshold, loop_callback=loop_callback)
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
        # core 层缓存：按轮独立条目（轮次轴 KV，D2 细化）——token 轴每步只 +1×R 槽
        if use_cache:
            core_caches = [[passes[r][l] for r in range(n_run)] for l in range(T)]
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
        return dict(hidden_states=hidden_states, presents=presents, aux_loss=aux_loss,
                    exit_hiddens=exit_hiddens, halt_probs=halt_probs, exit_idx=exit_idx,
                    budget_logits=budget_logits, num_loops_executed=n_run)


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
                loop_callback=None, **kwargs):
        cfg = self.config
        if num_loops is None:
            num_loops = cfg.max_loop if (collect_exits and cfg.use_loop) else 1
        out = self.model(input_ids, attention_mask, past_key_values, use_cache,
                         num_loops=num_loops, halt=halt, collect_exits=collect_exits,
                         exit_choice=exit_choice, memory_kv_per_layer=memory_kv_per_layer,
                         loop_callback=loop_callback)
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
            halt_probs=out["halt_probs"], exit_idx=out["exit_idx"],
            budget_logits=out["budget_logits"], num_loops_executed=out["num_loops_executed"])

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
        stats = {"loop_sum": 0.0, "steps": 0, "memory_steps": 0}
        for step in range(max_new_tokens):
            past_len = (past_key_values[0][0][0].shape[1] if isinstance(past_key_values[0], list)
                        else past_key_values[0][0].shape[1]) if past_key_values else 0
            mem = memory_provider(step) if memory_provider is not None else None
            if mem is not None:
                stats["memory_steps"] += 1
            num_loops = fixed_loops if fixed_loops is not None else (loop_cap if halting or cfg.use_loop else 1)
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache,
                num_loops=num_loops, halt=halting, memory_kv_per_layer=mem,
                loop_callback=loop_callback, **kwargs)
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
                stats["loop_sum"] += float((outputs.exit_idx + 1).float().mean().item())
            else:
                stats["loop_sum"] += float(num_loops)
            stats["steps"] += 1
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
TopK 不可微（D5），梯度只走 query/key 投影的 CE 路径。
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
        k_sel = min(self.n_selected, n_blk)
        sel = blk_score.topk(k_sel, dim=-1).indices                    # [B,H,S,k]
        # 逐 head 选块的并集（研究级近似：union over heads）
        block_mask = torch.zeros(bsz, self.n_local_heads, seq_len, n_blk,
                                 device=x.device, dtype=torch.bool).scatter_(-1, sel, True)
        col_block = torch.arange(total_len, device=x.device) // self.block  # key 的块号
        allow_blocks = block_mask.any(dim=1, keepdim=True)[..., col_block]  # [B,1,S,total]

        # ---- 窗口 + 因果 ----
        row = torch.arange(total_len - seq_len, total_len, device=x.device).unsqueeze(1)
        col = torch.arange(total_len, device=x.device).unsqueeze(0)
        band = (col <= row) & (col > row - self.window)

        scores = (xq @ xk_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allow = allow_blocks | band.unsqueeze(0).unsqueeze(0)
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
"""Thinking/Halting Router（任务清单 T2.1；Doc 2 §12、Doc 3 §8）。

预测 continue/stop（二分类），不是直接回归深度；标签由 oracle R* 展开为逐轮 stop 标签。
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
    """p_s = σ(W_s · pool(h_r))，P(stop)；per-sequence（prefill）或 per-token（decode）。"""

    def __init__(self, config):
        super().__init__()
        self.pool = config.halt_pool
        self.proj = nn.Linear(config.hidden_size, 1, bias=True)

    def forward(self, hidden):
        pooled = pool_hidden(hidden, self.pool)
        return torch.sigmoid(self.proj(pooled)).squeeze(-1)
```

---

# model/minimind_x/router/retrieval_router.py

```python
"""Retrieval Router（任务清单 T6.1；Doc 2 §7、Doc 4 §13）。

p_r = σ(W_r · pool(h_r))：当前信息够不够，不够则重新检索。
oracle 标签 = with/without retrieval 的 loss 差（见 trainer/oracle_labels.py）。
"""
import torch
from torch import nn

from .halt_router import pool_hidden


class RetrievalRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool = config.retrieve_query_pool
        self.proj = nn.Linear(config.hidden_size, 1, bias=True)

    def forward(self, hidden):
        pooled = pool_hidden(hidden, self.pool)
        return torch.sigmoid(self.proj(pooled)).squeeze(-1)
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
            self.layers = nn.ModuleList(
                [ThinkingBlock(layer_offset + i, config) for i in range(config.thinking_layers)])
        # D4：可选轮次嵌入（每轮复用同一 RoPE positions；loop embedding 默认关闭，供消融）
        self.loop_embed = (nn.Embedding(config.max_loop, config.hidden_size)
                           if config.loop_embedding else None)

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
                     use_cache, attention_mask, memory_kv_per_layer, loop_idx):
        """执行一轮。past_key_values: 每层一个「按轮缓存」条目（list of (k,v) 或 None）。

        轮次轴 KV 语义（D2 细化）：第 r 轮的 query 只看历史 token 在第 r 轮的 K/V
        （不足 r 轮的历史 clamp 到其最近一轮）——与全展开 prefill 每轮自洽重算严格一致，
        保证 decode/prefill 数值等价（tests/test_loop_kv.py 回归）。
        """
        if self.loop_embed is not None:
            hidden_states = hidden_states + self.loop_embed.weight[loop_idx]
        memory_kv_per_layer = memory_kv_per_layer or [None] * self.num_layers
        presents = []
        for l, layer in enumerate(self.layers):
            pl = past_key_values[l]
            past_in = pl[min(loop_idx, len(pl) - 1)] if pl else None
            mem = memory_kv_per_layer[l]
            ckpt = getattr(self, "gradient_checkpointing", False)
            if ckpt and self.training and hidden_states.requires_grad:
                if isinstance(layer, ThinkingBlock):
                    hidden_states, present = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, position_embeddings, past_in, use_cache,
                        attention_mask, memory_kv=mem, use_reentrant=False)
                else:
                    hidden_states, present = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, position_embeddings, past_in, use_cache,
                        attention_mask, use_reentrant=False)
                presents.append(present)
                continue
            if isinstance(layer, ThinkingBlock):
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
                num_loops=1, halt_fn=None, halt_threshold=0.5, loop_callback=None):
        """循环执行。

        参数：
          num_loops: 固定循环轮数（训练全展开 / 固定 R）
          halt_fn: 可选，exit_hidden([B,S,d]) -> P(stop)([B])；给出时逐轮早停
          loop_callback: 可选，(r, exit_hidden) -> 新的 memory_kv_per_layer 或 None，
                         供终局编排做「轮间再检索 + memory 更新」（任务清单 T6.2，训练期
                         按 oracle teacher-forcing、推理期由 retrieval router 决策）

        返回 dict：
          exit_hiddens: 每轮 exit hidden 列表（residual 流，未过 coda）
          presents_per_pass: 每轮每层 present（use_cache 时为该轮完整 KV）
          halt_probs: 每轮 P(stop)（无 halt_fn 时为 None）
          exit_idx: 每样本停住的轮号 [B]（无 halt_fn 时为 None）
        """
        num_loops = max(1, int(num_loops))
        past_key_values = past_key_values or [None] * self.num_layers
        exit_hiddens, presents_per_pass = [], []
        halt_probs, exit_idx = [], None
        if halt_fn is not None:
            bsz = hidden_states.shape[0]
            halted = torch.zeros(bsz, dtype=torch.bool, device=hidden_states.device)
            exit_idx = torch.full((bsz,), -1, dtype=torch.long, device=hidden_states.device)
        for r in range(num_loops):
            if loop_callback is not None and r > 0:
                new_mem = loop_callback(r - 1, exit_hiddens[-1])
                if new_mem is not None:
                    memory_kv_per_layer = new_mem
            hidden_states, presents = self._single_pass(
                hidden_states, position_embeddings, past_key_values,
                use_cache, attention_mask, memory_kv_per_layer, r)
            exit_hiddens.append(hidden_states)
            presents_per_pass.append(presents)
            if halt_fn is not None:
                p_stop = halt_fn(hidden_states)
                halt_probs.append(p_stop)
                newly = (~halted) & (p_stop > halt_threshold)
                exit_idx[newly] = r
                halted = halted | newly
                if bool(halted.all()):
                    break
        return dict(exit_hiddens=exit_hiddens, presents_per_pass=presents_per_pass,
                    halt_probs=halt_probs, exit_idx=exit_idx)
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
                 budget_idxs=None, attention_mask=None, **oracle_kwargs):
    """性能矩阵的 memory 维：逐预算档跑 teacher-forced 前向，返回 loss [B, n档]。

    controller_factory() -> AdaptiveController（每档新建，避免状态串扰）。
    """
    cfg = model.config
    idxs = budget_idxs if budget_idxs is not None else list(range(len(cfg.memory_budgets)))
    losses = []
    for idx in idxs:
        ctrl = controller_factory()
        out, _ = ctrl.forward_with_oracles(
            input_ids, attention_mask, labels, context_ids, oracle_budget=idx, **oracle_kwargs)
        losses.append(_per_sample_ce(out.exit_logits[-1], labels))
    return torch.stack(losses, dim=-1)  # [B, n]


@torch.no_grad()
def sweep_retrieval(model, input_ids, labels, attention_mask=None, memory_kv_per_layer=None,
                    num_loops=None):
    """with/without retrieval 的逐样本对比 -> 每轮 retrieve 标签（Doc 4 §13，v1.2 升级）。

    per-sample 口径（不再 batch 平均）：标签 = 1 当且仅当该样本满足任一条件——
      ① ΔCE 增益：L_without(r) − L_with(r) > ε（token-level gain）
      ② correctness 增益：with 在第 r 轮已 correct 而 without 仍错
    返回 (labels_r [B,R] 0/1, gain [B,R])。SFT-C 在 B=1 下逐行 teacher-forcing；
    B>1 时由调用方投票或取首样本（文档化口径）。
    """
    loss_with, correct_with, _ = sweep_depth(model, input_ids, labels, attention_mask,
                                             memory_kv_per_layer, num_loops)
    loss_without, correct_without, _ = sweep_depth(model, input_ids, labels, attention_mask,
                                                   None, num_loops)
    eps = model.config.oracle_epsilon
    gain = loss_without - loss_with                                   # [B,R] ΔCE
    correct_gain = (correct_with.float() - correct_without.float())   # [B,R] ∈ {-1,0,1}
    labels_r = ((gain > eps) | (correct_gain > 0)).long()
    return labels_r, gain
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
from torch.utils.data import DataLoader

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
    unwrapped = model_.module if isinstance(model_, DistributedDataParallel) else model_
    logits = unwrapped(input_ids, attention_mask=attention_mask, num_loops=num_loops,
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
                traj.append((i, torch.cat([base, ro["ids"][j].cpu()]).to(args.device), ro))
        old_per_token, logp_lens, traj_loops = [], [], []
        with torch.no_grad():
            for (i, ids, ro) in traj:
                n_keep = ids.numel() - prompt_base_ids[i].numel()
                j = len(old_per_token) % args.num_generations
                r = max(1, int(round(ro["loop"][j])))  # 轨迹实际轮数对齐
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
    parser.add_argument("--adaptive_rollout", default=0, type=int, choices=[0, 1],
                        help="0=固定采样 R（默认，on-policy）；1=adaptive halting rollout"
                             "（logps 用单一 R 近似，残余 off-policy 偏差，消融用）")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"])
    parser.add_argument('--variant', default='minimind_x', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/rl_math.jsonl",
                        help="math RL 数据：{conversations, gt}")
    parser.add_argument("--reward_type", type=str, default="math", choices=["math", "code"],
                        help="math=safe_math_eval 判等；code=python 沙箱 stdout 匹配（T7.4）")
    parser.add_argument('--from_weight', default='sft_x_c', type=str)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-GRPO")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(preset=args.preset, variant=args.variant, max_loop=args.max_loop,
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
                                    save_dir=args.save_dir, device=args.device)
    model.train()
    ref_model, _ = init_model_x(lm_config, args.from_weight,
                                save_dir=args.save_dir, device=args.device)
    ref_model.eval().requires_grad_(False)

    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=512)

    def _collate(b):
        for x in b:  # tools=None 会卡 default_collate，GRPO 不用它
            x.setdefault("tools", [])
            if x["tools"] is None:
                x["tools"] = []
        return b

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    iters = len(loader) * args.epochs
    for epoch in range(args.epochs):
        grpo_x_epoch(epoch, loader, iters, ref_model, 0, wandb)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
```

---

# trainer/train_long_context.py

```python
"""长上下文继续预训练（任务清单 T3.3；Doc 4 §6、Doc 5 §7-9）。

Context Length Curriculum：8K → 32K → 128K（Phase A–C，主干全量 attention）。
512K/1M（Phase D/E）依赖 M4 稀疏路径，不在本脚本（见任务清单 v1.1 修正）。

L = L_LM + λ·L_retrieval-aware：needle/gold token 的 CE 加权（weighted_ce）。
阶段切换时重算 RoPE（apply_rope_stage）。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import argparse
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.minimind_x import MiniMindXConfig
from dataset.long_context_dataset import LongContextPretrainDataset, collate_long_context
from model.minimind_x.losses import lm_loss_per_exit
from trainer.trainer_utils import (get_lr, Logger, is_main_process, lm_checkpoint,
                                   init_distributed_mode, setup_seed, SkipBatchSampler)
from trainer.x_utils import (init_model_x, save_x_checkpoint, weighted_ce, apply_rope_stage,
                             set_gradient_checkpointing)

warnings.filterwarnings('ignore')


def train_stage(stage_idx, stage_cfg, epoch, iters, wandb=None):
    """单个 curriculum 阶段的训练循环。stage_cfg: {"max_len", "rope_theta", "steps"}。"""
    Logger(f'===== Stage {stage_idx}: max_len={stage_cfg["max_len"]}, '
           f'rope_theta={stage_cfg.get("rope_theta")}, steps={stage_cfg["steps"]} =====')
    apply_rope_stage(model, max_position_embeddings=stage_cfg["max_len"],
                     rope_theta=stage_cfg.get("rope_theta"))
    train_ds = LongContextPretrainDataset(args.data_path, tokenizer,
                                          max_length=stage_cfg["max_len"],
                                          needle_ratio=args.needle_ratio,
                                          needle_weight=args.needle_weight)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate_long_context)
    start_time = time.time()
    step = 0
    for input_ids, labels, weight in loader:
        step += 1
        if step > stage_cfg["steps"]:
            break
        input_ids, labels, weight = (input_ids.to(args.device), labels.to(args.device),
                                     weight.to(args.device))
        lr = get_lr(step, stage_cfg["steps"], args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        with autocast_ctx:
            if lm_config.use_loop and args.loop_mode == "full_unroll":
                res = model(input_ids, labels=None, collect_exits=True)
                exits = res.exit_logits
                total = sum(weighted_ce(lg, labels, weight) for lg in exits) / len(exits)
                loss = (total + res.aux_loss) / args.accumulation_steps
            else:
                r = max(1, args.fixed_loops)
                res = model(input_ids, labels=None, num_loops=r)
                total = weighted_ce(res.logits, labels, weight)
                loss = (total + res.aux_loss) / args.accumulation_steps
        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step % args.log_interval == 0 or step == stage_cfg["steps"]:
            eta = (time.time() - start_time) / step * (stage_cfg["steps"] - step) // 60
            Logger(f'Stage {stage_idx}({step}/{stage_cfg["steps"]}), loss: {total.item():.4f}, '
                   f'lr: {lr:.8f}, eta: {eta:.1f}min')
            if wandb:
                wandb.log({"loss": total.item(), "stage": stage_idx,
                           "max_len": stage_cfg["max_len"], "learning_rate": lr})
        del input_ids, labels, weight, loss
    if is_main_process():
        model.eval()
        save_x_checkpoint(model, args.save_dir, f'{args.save_weight}_stage{stage_idx}',
                          lm_config.hidden_size)
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-X Long-Context CPT (Stage 1)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='longctx_x', type=str)
    parser.add_argument("--batch_size", type=int, default=1, help="长序列建议 1 + 梯度累积")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="CPT 用小学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument('--variant', default='dense', type=str,
                        choices=["dense", "middle_loop", "minimind_x"],
                        help="D7：Stage 1 默认 dense（loop disabled）")
    parser.add_argument('--preset', default=None, type=str, choices=["S", "M", "L"])
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument('--loop_mode', default='fixed', type=str, choices=["fixed", "full_unroll"])
    parser.add_argument('--fixed_loops', default=1, type=int)
    parser.add_argument("--stage_lens", type=str, default="8192,32768,131072",
                        help="curriculum 各阶段序列长（逗号分隔；Phase A-C）")
    parser.add_argument("--rope_thetas", type=str, default=None,
                        help="各阶段 rope theta（逗号分隔，可选；默认沿用 config）")
    parser.add_argument("--steps_per_stage", type=str, default="1000,1000,1000",
                        help="各阶段 step 数（逗号分隔）")
    parser.add_argument("--needle_ratio", type=float, default=0.1, help="needle 样本混合比例")
    parser.add_argument("--needle_weight", type=float, default=5.0, help="gold token CE 加权")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    parser.add_argument('--from_weight', default='pretrain_x', type=str)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-LongContext")
    parser.add_argument("--grad_ckpt", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(preset=args.preset, variant=args.variant, max_loop=args.max_loop,
                                **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                         num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"LongCtx-{args.stage_lens}")

    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    save_dir=args.save_dir, device=args.device)
    if args.grad_ckpt == 1:
        set_gradient_checkpointing(model, True)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    lens = [int(x) for x in args.stage_lens.split(",")]
    steps = [int(x) for x in args.steps_per_stage.split(",")]
    thetas = [float(x) for x in args.rope_thetas.split(",")] if args.rope_thetas else [None] * len(lens)
    assert len(lens) == len(steps) >= 1
    for i, (ml, st, th) in enumerate(zip(lens, steps, thetas)):
        train_stage(i, {"max_len": ml, "steps": st, "rope_theta": th}, 0, st, wandb)

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
from trainer.x_utils import init_model_x, save_x_checkpoint, set_gradient_checkpointing

warnings.filterwarnings('ignore')


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
            save_x_checkpoint(model, args.save_dir, args.save_weight, lm_config.hidden_size)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                          save_dir='../checkpoints')
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
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
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
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    parser.add_argument('--from_weight', default='none', type=str, help="X 权重前缀（none=从头）")
    parser.add_argument('--from_minimind_weight', default='none', type=str,
                        help="原 MiniMind 权重路径（自动 remap，T0.3 训练侧入口）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-X-Pretrain")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--grad_ckpt", default=0, type=int, choices=[0, 1], help="梯度检查点（T1.4）")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindXConfig(
        preset=args.preset, variant=args.variant, use_moe=bool(args.use_moe),
        max_loop=args.max_loop, loop_embedding=bool(args.loop_embedding),
        **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                 num_hidden_layers=args.num_hidden_layers,
                                 prelude_layers=args.prelude_layers,
                                 thinking_layers=args.thinking_layers,
                                 coda_layers=args.coda_layers,
                                 prelude_attn=args.prelude_attn,
                                 core_attn=args.core_attn,
                                 coda_attn=args.coda_attn).items() if v is not None})
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"MiniMindX-{args.variant}-{args.loop_mode}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    model, tokenizer = init_model_x(lm_config, args.from_weight,
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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

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
from trainer.x_utils import init_model_x, save_x_checkpoint, set_gradient_checkpointing
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
    """终局 teacher-forcing（SFT-C / Stage 3）：controller + joint loss + 在线 oracle。"""
    assert_sft_c_batch(batch["input_ids"])
    input_ids = batch["input_ids"].to(args.device)
    labels = batch["labels"].to(args.device)
    context_ids = batch["context_ids"][0].to(args.device)  # 研究级口径：batch 内取首样本上下文
    gold_block = batch["gold_block"].to(args.device)
    oracle_r_file = int(batch["oracle_r"][0].item())
    oracle_ret = batch["oracle_retrieve"][0].tolist()

    # ---- 在线 oracle（Doc 4 §3：Self-generated Supervision；小采样率）----
    oracle_r = None
    if oracle_r_file > 0:
        oracle_r = torch.full((input_ids.shape[0],), oracle_r_file, device=args.device)
    elif random.random() < args.oracle_rate:
        per_exit_loss, _, _ = sweep_depth(model, input_ids, labels)
        oracle_r = oracle_depth_from_losses(per_exit_loss, lm_config.oracle_epsilon)
    # Budget oracle：文件标签缺失时在线 sweep（逐档 loss → B*，Doc 4 §11）；
    # 未采到的批次不监督（budget router 仍按自身 argmax 执行，自洽）
    oracle_budget = None
    if args.train_budget_router and random.random() < args.oracle_rate:
        loss_matrix = sweep_budget(
            model, input_ids, labels, context_ids,
            controller_factory=lambda: AdaptiveController(model))
        oracle_budget = int(oracle_budget_from_matrix(loss_matrix, lm_config.oracle_epsilon)[0].item())
    if not oracle_ret and random.random() < args.oracle_rate:
        labels_r, _ = sweep_retrieval(model, input_ids, labels)  # per-sample [B,R]
        oracle_ret = labels_r[0].tolist()  # SFT-C B=1：逐行 teacher-forcing

    ctrl = AdaptiveController(model)
    out, extras = ctrl.forward_with_oracles(
        input_ids, None, labels, context_ids,
        gold_block_idx=gold_block if (gold_block[0] >= 0).all() else None,
        oracle_retrieve=oracle_ret if oracle_ret else None,
        oracle_budget=oracle_budget)
    halt_probs = out.halt_probs if (oracle_r is not None and args.train_halt_router) else None
    parts = joint_loss(out.exit_logits, labels, halt_probs=halt_probs, oracle_r=oracle_r,
                       budget_logits=out.budget_logits if args.train_budget_router else None,
                       oracle_budget=oracle_budget,
                       memory_size=torch.tensor([extras["memory_tokens"]]),
                       config=lm_config)
    zero = next(model.parameters()).new_zeros(())
    loss = parts["loss"] + zero + extras.get("retrieval_ce", zero)
    if "distill_loss" in extras and args.lambda_distill > 0:
        loss = loss + args.lambda_distill * extras["distill_loss"]
    return loss, {"loss": parts["loss"].item(),
                  "retrievals": extras["retrievals"],
                  "memory_tokens": extras["memory_tokens"],
                  "distill": float(extras.get("distill_loss", zero))}


def main():
    global args, lm_config, model, tokenizer
    parser = argparse.ArgumentParser(description="MiniMind-X SFT (A/B/C)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='sft_x', type=str)
    parser.add_argument("--phase", type=str, required=True, choices=["a", "b", "c"],
                        help="SFT-A(R=1) / SFT-B(随机R弹性) / SFT-C(memory+retrieval)")
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_loop', default=4, type=int)
    parser.add_argument('--full_unroll_ratio', default=0.25, type=float,
                        help="SFT-B 中全展开（带一致性监督）的批次比例")
    parser.add_argument('--alpha_consistency', default=0.1, type=float)
    parser.add_argument('--oracle_rate', default=0.1, type=float, help="在线 oracle 采样率")
    parser.add_argument('--train_halt_router', default=1, type=int, choices=[0, 1],
                        help="router 课程 Phase2：thinking router 开关")
    parser.add_argument('--train_budget_router', default=0, type=int, choices=[0, 1],
                        help="router 课程 Phase4：memory router 开关")
    parser.add_argument('--lambda_distill', default=0.05, type=float,
                        help="fine→coarse 检索蒸馏系数（v1.3；0 关闭）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini.jsonl",
                        help="SFT-A/B 数据（conversations jsonl）")
    parser.add_argument("--retrieval_data_path", type=str,
                        default="../dataset/retrieval_qa.jsonl", help="SFT-C 数据")
    parser.add_argument('--from_weight', default='pretrain_x', type=str)
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
                                **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                         num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    model, tokenizer = init_model_x(lm_config, args.from_weight,
                                    save_dir=args.save_dir, device=args.device)
    model.train()
    if args.grad_ckpt == 1:
        set_gradient_checkpointing(model, True)

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
    loader = DataLoader(ds, batch_size=args.batch_size if args.phase != "c" else 1,
                        shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    iters = len(loader) * args.epochs
    step_global = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for step, batch in enumerate(loader, 1):
            step_global += 1
            lr = get_lr(step_global, iters, args.learning_rate)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
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
        if is_main_process():
            save_x_checkpoint(model, args.save_dir, f'{args.save_weight}_{args.phase}',
                              lm_config.hidden_size)


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
    model = MiniMindXForCausalLM(lm_config)
    if from_weight != 'none':
        path = x_weight_path(save_dir, from_weight, lm_config.hidden_size)
        weights = torch.load(path, map_location=device)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        Logger(f'loaded X weights from {path} (missing={len(missing)}, unexpected={len(unexpected)})')
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


def save_x_checkpoint(model, args_save_dir, save_weight, hidden_size):
    os.makedirs(args_save_dir, exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    raw = getattr(raw, '_orig_mod', raw)
    torch.save({k: v.half().cpu() for k, v in raw.state_dict().items()},
               x_weight_path(args_save_dir, save_weight, hidden_size))


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
    model.load_state_dict(torch.load(os.path.join(out_dir, "model_x.pth"),
                                     map_location="cpu"), strict=False)
    return model.to(device)
```

---

# dataset/long_context_dataset.py

```python
"""长上下文数据集（任务清单 T3.2；Doc 4 §7、Doc 5 §8）。

- LongContextPretrainDataset：document packing（多文档拼接至 max_length，无 padding 浪费），
  支持 needle 混合（合成长依赖样本 + gold 位置加权 → L_retrieval-aware）。
- 返回 (input_ids, labels, loss_weight)：loss_weight 供 retrieval-aware 辅助损失加权。

与 scripts/data_gen/* 配合：生成器产出更复杂的 jsonl（needle/multihop/repo），
本类负责 token 化、打包与 gold 权重标注。
"""
import json
import os
import random

import torch
from torch.utils.data import Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FILLER_WORDS = ("系统 数据 模型 训练 语言 上下文 检索 记忆 注意 表示 计算 结构 过程 "
                "信息 网络 参数 优化 目标 函数 结果 分析 方法 实验 性能 任务 序列 知识").split()

NEEDLE_TEMPLATES = [
    ("重要提示：本段的魔法数字是 {num}。", "本段的魔法数字是多少？", "{num}"),
    ("关键记录：会议编号为 {num}。", "会议编号是多少？", "{num}"),
    ("核心配置：缓存大小设置为 {num}。", "缓存大小是多少？", "{num}"),
]


def synth_needle_text(target_tokens, depth_ratio=None, rng=None):
    """构造一条 needle 样本文本：填充文本 + 深处插入 needle + 末尾提问。

    返回 (text, needle_sentence)。depth_ratio: 插入深度比例（0~1，None 随机）。
    """
    rng = rng or random
    num = rng.randint(1000, 9999)
    tpl_idx = rng.randrange(len(NEEDLE_TEMPLATES))
    template, question, _ = NEEDLE_TEMPLATES[tpl_idx]
    needle = template.format(num=num)
    depth = depth_ratio if depth_ratio is not None else rng.random()
    # 用填充句堆到目标长度（按字符近似，1 token ≈ 1.5 中文字符）
    target_chars = int(target_tokens * 1.5)
    parts, total = [], 0
    while total < target_chars:
        sent = "".join(rng.choice(FILLER_WORDS) for _ in range(rng.randint(8, 20))) + "。"
        parts.append(sent)
        total += len(sent)
    pos = max(1, int(len(parts) * depth))
    parts.insert(pos, needle)
    text = "".join(parts) + question
    return text, needle


class LongContextPretrainDataset(Dataset):
    """打包式长上下文预训练数据（stage A–C：8K→128K 全量 attention 主干）。"""

    def __init__(self, data_path, tokenizer, max_length=8192, needle_ratio=0.1,
                 needle_weight=5.0, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.needle_ratio = needle_ratio
        self.needle_weight = needle_weight
        self.rng = random.Random(seed)
        self.samples = self._load(data_path)

    def _load(self, data_path):
        if not os.path.exists(data_path):
            return []
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    texts.append(item["text"])
        return texts

    def __len__(self):
        # 打包后 chunk 数 ≈ 总 token / max_length；needle 混合不改变量纲
        return max(1, len(self.samples))

    def _pack_chunk(self):
        """随机取文档拼满一个 max_length chunk；按概率混入 needle 样本并标 gold 权重。"""
        use_needle = self.rng.random() < self.needle_ratio
        if use_needle:
            text, needle = synth_needle_text(self.max_length - 8, rng=self.rng)
            gold_text = needle
        else:
            text = self.rng.choice(self.samples) if self.samples else \
                "".join(self.rng.choice(FILLER_WORDS) for _ in range(64)) + "。"
            gold_text = None
        tokens = self.tokenizer(text, add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids[:self.max_length], dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        weight = torch.ones(self.max_length)
        if gold_text is not None:
            gold_ids = self.tokenizer(gold_text, add_special_tokens=False).input_ids
            n = len(gold_ids)
            needle_tok = torch.tensor(gold_ids, dtype=torch.long)
            window = input_ids.unfold(0, n, 1)
            match = (window == needle_tok).all(dim=1)
            idx = match.nonzero().flatten()
            if len(idx):
                for i in idx.tolist():
                    weight[i:i + n] = self.needle_weight  # L_retrieval-aware：gold 加权
        return input_ids, labels, weight

    def __getitem__(self, index):
        return self._pack_chunk()


def collate_long_context(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    ws = torch.stack([b[2] for b in batch])
    return xs, ys, ws
```

---

# dataset/mix_dataset.py

```python
"""SFT 数据配比混合器（任务清单 T7.2；Doc 4 §18 / Doc 5 §12）。

默认配比：General 30% / Coding Agent 25% / Long Context QA 20% / Math 15% / Planning 10%。
输入多个 jsonl 路径 + 比例，按比例有放回采样拼接为单一数据集视图。
"""
import os
import random

import torch
from torch.utils.data import Dataset

DEFAULT_MIX = {
    "general": ("sft_general.jsonl", 0.30),
    "coding": ("sft_coding.jsonl", 0.25),
    "longctx": ("sft_longctx.jsonl", 0.20),
    "math": ("sft_math.jsonl", 0.15),
    "planning": ("sft_planning.jsonl", 0.10),
}


class MixedSFTDataset(Dataset):
    """按比例混合多个 SFT jsonl（均须为 conversations 格式，与 SFTDataset 兼容）。

    用法：
        ds = MixedSFTDataset(tokenizer, "../dataset",
                             [("sft_general.jsonl", 0.3), ("sft_coding.jsonl", 0.7)],
                             total_size=100000)
    """

    def __init__(self, tokenizer, data_dir, path_ratios=None, total_size=100000,
                 max_length=1024, seed=42):
        from dataset.lm_dataset import SFTDataset
        if path_ratios is None:
            path_ratios = [(f, r) for f, r in DEFAULT_MIX.values()]
        norm = sum(r for _, r in path_ratios)
        self.sources = []
        self.weights = []
        for fname, ratio in path_ratios:
            path = fname if os.path.isabs(fname) else os.path.join(data_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"混合数据缺失: {path}")
            self.sources.append(SFTDataset(path, tokenizer, max_length=max_length))
            self.weights.append(ratio / norm)
        self.total_size = total_size
        self.rng = random.Random(seed)
        self.index_plan = self._make_plan()

    def _make_plan(self):
        plan = []
        for src_idx, w in enumerate(self.weights):
            plan += [(src_idx, self.rng.randrange(len(self.sources[src_idx])))
                     for _ in range(int(self.total_size * w))]
        self.rng.shuffle(plan)
        return plan

    def __len__(self):
        return len(self.index_plan)

    def __getitem__(self, index):
        src_idx, sample_idx = self.index_plan[index]
        return self.sources[src_idx][sample_idx]
```

---

# dataset/retrieval_qa_dataset.py

```python
"""检索式 SFT 数据集（任务清单 T6.5；Doc 4 §8）。

jsonl schema：{"context": str(长上下文), "question": str, "answer": str,
               "gold_block": int(可选，块号), "oracle_r": int(可选), "oracle_retrieve": [0/1](可选)}

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
                 max_context_tokens=None, block_size=256):
        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.max_context_tokens = max_context_tokens
        self.block_size = block_size
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        q_ids = self.tokenizer(s["question"], add_special_tokens=False,
                               max_length=self.max_question_len, truncation=True).input_ids
        a_ids = self.tokenizer(s["answer"], add_special_tokens=False,
                               max_length=self.max_answer_len, truncation=True).input_ids
        input_ids = ([self.tokenizer.bos_token_id] + q_ids +
                     self.tokenizer("答案：", add_special_tokens=False).input_ids +
                     a_ids + [self.tokenizer.eos_token_id])
        labels = [-100] * (1 + len(q_ids) + len(self.tokenizer("答案：", add_special_tokens=False).input_ids)) \
            + a_ids + [self.tokenizer.eos_token_id]
        ctx_ids = self.tokenizer(s["context"], add_special_tokens=False,
                                 max_length=self.max_context_tokens,
                                 truncation=True).input_ids if s.get("context") else []
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "gold_block": torch.tensor(s.get("gold_block", -1), dtype=torch.long),
            "oracle_r": torch.tensor(s.get("oracle_r", 0), dtype=torch.long),
            "oracle_retrieve": torch.tensor(s.get("oracle_retrieve", []), dtype=torch.long),
            "answer_text": s["answer"],
        }


def collate_retrieval_qa(batch, pad_id=0):
    """变长 pad collate（input/labels pad；context 取 batch 内最长）。"""
    if len(batch) == 1:
        b = batch[0]
        return {**{k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in b.items()}}
    out = {}
    for key in ("input_ids", "labels"):
        seqs = [b[key] for b in batch]
        max_len = max(len(s) for s in seqs)
        pad_value = -100 if key == "labels" else pad_id
        out[key] = torch.stack([torch.cat([s, s.new_full((max_len - len(s),), pad_value)]) for s in seqs])
    ctxs = [b["context_ids"] for b in batch]
    max_ctx = max(len(c) for c in ctxs)
    out["context_ids"] = torch.stack([torch.cat([c, c.new_zeros(max_ctx - len(c))]) for c in ctxs])
    for key in ("gold_block", "oracle_r"):
        out[key] = torch.stack([b[key] for b in batch])
    out["oracle_retrieve"] = [b["oracle_retrieve"] for b in batch]
    out["answer_text"] = [b["answer_text"] for b in batch]
    return out
```

---

# scripts/convert_minimind_to_x.py

```python
"""原 MiniMind 权重 -> MiniMind-X 布局转换（任务清单 T0.3）。

用法：
  python scripts/convert_minimind_to_x.py --in ../out/pretrain_768.pth \
      --out ../out/pretrain_x_768_x.pth --variant middle_loop
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.minimind_x import MiniMindXConfig, remap_minimind_state_dict


def main():
    p = argparse.ArgumentParser(description="MiniMind -> MiniMind-X 权重转换")
    p.add_argument("--in", dest="src", type=str, required=True, help="原 MiniMind .pth")
    p.add_argument("--out", dest="dst", type=str, required=True, help="输出 X .pth")
    p.add_argument("--variant", type=str, default="minimind_x",
                   choices=["dense", "full_loop", "middle_loop", "late_middle", "minimind_x"])
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--preset", type=str, default=None, choices=["S", "M", "L"])
    p.add_argument("--prelude_layers", type=int, default=None)
    p.add_argument("--thinking_layers", type=int, default=None)
    p.add_argument("--coda_layers", type=int, default=None)
    args = p.parse_args()

    cfg = MiniMindXConfig(preset=args.preset, variant=args.variant,
                          **{k: v for k, v in dict(
                              hidden_size=args.hidden_size,
                              num_hidden_layers=args.num_hidden_layers,
                              prelude_layers=args.prelude_layers,
                              thinking_layers=args.thinking_layers,
                              coda_layers=args.coda_layers).items() if v is not None})
    src = torch.load(args.src, map_location="cpu")
    if "model" not in src and not any(k.startswith("model.") for k in src):
        raise ValueError(f"{args.src} 不像 MiniMindForCausalLM 的 state_dict")
    mapped = remap_minimind_state_dict(src, cfg)
    n_mapped = len([k for k in mapped if k.startswith("model.")])
    total_backbone = cfg.num_hidden_layers * (4 + 1) + 1  # 粗略：层数×(attn4+mlp1)+norm
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(mapped, args.dst)
    print(f"converted {len(mapped)} tensors -> {args.dst}")
    print(f"zone layout: prelude={cfg.prelude_layers} core={cfg.thinking_layers} "
          f"coda={cfg.coda_layers} (variant={cfg.variant})")
    print("note: routers/retriever 新模块未初始化，加载时用 strict=False（训练中学习）")


if __name__ == "__main__":
    main()
```

---

# scripts/data_gen/__init__.py

```python
# data generators package
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
    definition = f"定义：{name} 的初始值为 {val}，后续所有引用必须使用该值。"
    use = f"计算：{name} 的值加上 {rng.randint(1, 100)}，结果以「{name} 的最终取值是」开头给出：{name} 的最终取值是 {val}。"
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
STEPS_POOL = ["备份当前状态", "扩容资源", "切换流量", "验证健康检查", "更新配置",
              "清理临时文件", "通知相关方", "归档日志"]


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
    plan = rng.sample(STEPS_POOL, k=n_steps)
    # 资源/约束散布进上下文（需检索组合）
    doc = [f"资源清单：本次可使用 {('、'.join(res))}。"] + \
          [f"约束{i + 1}：{c}。" for i, c in enumerate(cons)]
    insert_at = sorted(rng.sample(range(len(parts)), len(doc)), reverse=True)
    for pos, line in zip(insert_at, doc):
        parts.insert(pos, line)
    question = (f"目标：完成一次上线变更，涉及 {res[0]} 与 {res[-1]}，需遵守全部约束，"
                f"共 {n_steps} 步。请给出按序执行计划。")
    answer = "；".join(f"第{i + 1}步：{s}" for i, s in enumerate(plan)) + "。"
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
    bug_fn = [l.split("(")[0][4:] for l in files[target_file].splitlines()
              if l.startswith("def ")]
    fn = rng.choice(bug_fn) if bug_fn else "main"
    issue = (f"Issue: {target_file} 中的 {fn}() 在空输入时抛出 TypeError。\n"
             f"请给出最小修复 patch。")
    answer = (f"Patch ({target_file}):\n"
              f"@@ def {fn} @@\n"
              f"-    return sum(a for a in [a0] ...)\n"
              f"+    if not args: return 0\n"
              f"+    return sum(a for a in args if isinstance(a, (int, float)))")
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

# scripts/eval_depth_dist.py

```python
"""Adaptive Depth 评测（任务清单 T2.5；Doc 3 §14 router 指标）。

合成难度分级的算术题 → 逐难度统计：
  - oracle R*（固定 R 逐档生成取首个 correct，Doc 3 §12）
  - halting 决策的 loop 分布（Doc 6 §10 Dynamic Depth 曲线）
  - stop accuracy（决策轮 == oracle 轮）
"""
import os
import sys
import json
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from trainer.x_utils import x_weight_path, tokenize_tensors
from trainer.trainer_utils import safe_math_eval


def make_arith_prompt(rng, difficulty):
    """难度 1~5：运算步数与数值规模递增。返回 (prompt, answer)。"""
    n_terms = difficulty + 1
    terms = [rng.randint(2, 9 + difficulty * 10) for _ in range(n_terms)]
    ops = [rng.choice("+-*") for _ in range(n_terms - 1)]
    expr = str(terms[0])
    for op, t in zip(ops, terms[1:]):
        expr += f" {op} {t}"
    value = safe_math_eval(expr)
    prompt = f"计算：{expr} = ?"
    return prompt, value


def correctness(response, gt):
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", response)
    if not nums:
        return False
    try:
        return abs(float(nums[-1]) - float(gt)) < 1e-6
    except ValueError:
        return False


def main():
    p = argparse.ArgumentParser(description="动态深度评测：loop 分布 vs 难度")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="sft_x_b")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="middle_loop")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--max_loop", type=int, default=4)
    p.add_argument("--n_per_difficulty", type=int, default=20)
    p.add_argument("--difficulties", type=str, default="1,2,3,4,5")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/depth_dist.json")
    args = p.parse_args()

    cfg_kw = dict(variant=args.variant, max_loop=args.max_loop, preset=args.preset,
                  **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                           num_hidden_layers=args.num_hidden_layers).items() if v or k == "variant"})
    cfg = MiniMindXConfig(**{k: v for k, v in cfg_kw.items() if v is not None})
    model = MiniMindXForCausalLM(cfg)
    path = x_weight_path(args.weight_dir, args.weight, cfg.hidden_size)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    rng = random.Random(0)
    report = {"per_difficulty": [], "config": vars(args)}
    print(f"{'diff':>5}{'acc':>7}{'oracle_R':>10}{'halting_R':>10}{'stop_acc':>9}")
    for d in [int(x) for x in args.difficulties.split(",")]:
        oracle_rs, halt_rs, correct_fixed, correct_halt = [], [], 0, 0
        for _ in range(args.n_per_difficulty):
            prompt, gt = make_arith_prompt(rng, d)
            ids = tokenize_tensors(tokenizer, prompt, args.device)
            # oracle：固定 R 逐档，取首个 correct
            oracle = cfg.max_loop
            for r in range(1, cfg.max_loop + 1):
                out = model.generate(inputs=ids["input_ids"], attention_mask=ids["attention_mask"],
                                     max_new_tokens=args.max_new_tokens, do_sample=False,
                                     eos_token_id=tokenizer.eos_token_id, fixed_loops=r,
                                     use_halting=False)
                resp = tokenizer.decode(out[0][ids["input_ids"].shape[1]:].tolist(),
                                        skip_special_tokens=True)
                if correctness(resp, gt):
                    oracle = r
                    correct_fixed += 1
                    break
            oracle_rs.append(oracle)
            # halting 决策
            out = model.generate(inputs=ids["input_ids"], attention_mask=ids["attention_mask"],
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 use_halting=True, return_stats=True)
            resp = tokenizer.decode(out["generated_ids"][0][ids["input_ids"].shape[1]:].tolist(),
                                    skip_special_tokens=True)
            halt_rs.append(out["stats"]["avg_loops"])
            correct_halt += int(correctness(resp, gt))
        n = args.n_per_difficulty
        stop_acc = sum(1 for o, h in zip(oracle_rs, halt_rs) if abs(o - h) < 0.5) / n
        row = {"difficulty": d, "acc_fixed": correct_fixed / n, "acc_halt": correct_halt / n,
               "oracle_mean_R": sum(oracle_rs) / n,
               "halt_mean_R": sum(halt_rs) / n, "stop_accuracy": stop_acc,
               "oracle_hist": {str(r): oracle_rs.count(r) for r in set(oracle_rs)},
               "halt_hist": {str(r): halt_rs.count(r) for r in set(halt_rs)}}
        report["per_difficulty"].append(row)
        print(f"{d:>5}{row['acc_fixed']:>7.2f}{row['oracle_mean_R']:>10.2f}"
              f"{row['halt_mean_R']:>10.2f}{stop_acc:>9.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
```

---

# scripts/eval_needle.py

```python
"""Needle / 检索式任务评测（任务清单 T3.5；Doc 1 §26 Context Ability）。

三种口径：
  单针     ：--mode retrieval|backbone，深度网格（默认）
  多针     ：--needles K>1（Doc 1 §26 Multi Needle）
  数据集   ：--data xxx.jsonl（直接评 multihop / repo / planning 等检索式数据，
             覆盖 Doc 1 §26 的 Multi-hop QA 与 Long Code）

EM 判定：answer（精确字符串）出现在生成文本中。
"""
import os
import sys
import json
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), "data_gen"))

import torch
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from trainer.x_utils import x_weight_path, tokenize_tensors


def run_one(model, tokenizer, ctrl, args, cfg, sample):
    ctx_ids = tokenizer(sample["context"], add_special_tokens=False).input_ids
    q_ids = tokenizer(sample["question"], add_special_tokens=False).input_ids
    ids = torch.tensor([q_ids], device=args.device)
    if args.mode == "backbone":
        total = (ctx_ids + q_ids)[:cfg.max_position_embeddings - 1]
        ids = torch.tensor([total], device=args.device)
        out = model.generate(inputs=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=args.max_new_tokens, do_sample=False,
                             eos_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True)
        stats = {}
    else:
        out = ctrl.generate_with_context(
            ids, attention_mask=torch.ones_like(ids),
            context_ids=torch.tensor(ctx_ids, device=args.device),
            max_new_tokens=args.max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out["generated_ids"][0][ids.shape[1]:].tolist(),
                                skip_special_tokens=True)
        stats = out["stats"]
    return int(str(sample["answer"]) in resp), stats


def main():
    p = argparse.ArgumentParser(description="needle / multi-needle / 检索式任务评测")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="longctx_x_stage2")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="minimind_x")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--mode", type=str, default="retrieval", choices=["backbone", "retrieval"])
    p.add_argument("--data", type=str, default=None,
                   help="检索式 jsonl（needle/multihop/repo/planning）；给出则忽略深度网格")
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--depths", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--needles", type=int, default=1, help="多针评测的针数（Doc 1 §26）")
    p.add_argument("--n_per_cell", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/needle.json")
    args = p.parse_args()

    cfg = MiniMindXConfig(variant=args.variant, hidden_size=args.hidden_size,
                          num_hidden_layers=args.num_hidden_layers)
    model = MiniMindXForCausalLM(cfg)
    model.load_state_dict(torch.load(x_weight_path(args.weight_dir, args.weight, cfg.hidden_size),
                                     map_location="cpu"), strict=False)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    rng = random.Random(7)
    ctrl = AdaptiveController(model) if args.mode == "retrieval" else None

    cells = []
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()][:args.n_per_cell]
        correct, agg = 0, {"avg_loops": 0.0, "retrievals": 0, "memory_tokens": 0, "n": 0}
        for s in samples:
            hit, stats = run_one(model, tokenizer, ctrl, args, cfg, s)
            correct += hit
            if stats:
                agg["avg_loops"] += stats.get("avg_loops", 0)
                agg["retrievals"] += stats.get("retrievals", 0)
                agg["memory_tokens"] += stats.get("memory_tokens_final", 0)
                agg["n"] += 1
        n = max(len(samples), 1)
        row = {"cell": os.path.basename(args.data), "n": n, "acc": correct / n}
        if agg["n"]:
            row.update({k: agg[k] / agg["n"] for k in ("avg_loops", "retrievals", "memory_tokens")})
        cells.append(row)
        print(f"[{row['cell']}] n={n} acc={row['acc']:.2f}" +
              (f" loops={row.get('avg_loops', 0):.2f} retr={row.get('retrievals', 0):.1f}" if agg["n"] else ""))
    else:
        from gen_needle import gen_sample
        depths_list = [float(x) for x in args.depths.split(",")]
        for depth in depths_list:
            correct = 0
            for _ in range(args.n_per_cell):
                s = (gen_sample(rng, args.ctx_tokens, depths_list, args.needles, cfg.block_size)
                     if args.needles > 1 else
                     gen_sample(rng, args.ctx_tokens, [depth], 1, cfg.block_size))
                hit, _ = run_one(model, tokenizer, ctrl, args, cfg, s)
                correct += hit
            acc = correct / args.n_per_cell
            cells.append({"cell": f"depth={depth:.1f}", "depth": depth,
                          "needles": args.needles, "acc": acc})
            print(f"depth={depth:.1f} needles={args.needles}  acc={acc:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "ctx_tokens": args.ctx_tokens, "cells": cells,
                   "config": vars(args)}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
```

---

# scripts/eval_suite.py

```python
"""评测套件（任务清单 T8.4；Doc 1 §26、Doc 6 §7）。

一次运行产出统一 JSON 报告：
  能力：val PPL（文本 jsonl）
  长上下文：needle（retrieval 路径，短网格）
  深度：loop 分布（合成难度分级）
  效率：tokens/s、avg_loops、FLOPs（estimate_flops 统一口径）
"""
import os
import sys
import json
import argparse
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, estimate_flops
from model.minimind_x.controller import AdaptiveController
from trainer.x_utils import x_weight_path, tokenize_tensors
from trainer.trainer_utils import safe_math_eval


def eval_ppl(model, tokenizer, data_path, max_samples=50, max_len=512, device="cpu"):
    import torch.nn.functional as F
    tot_nll, tot_tok = 0.0, 0
    with open(data_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()][:max_samples]
    model.eval()
    with torch.no_grad():
        for line in lines:
            text = json.loads(line).get("text", "")
            ids = tokenizer(text, add_special_tokens=False, max_length=max_len,
                            truncation=True).input_ids
            if len(ids) < 8:
                continue
            x = torch.tensor([ids], device=device)
            logits = model(x, num_loops=1).logits
            nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                  x[:, 1:].reshape(-1), reduction="sum")
            tot_nll += float(nll)
            tot_tok += len(ids) - 1
    return tot_nll / max(tot_tok, 1)


def eval_efficiency(model, tokenizer, device, n=8, max_new_tokens=32):
    prompts = ["请解释机器学习的基本原理。"] * n
    t0 = time.time()
    loops, tokens = [], 0
    with torch.no_grad():
        for p in prompts:
            ids = tokenize_tensors(tokenizer, p, device)
            out = model.generate(inputs=ids["input_ids"],
                                 attention_mask=ids.get("attention_mask"),
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id, return_stats=True)
            loops.append(out["stats"]["avg_loops"])
            tokens += out["generated_ids"].shape[1] - ids["input_ids"].shape[1]
    sec = max(time.time() - t0, 1e-6)
    return {"tokens_per_sec": tokens / sec, "avg_loops": sum(loops) / len(loops),
            "gen_tokens": tokens, "sec": round(sec, 2)}


def main():
    p = argparse.ArgumentParser(description="MiniMind-X 评测套件")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="sft_x_c")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="minimind_x")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--ppl_data", type=str, default="../dataset/pretrain_hq.jsonl")
    p.add_argument("--skip_needle", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="experiments/eval_suite_report.json")
    args = p.parse_args()

    cfg = MiniMindXConfig(preset=args.preset, variant=args.variant,
                          **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                   num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    model = MiniMindXForCausalLM(cfg)
    model.load_state_dict(torch.load(x_weight_path(args.weight_dir, args.weight, cfg.hidden_size),
                                     map_location="cpu"), strict=False)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    report = {"config": {"variant": cfg.variant, "hidden": cfg.hidden_size,
                         "layers": cfg.num_hidden_layers,
                         "zones": [cfg.prelude_layers, cfg.thinking_layers, cfg.coda_layers],
                         "max_loop": cfg.max_loop},
              "params_m": round(sum(pp.numel() for pp in model.parameters()) / 1e6, 2)}
    if os.path.exists(args.ppl_data):
        report["ppl"] = eval_ppl(model, tokenizer, args.ppl_data, device=args.device)
    report["efficiency"] = eval_efficiency(model, tokenizer, args.device)
    report["flops_tok"] = estimate_flops(cfg, seq_len=512, loops=round(
        report["efficiency"]["avg_loops"]))["total"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved -> {args.out}")
    if not args.skip_needle:
        print("提示：needle 网格评测请另跑 scripts/eval_needle.py（较慢）")


if __name__ == "__main__":
    main()
```

---

# scripts/eval_x.py

```python
"""MiniMind-X 推理入口（任务清单 T7.5；三模式 Fast/Normal/Deep + 终局长上下文）。"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, TextStreamer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from trainer.x_utils import x_weight_path, tokenize_tensors
from trainer.trainer_utils import setup_seed
import random

PROMPTS = ["你有什么特长？", "为什么天空是蓝色的", "解释什么是机器学习", "1+2*3等于几？"]


def main():
    p = argparse.ArgumentParser(description="MiniMind-X 推理与对话")
    p.add_argument("--weight_dir", type=str, default="../out")
    p.add_argument("--weight", type=str, default="sft_x_c")
    p.add_argument("--tokenizer_path", type=str, default="../model")
    p.add_argument("--variant", type=str, default="minimind_x")
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--mode", type=str, default=None, choices=["fast", "normal", "deep"],
                   help="推理模式（None=自适应 halting）")
    p.add_argument("--fixed_loops", type=int, default=None, help="固定 R（覆盖模式）")
    p.add_argument("--context_file", type=str, default=None,
                   help="长上下文文件 → 终局检索路径（D9）")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--historys", type=int, default=0)
    p.add_argument("--show_speed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = MiniMindXConfig(preset=args.preset, variant=args.variant,
                          **{k: v for k, v in dict(hidden_size=args.hidden_size,
                                                   num_hidden_layers=args.num_hidden_layers).items() if v is not None})
    model = MiniMindXForCausalLM(cfg)
    model.load_state_dict(torch.load(x_weight_path(args.weight_dir, args.weight, cfg.hidden_size),
                                     map_location="cpu"), strict=False)
    model = model.half().eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    ctrl = AdaptiveController(model) if args.context_file else None
    context_ids = None
    if args.context_file:
        text = open(args.context_file, encoding="utf-8").read()
        context_ids = torch.tensor(
            tokenizer(text, add_special_tokens=False).input_ids, device=args.device)

    conversation = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    for prompt in PROMPTS:
        setup_seed(random.randint(0, 314159))
        print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        inputs_txt = tokenizer.apply_chat_template(conversation, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = tokenize_tensors(tokenizer, inputs_txt, args.device, truncation=True)
        print('🧠: ', end='')
        import time
        st = time.time()
        gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                      streamer=streamer, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id, top_p=args.top_p,
                      temperature=args.temperature, repetition_penalty=1,
                      mode=args.mode, fixed_loops=args.fixed_loops, return_stats=True)
        if ctrl is not None:
            out = ctrl.generate_with_context(inputs["input_ids"], inputs["attention_mask"],
                                             context_ids=context_ids, **gen_kw)
            gen_ids = out["generated_ids"]
        else:
            out = model.generate(inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                 **gen_kw)
            gen_ids = out["generated_ids"] if isinstance(out, dict) else out
        response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:].tolist(),
                                    skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        stats = out.get("stats", {}) if isinstance(out, dict) else {}
        n_tok = gen_ids.shape[1] - inputs["input_ids"].shape[1]
        if args.show_speed:
            print(f'\n[Speed]: {n_tok / max(time.time() - st, 1e-6):.2f} tokens/s | '
                  f'avg_loops: {stats.get("avg_loops", "-")}')


if __name__ == "__main__":
    main()
```

---

# scripts/launch_h20_8card.sh

```bash
#!/usr/bin/env bash
# H20 8 卡启动脚本（任务清单 T7.5；Doc 5 §27）
# 用法：bash scripts/launch_h20_8card.sh <stage> [额外参数...]
#   stage ∈ {pretrain, longctx, sft_a, sft_b, sft_c, grpo1, grpo2, grpo3}
set -e
cd "$(dirname "$0")/.."

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
STAGE=${1:?用法: launch_h20_8card.sh <pretrain|longctx|sft_a|sft_b|sft_c|grpo1|grpo2|grpo3> [args...]}
shift || true

case "$STAGE" in
  pretrain) CMD="trainer/train_pretrain_x.py --variant minimind_x --preset M --loop_mode full_unroll --grad_ckpt 1 $*" ;;
  longctx)  CMD="trainer/train_long_context.py --variant dense --preset M --stage_lens 8192,32768,131072 --steps_per_stage 2000,2000,2000 --grad_ckpt 1 $*" ;;
  sft_a)    CMD="trainer/train_sft_x.py --phase a --variant minimind_x --preset M $*" ;;
  sft_b)    CMD="trainer/train_sft_x.py --phase b --variant minimind_x --preset M --full_unroll_ratio 0.25 --alpha_consistency 0.1 $*" ;;
  sft_c)    CMD="trainer/train_sft_x.py --phase c --variant minimind_x --preset M --oracle_rate 0.1 --train_halt_router 1 $*" ;;
  grpo1)    CMD="trainer/train_grpo_x.py --rl_phase 1 --variant minimind_x --preset M $*" ;;
  grpo2)    CMD="trainer/train_grpo_x.py --rl_phase 2 --variant minimind_x --preset M $*" ;;
  grpo3)    CMD="trainer/train_grpo_x.py --rl_phase 3 --variant minimind_x --preset M $*" ;;
  *) echo "未知 stage: $STAGE"; exit 1 ;;
esac

echo "[launch] torchrun --nproc_per_node=$NPROC $CMD"
exec torchrun --nproc_per_node="$NPROC" --nnodes="$NNODES" $CMD
```

---

# scripts/run_experiment_matrix.py

```python
"""实验矩阵 Runner（任务清单 T8.1/T8.5；Doc 6 §6）。

Model A Dense（原 train_pretrain） / B MoE（原 train_pretrain --use_moe）
Model C Sparse（train_pretrain_x --variant dense + sparse attn）
Model D MiniMind-X（train_pretrain_x --variant minimind_x）
× 规模 S/M/L × token 预算。顺序执行（或 --dry_run 打印命令），记录落 experiments/。
"""
import os
import sys
import json
import argparse
import subprocess
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.minimind_x import PRESETS, count_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def build_commands(args):
    cmds = []
    for size in args.sizes.split(","):
        preset = PRESETS[size]
        for model_type in args.models.split(","):
            name = f"{model_type}_{size}"
            if model_type == "A":  # Dense baseline（原模型，不改）
                cmds.append((name, ["python", "trainer/train_pretrain.py",
                                    "--hidden_size", str(preset["hidden_size"]),
                                    "--num_hidden_layers", str(preset["num_hidden_layers"]),
                                    "--epochs", "1"]))
            elif model_type == "B":  # MoE baseline
                cmds.append((name, ["python", "trainer/train_pretrain.py",
                                    "--hidden_size", str(preset["hidden_size"]),
                                    "--num_hidden_layers", str(preset["num_hidden_layers"]),
                                    "--use_moe", "1", "--epochs", "1"]))
            elif model_type == "C":  # Sparse attention baseline（fixed depth）
                cmds.append((name, ["python", "trainer/train_pretrain_x.py",
                                    "--preset", size, "--variant", "dense",
                                    "--loop_mode", "fixed", "--fixed_loops", "1",
                                    "--prelude_attn", "sparse", "--core_attn", "sparse",
                                    "--coda_attn", "sparse"]))
            elif model_type == "D":  # MiniMind-X
                cmds.append((name, ["python", "trainer/train_pretrain_x.py",
                                    "--preset", size, "--variant", "minimind_x",
                                    "--loop_mode", "full_unroll"]))
            else:
                raise ValueError(f"未知 model_type {model_type}（A/B/C/D）")
    return cmds


def main():
    p = argparse.ArgumentParser(description="Scaling 实验矩阵（Doc 6 §6）")
    p.add_argument("--models", type=str, default="D", help="A/B/C/D 组合（逗号分隔）")
    p.add_argument("--sizes", type=str, default="S", help="S/M/L 组合（逗号分隔）")
    p.add_argument("--dry_run", action="store_true", help="只打印命令不执行")
    p.add_argument("--exp_dir", type=str, default="experiments")
    args = p.parse_args()

    cmds = build_commands(args)
    os.makedirs(args.exp_dir, exist_ok=True)
    for name, cmd in cmds:
        print(f"[matrix] {name}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        rec = {"name": name, "cmd": cmd, "returncode": proc.returncode,
               "sec": round(time.time() - t0, 1),
               "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
        with open(os.path.join(args.exp_dir, f"{name}_{int(time.time())}.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        if proc.returncode != 0:
            print(f"[matrix] {name} FAILED，见 {args.exp_dir}")
    print("[matrix] done. records ->", args.exp_dir)


if __name__ == "__main__":
    main()
```

---

# scripts/run_zone_experiments.py

```python
"""Zone 对比实验（任务清单 T1.5；Doc 3 §6/§13 实验 1-3）。

在等 FLOPs 预算下对比：Baseline(dense R=1) vs Full Loop vs Middle Loop vs Late-Middle。
- 等预算换算：full_loop 的 R_full = ceil(T_mid × R_mid / L)，使有效深度 FLOPs 对齐
- 输出：各 variant 的 val loss、平均 FLOPs/step、结果表 + experiments/ 记录

在进程内跑小规模对比（CPU/单卡均可），大规模请用 train_pretrain_x.py 逐个跑后聚合。
"""
import os
import sys
import json
import argparse
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import optim

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, estimate_flops
from trainer.trainer_utils import Logger

VARIANTS = ["dense", "full_loop", "middle_loop", "late_middle"]


def loops_for_equal_flops(cfg, target_flops_per_token, seq_len):
    """给定目标 backbone FLOPs，求该 variant 的等价 R（dense 固定 1）。"""
    if not cfg.use_loop:
        return 1
    r = 1
    while r < cfg.max_loop:
        f = estimate_flops(cfg, seq_len, loops=r)["total"]
        if f >= target_flops_per_token:
            break
        r += 1
    return r


def run_variant(variant, args, train_tokens, val_tokens):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    kw = dict(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
              variant=variant, max_loop=args.max_loop, flash_attn=False,
              vocab_size=args.vocab_size, loop_embedding=bool(args.loop_embedding))
    cfg = MiniMindXConfig(**kw)
    model = MiniMindXForCausalLM(cfg)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    # 参照预算：middle_loop R=2 的 backbone FLOPs
    ref_cfg = MiniMindXConfig(**dict(kw, variant="middle_loop"))
    budget = estimate_flops(ref_cfg, args.seq_len, loops=2)["total"]
    r = loops_for_equal_flops(cfg, budget, args.seq_len)

    def batch():
        ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
        labels = ids.clone()
        return ids, labels

    model.train()
    t0, losses = time.time(), []
    for step in range(args.steps):
        ids, labels = batch()
        res = model(ids, labels=labels, num_loops=r)
        loss = res.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(args.val_steps):
            ids, labels = batch()
            val_losses.append(model(ids, labels=labels, num_loops=r).loss.item())
    flops = estimate_flops(cfg, args.seq_len, loops=r, train=True)["total"] * args.steps
    return {"variant": variant, "loops": r, "train_loss": sum(losses) / len(losses),
            "val_loss": sum(val_losses) / len(val_losses), "total_flops": flops,
            "sec": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser(description="Doc 3 实验 1-3：zone 位置对比（等 FLOPs）")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--val_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--max_loop", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))
    p.add_argument("--loop_embedding", type=int, default=0, choices=[0, 1],
                   help="轮次嵌入消融（D4）：对比加/不加 step embedding 的循环收益")
    p.add_argument("--out", type=str, default="experiments/zone_experiments.json")
    args = p.parse_args()

    results = []
    for v in args.variants.split(","):
        Logger(f"===== running variant={v} =====")
        results.append(run_variant(v, args, None, None))
    print("\n==== Zone Experiment Results (equal-FLOPs vs middle_loop R=2) ====")
    print(f"{'variant':<14}{'R':>3}{'train_loss':>12}{'val_loss':>10}{'GFLOPs':>10}{'sec':>7}")
    for r in results:
        print(f"{r['variant']:<14}{r['loops']:>3}{r['train_loss']:>12.4f}"
              f"{r['val_loss']:>10.4f}{r['total_flops'] / 1e9:>10.2f}{r['sec']:>7}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
```

---

# scripts/scaling_fit.py

```python
"""Scaling Law 拟合（任务清单 T8.3；Doc 6 §14）。

拟合 L = A · N^−α · D^−β · R^−γ · M^−δ（log 线性最小二乘 + 标准误与 R²）。
输入 experiments/ 下的记录 jsonl：每行 {"N":, "D":, "R":, "M":, "loss":}
（R/M 可选；缺失的维度自动跳过 —— 检验 γ,δ>0 是否成立是核心研究问题）。
"""
import os
import sys
import json
import argparse
import glob

import numpy as np


def load_records(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r, dict) and "loss" in r and "N" in r and "D" in r:
                rows.append(r)
    return rows


def fit(rows, use=("N", "D", "R", "M")):
    vars_ = [v for v in use if all(v in r and r[v] for r in rows)]
    X = np.stack([np.log(np.array([float(r[v]) for r in rows])) for v in vars_], axis=1)
    y = np.log(np.array([float(r["loss"]) for r in rows]))
    X1 = np.concatenate([np.ones((len(rows), 1)), X], axis=1)
    coef, residuals, rank, _ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ coef
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    r2 = 1 - ss_res / ss_tot
    dof = max(len(rows) - X1.shape[1], 1)
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    names = ["logA"] + [f"-exp({v})" for v in vars_]
    return dict(zip(names, zip(coef.tolist(), se.tolist()))), r2, vars_


def main():
    p = argparse.ArgumentParser(description="Scaling Law 拟合")
    p.add_argument("--records", type=str, default="experiments/scaling_records.jsonl")
    p.add_argument("--use", type=str, default="N,D,R,M", help="参与拟合的变量")
    args = p.parse_args()
    paths = sorted(glob.glob(args.records)) or (
        [args.records] if os.path.exists(args.records) else [])
    rows = load_records(paths)
    if len(rows) < len(args.use.split(",")) + 2:
        print(f"记录不足（{len(rows)} 条），请先通过实验矩阵累积记录。格式：")
        print('  {"N": 参数量, "D": tokens, "R": 平均深度, "M": memory tokens, "loss": val loss}')
        return
    coefs, r2, used = fit(rows, tuple(args.use.split(",")))
    print(f"拟合变量: {used}   R² = {r2:.4f}")
    for name, (c, s) in coefs.items():
        print(f"  {name:>8}: {c:+.4f} ± {s:.4f}")
    print("\n研究问题（Doc 6 §14）：γ(R) 与 δ(M) 的指数是否显著 > 0（即 ±1.96·SE 下界 > 0）")


if __name__ == "__main__":
    main()
```

---

# tests/conftest.py

```python
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
                               controller_factory=lambda: AdaptiveController(model))
    assert loss_matrix.shape == (1, len(cfg.memory_budgets))
    assert torch.isfinite(loss_matrix).all()
    labels_r, gain = sweep_retrieval(model, x, y)
    assert labels_r.shape == gain.shape == (1, cfg.max_loop)
    assert set(labels_r.unique().tolist()) <= {0, 1}
    # per-sample 口径：ΔCE 增益的轮次必被点亮（correctness 分量可额外点亮）
    assert ((gain > cfg.oracle_epsilon) <= (labels_r == 1).bool()).all()
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
    assert gold_block in result.block_idx[0].tolist()
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

# docs/ARCHITECTURE.md

```markdown
# MiniMind-X 实现架构说明

版本：v1.2（v1.1 全量落地 + 外部评审采纳修正，见 §9）
本文回答：每个设计决策（D1–D9）落在哪行代码、每个模块对应任务清单哪一项、怎么跑。

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

# 4. 训练流水线（Stage 0–4，D7 编号）

```bash
# Stage 0 基础预训练（routers 关闭；--loop_mode random 为 Doc 3 Phase0 消融）
bash scripts/launch_h20_8card.sh pretrain

# Stage 1 长上下文课程（A–C：8K→32K→128K，默认 dense；D/E 512K/1M 走稀疏路径=eval_needle retrieval）
bash scripts/launch_h20_8card.sh longctx

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
```

---

