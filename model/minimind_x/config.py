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
