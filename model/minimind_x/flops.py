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
