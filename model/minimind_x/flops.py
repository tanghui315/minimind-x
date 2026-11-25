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
                   train=False, topk_coarse=None, topk_fine=None, *, batch_size=1,
                   past_tokens=0, index_blocks=None, indexed_blocks=0, encoded_blocks=0,
                   memory_projections=1, warm_rescores=0, output_tokens=None) -> dict:
    """Dense score shapes, including masked local/sparse implementations. One forward chunk.

    index_blocks measures the searchable corpus, indexed_blocks counts newly built keys,
    encoded_blocks counts cache misses. Without corpus counters retrieval is a lower bound
    based on the supplied resident blocks. External store is shared across the batch.
    """
    d, bs = config.hidden_size, config.block_size
    pre, core, coda = config.prelude_layers, config.thinking_layers, config.coda_layers
    r = max(1, loops) if config.use_loop else 1
    depth = pre + coda + core * r
    s = batch_size * seq_len
    heads = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    inter = config.moe_intermediate_size * config.num_experts_per_tok if config.use_moe else config.intermediate_size
    linear = 2 * (2 * d * heads + 2 * d * kv + 3 * d * inter)
    backbone = s * depth * (linear + 4 * (past_tokens + seq_len) * heads)
    lm_head = 2 * batch_size * (seq_len if output_tokens is None else output_tokens) * d * config.vocab_size
    m = memory_blocks * bs
    v2 = bool(config.residual_v2)
    qh = config.reader_query_heads or max(2, d // 64)
    hd = max(8, min(config.reader_head_dim, d // qh))
    reader_attn = 4 * s * (m + 1) * qh * hd * r if v2 and m else 0
    reader_proj = (4 * s * d * qh * hd * r +
                   4 * batch_size * m * d * config.reader_kv_heads * hd * memory_projections) if v2 and m else 0
    reader_gate = 2 * s * r * (2 * d * 32 + 32) if v2 and m else 0
    mhar = 4 * s * d * (r * (r + 1) / 2) if v2 else 0
    update = 2 * s * r * 4 * d * config.update_rank if v2 else 0
    legacy = 4 * core * (s * r * m * heads + batch_size * m * d * kv * memory_projections) if not v2 else 0
    n = memory_blocks if index_blocks is None else index_blocks
    candidates = min(n, config.topk_coarse if topk_coarse is None else topk_coarse)
    index = indexed_blocks * (bs * d + 2 * d * d)
    encoding = encoded_blocks * bs * pre * (linear + 4 * bs * heads)
    search = batch_size * (n_retrievals * 2 * n * d +
             (n_retrievals + warm_rescores) * (2 * d * d + 2 * candidates * bs * d))
    residual = reader_attn + reader_proj + reader_gate + mhar + update
    forward = backbone + lm_head + legacy + residual + index + encoding + search
    return dict(total=float(forward * (3 if train else 1)), backbone=backbone, lm_head=lm_head,
                memory_attn=legacy, retrieval=search, index_build=index, candidate_encoding=encoding,
                reader_attn=reader_attn, reader_proj=reader_proj, reader_gate=reader_gate, mhar=mhar,
                update_gate=update, residual_modules=residual, effective_depth=depth)
