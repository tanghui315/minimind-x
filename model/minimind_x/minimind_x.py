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
