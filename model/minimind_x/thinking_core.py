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
                reader=None, memory_payload=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask)
        hidden_states = hidden_states + residual
        if reader is not None and memory_payload is not None:
            payload, valid = memory_payload
            hidden_states, _ = reader(hidden_states, reader.project_memory(payload, valid))
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
                        memory_payload=memory_payload, use_reentrant=False)
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
                    memory_payload=memory_payload)
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
                gated, gate = self.update_gate(previous, anchor, candidate)
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
            if mv is None and isinstance(memory_payload, tuple) and len(memory_payload) >= 3:
                mv = memory_payload[2]
            memory_versions.append(int(mv) if mv is not None else
                                  (0 if memory_payload is None else id(memory_payload) % 100000))
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
