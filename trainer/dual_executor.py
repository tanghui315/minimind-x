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
