"""双执行器（任务清单 T4.6；V2 §10.4）。

1. reference_forward_by_prefix：逐前缀参考执行器——每个位置 t 用可见前缀 x[0..t]
   与「截止 t 的 memory 时间线」独立前向，是正确性标准。
2. segmented_forward：按 memory-version/动作边界分段的批处理执行器——段内 Memory
   在段开始前已确定；利用因果性，段内位置可共用一次批处理前向。

共同语义（两执行器必须逐位对应）：
  per_token_ce[t] = CE( ŷ_{t+1} | x_{0..t}, M(t) )，t < T-1；per_token_ce[T-1] = 0
  M(t) = 该 token 所在 segment 生效的 memory（在 segment 开始前已定）

动作时间线：retrieve_actions[i] 是「进入 segment i 时」执行的动作
（0=REUSE, 1=RESCORE_WARM, 2=GLOBAL_REFRESH）；非 REUSE 在段边界用可见前缀查询执行。
训练用给定动作边界时，检索 query 仍只由当前可见前缀计算（T0.1）。
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
    """构造动作时间线。retrieve_actions[i]：进入 segment i 时执行的动作 id。"""
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
    """按时间线预演 memory：返回 list[mem_kw]，与 timeline 一一对应（段开始前已定）。"""
    cfg = model.config
    T = input_ids.shape[1]
    ctrl = controller_factory()
    ctrl.begin(context_ids)
    causal = input_ids[:, :query_len]
    causal_mask = attention_mask[:, :query_len] if attention_mask is not None else None
    q0 = ctrl.raw.model.prelude_hidden(causal, causal_mask)
    ctrl.memory = ctrl.retrieve_into_memory(q0)  # 初始全局检索

    mem_list = []
    for seg in timeline:
        a = int(seg["retrieval_action"])
        t0 = int(seg["token_start"])
        # 进入本段时执行动作（query 只用可见前缀，T0.1）
        if a != ACTION_REUSE:
            vis_len = max(query_len, min(t0 + 1, T))
            vis = ctrl.raw.model.prelude_hidden(input_ids[:, :vis_len],
                                               attention_mask[:, :vis_len]
                                               if attention_mask is not None else None)
            ctrl.memory = ctrl.execute_retrieval_action(
                a, vis, budget_idx=ctrl._budget_idx(vis), mem=ctrl.memory)
        mem_list.append(_mem_kw(ctrl))
    return mem_list


def _ce_at_positions(logits, labels, positions: Sequence[int]) -> torch.Tensor:
    """per_token_ce[t] = CE(logits[t] → labels[t+1])。返回 [B, T]（未填位置为 0）。"""
    B, T, _ = logits.shape
    out = logits.new_zeros(B, T)
    for t in positions:
        if t + 1 >= T:
            continue
        out[:, t] = F.cross_entropy(logits[:, t, :], labels[:, t + 1],
                                    ignore_index=-100, reduction="none")
    return out


@torch.no_grad()
def reference_forward_by_prefix(model, input_ids, labels, context_ids, query_len,
                                timeline, controller_factory, attention_mask=None,
                                num_loops=None):
    """逐前缀参考执行器（正确性标准）。返回 (per_token_ce [B,T], last_logits)。"""
    cfg = model.config
    B, T = input_ids.shape
    loops = 1 if not cfg.use_loop else (num_loops or min(cfg.max_loop, cfg.max_loop))
    mem_list = _prepare_memories(model, input_ids, context_ids, query_len, timeline,
                                 controller_factory, attention_mask)

    def mem_at(t):
        for seg, mk in zip(timeline, mem_list):
            if seg["token_start"] <= t < seg["token_end"]:
                return mk
        return mem_list[-1] if mem_list else {}

    per_token_ce = torch.zeros(B, T, device=input_ids.device, dtype=torch.float32)
    if labels is None:
        return per_token_ce, None
    last_logits = None
    for t in range(T):
        if t + 1 >= T:
            break  # 位置 T-1 无 next
        pref = input_ids[:, :t + 1]
        pref_mask = attention_mask[:, :t + 1] if attention_mask is not None else None
        out = model(pref, attention_mask=pref_mask, labels=None, num_loops=loops,
                    **mem_at(t))
        last_logits = out.logits
        per_token_ce[:, t] = F.cross_entropy(
            out.logits[:, -1, :], labels[:, t + 1],
            ignore_index=-100, reduction="none")
    return per_token_ce, last_logits


@torch.no_grad()
def segmented_forward(model, input_ids, labels, context_ids, query_len,
                      timeline, controller_factory, attention_mask=None,
                      num_loops=None):
    """分段批处理执行器（优化路径）。返回 (per_token_ce [B,T], last_logits)。

    段内 Memory 已在段开始前确定；因因果 mask，位置 t∈[s,e) 只看见 x_{0..t}，
    故可用一次前向（整序列 + 本段 memory）抽出段内 per-token CE，与逐前缀逐位一致。
    """
    cfg = model.config
    B, T = input_ids.shape
    loops = 1 if not cfg.use_loop else (num_loops or cfg.max_loop)
    mem_list = _prepare_memories(model, input_ids, context_ids, query_len, timeline,
                                 controller_factory, attention_mask)

    per_token_ce = torch.zeros(B, T, device=input_ids.device, dtype=torch.float32)
    last_logits = None
    if labels is None:
        return per_token_ce, None
    for seg, mk in zip(timeline, mem_list):
        s, e = int(seg["token_start"]), int(seg["token_end"])
        s, e = max(0, s), min(T, e)
        if s >= e:
            continue
        out = model(input_ids, attention_mask=attention_mask, labels=None,
                    num_loops=loops, **mk)
        last_logits = out.logits
        # 段内位置 t ∈ [s, e)：CE(logits[t] → labels[t+1])；t=T-1 无 next 保持 0
        positions = list(range(s, min(e, T - 1)))
        if positions:
            per_token_ce[:, s:min(e, T - 1)] = _ce_at_positions(
                out.logits, labels, positions)[:, s:min(e, T - 1)]
    return per_token_ce, last_logits


def assert_executors_correspond(ref_ce, seg_ce, atol=1e-5):
    """T4.6 验收：两执行器语义对应（数值容差内逐位一致）。"""
    if ref_ce.shape != seg_ce.shape:
        return False, f"shape mismatch {ref_ce.shape} vs {seg_ce.shape}"
    if not torch.isfinite(ref_ce).all() or not torch.isfinite(seg_ce).all():
        return False, "non-finite CE"
    diff = (ref_ce - seg_ce).abs().max().item()
    return diff <= atol, f"max |ΔCE|={diff}"
