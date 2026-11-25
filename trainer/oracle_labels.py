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
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
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
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    cfg = model.config
    idxs = budget_idxs if budget_idxs is not None else list(range(len(cfg.memory_budgets)))
    losses = []
    for idx in idxs:
        ctrl = controller_factory()
        with torch.random.fork_rng(devices=[input_ids.device] if input_ids.is_cuda else []):
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
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
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
                           controller_factory, attention_mask=None, num_loops=None,
                           budget_idx=None, append_only=False):
    """T2.5/T4.4 同快照三动作反事实（评审 4.2 修复）。

    从**同一状态快照**（相同 H/W/Warm Cache/可见前缀）分叉三动作，比较**动作之后**
    的出口 loss——不是三条全程独立轨迹，也不是「第 r 轮 exit 给第 r 轮后动作」打分。

    动作 id（schema v2）：0=REUSE, 1=RESCORE_WARM, 2=GLOBAL_REFRESH。
    返回 (best_action [B,R-1], loss_matrix [B,R-1,3])。
    """
    cfg = getattr(model, "module", model).config
    rounds = (num_loops or cfg.max_loop) - 1
    if input_ids.size(0) != 1:
        raise ValueError("Retrieval oracle requires one external store per sample")
    matrix = torch.full((1, rounds, 3), torch.inf, device=input_ids.device)
    prefix = []
    for r in range(rounds):
        for action in range(3):
            ctrl = controller_factory()
            with torch.random.fork_rng(devices=[input_ids.device] if input_ids.is_cuda else []):
                out, extra = ctrl.forward_with_oracles(
                    input_ids, attention_mask, labels, context_ids, query_len=query_len,
                    oracle_budget=budget_idx, distill=False, append_only=append_only,
                    oracle_retrieve=[*prefix, action] + [0] * (cfg.max_loop - r - 2))
            decisions = [e for e in extra["trajectory"] if "round" in e]
            if action == 1 and not decisions[r]["warm_available"]:
                continue
            matrix[:, r, action] = _per_sample_ce(out.exit_logits[r + 1], labels)
        prefix.append(int(matrix[0, r].argmin()))
    return matrix.argmin(-1), matrix
