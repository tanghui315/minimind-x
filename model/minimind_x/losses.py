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
               retrieval_action_masks: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
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
    parts["loss"] = total
    return parts
