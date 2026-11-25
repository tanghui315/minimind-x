"""Paired, single-decision counterfactuals for MiniMind-X retrieval QA."""
import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
import sys
import time

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.retrieval_qa_dataset import encode_evidence
from model.minimind_x.controller import AdaptiveController
from scripts.eval_needle import exact_match
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval

ACTIONS = ("stop", "reuse", "rescore_warm", "global_refresh")


def fork_controller(source, cold=False):
    """Share read-only tensors and model. Isolate mutable request state and lazy caches."""
    ctrl = copy.copy(source)
    ctrl.store = copy.copy(source.store)
    ctrl.store._hidden_cache = {} if cold else dict(source.store._hidden_cache)
    ctrl.store._cache_order = [] if cold else list(source.store._cache_order)
    ctrl.memory = copy.copy(source.memory)
    ctrl.warm_candidates = copy.copy(source.warm_candidates)
    ctrl.trajectory = list(source.trajectory)
    ctrl.stats = dict(source.stats)
    return ctrl


def memory_kw(ctrl):
    key = "memory_payload" if ctrl.cfg.residual_v2 else "memory_kv_per_layer"
    return {key: ctrl._memory_input()}


@dataclass(frozen=True)
class Policy:
    depth: int = None
    initial_only: bool = False


def decode(ctrl, ids, past, policy, prefill=False):
    cfg = ctrl.cfg
    use_halt = policy.depth is None and (not prefill or getattr(cfg, "prefill_policy", "fixed_cap") != "fixed_cap")
    return ctrl.raw(ids, past_key_values=past, use_cache=True, logits_to_keep=1,
                    num_loops=policy.depth or cfg.max_loop, halt=use_halt, collect_routers=False,
                    loop_callback=None if policy.initial_only else ctrl.make_loop_callback(),
                    **memory_kw(ctrl))


def prepare(model, question, delimiter, context, visibility, policy=Policy(), budget_idx=None):
    """Match the SFT question prefill. Consume only the supplied, known answer delimiter."""
    ctrl = AdaptiveController(model, block_visible_from=visibility, fixed_budget_idx=budget_idx)
    ctrl.begin(context)
    q = model.model.prelude_hidden(question)
    ctrl.memory = ctrl.retrieve_into_memory(q, ctrl._budget_idx(q))
    out = decode(ctrl, question, None, policy, prefill=True)
    for chunk in delimiter.split(1, dim=1)[:-1]:
        out = decode(ctrl, chunk, out.past_key_values, policy)
    return ctrl, out.past_key_values


@dataclass(frozen=True)
class Decision:
    ctrl: AdaptiveController
    chunk: torch.Tensor
    past: list
    prefix: object
    depth: int
    router: str
    legal: tuple


def capture(ctrl, chunk, past, depth):
    cfg = ctrl.cfg
    if not 1 <= depth <= cfg.max_loop:
        raise ValueError("decision round outside model depth")
    snap = ctrl.raw.model(chunk, past_key_values=past, use_cache=True, num_loops=depth,
                          capture_round=True, collect_routers=False,
                          loop_callback=ctrl.make_loop_callback(), **memory_kw(ctrl))
    legal, router = ("stop",), "stop"
    if depth < cfg.max_loop:
        legal = tuple(a for a in ACTIONS if a != "rescore_warm" or ctrl.has_warm_cache())
        hidden = snap.core["exit_hiddens"][-1]
        stats = snap.core["residual_stats"][-1]
        if ctrl.raw.model.halt_router(hidden, stats).item() <= cfg.halt_threshold:
            logits = ctrl.raw.model.retrieve_router.action_logits(hidden, ctrl.has_warm_cache())
            router = ACTIONS[1 + int(logits.argmax(-1))]
    return Decision(ctrl, chunk, past, snap, depth, router, legal)


def branch(decision, ctrl, action):
    if action not in decision.legal:
        raise ValueError(f"Unavailable action {action}")
    # Override exactly one decision; all subsequent decisions use the normal controller.
    forced = [0] * (decision.depth - 1) + [max(0, ACTIONS.index(action) - 1)]
    return ctrl.raw(decision.chunk, past_key_values=decision.past, use_cache=True, logits_to_keep=1,
                    round_prefix=decision.prefix, num_loops=decision.depth if action == "stop" else ctrl.cfg.max_loop,
                    halt=action != "stop", loop_callback=ctrl.make_loop_callback(oracle_retrieve=forced),
                    collect_routers=False, **memory_kw(ctrl))


def complete(ctrl, out, max_tokens, eos, policy=Policy()):
    ids = []
    for step in range(max_tokens):
        token = out.logits[0, -1].argmax()
        if token.item() == eos:
            break
        ids.append(int(token))
        if step + 1 < max_tokens:
            out = decode(ctrl, token.view(1, 1), out.past_key_values, policy)
    return ids


def score(ctrl, out, reference, policy=Policy()):
    logps = []
    for i, token in enumerate(reference):
        logps.append(out.logits[0, -1].float().log_softmax(-1)[token])
        if i + 1 < len(reference):
            out = decode(ctrl, token.view(1, 1), out.past_key_values, policy)
    return torch.stack(logps).tolist()


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter_ns()


def evidence(ctrl):
    return sorted(ctrl.stats.get("resident_source_ids", []))


def evaluate_decision(decision, reference, tokenizer, answer, cap, repeats, cache_mode, seed):
    device = decision.chunk.device
    rng = random.Random(seed)
    result = {}
    for action in decision.legal:
        ctrl = fork_controller(decision.ctrl, cold=cache_mode == "cold")
        out = branch(decision, ctrl, action)
        sources = evidence(ctrl)
        lp = score(ctrl, out, reference)
        result[action] = dict(token_logps=lp, mean_logp=mean(lp), source_ids_at_exit=sources,
                              decision_ms_samples=[], generation_ms_samples=[])
    # Warm up each complete execution. Rotate order across paired timing repetitions.
    for rep in range(repeats + 1):
        order = list(decision.legal)
        rng.shuffle(order)
        for action in order:
            ctrl = fork_controller(decision.ctrl, cold=cache_mode == "cold")
            start = clock(device)
            out = branch(decision, ctrl, action)
            first = clock(device)
            tokens = complete(ctrl, out, cap, tokenizer.eos_token_id)
            end = clock(device)
            item = result[action]
            if rep:
                item["decision_ms_samples"].append((first - start) / 1e6)
                item["generation_ms_samples"].append((end - start) / 1e6)
                if tokens != item["generated_ids"]:
                    raise RuntimeError("Greedy generation changed between repetitions")
            else:
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                item.update(generated_ids=tokens, prediction=text, correct=int(exact_match(text, answer)))
    for item in result.values():
        item["generation_ms"] = median(item["generation_ms_samples"])
        item["decision_ms"] = median(item["decision_ms_samples"])
        for key in ("generation_ms", "decision_ms"):
            item["delta_" + key] = median(t - base for t, base in zip(
                item[key + "_samples"], result["stop"][key + "_samples"]))
        item["delta_logp"] = item["mean_logp"] - result["stop"]["mean_logp"]
    return result


def matched_em(rows, budget):
    """Exact multiple-choice oracle for binary EM and a summed measured latency budget."""
    base_cost, hits, upgrades = 0., 0, []
    for row in rows:
        arms = list(row["actions"].values())
        cheapest = min(arms, key=lambda v: (v["generation_ms"], -v["correct"]))
        base_cost += cheapest["generation_ms"]
        hits += cheapest["correct"]
        correct_cost = [a["generation_ms"] for a in arms if a["correct"]]
        if correct_cost and not cheapest["correct"]:
            upgrades.append(min(correct_cost) - cheapest["generation_ms"])
    if base_cost > budget + 1e-8:
        return None
    for delta in sorted(upgrades):
        if base_cost + delta > budget + 1e-8:
            break
        base_cost += delta
        hits += 1
    return dict(correct=hits, used_ms=base_cost, budget_ms=budget)


def matched_logp(rows, budget):
    """Multiple-choice knapsack, solved on observed per-example median generation times."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, vstack
    arms = [(i, v) for i, row in enumerate(rows) for v in row["actions"].values()]
    index, items = zip(*arms)
    count = len(items)
    choose_one = csr_matrix((np.ones(count), (index, np.arange(count))), shape=(len(rows), count))
    latency = np.array([v["generation_ms"] for v in items])
    constraints = LinearConstraint(vstack([choose_one, csr_matrix(latency)]),
                                   [*np.ones(len(rows)), -np.inf], [*np.ones(len(rows)), budget + 1e-8])
    # Center by the observed per-example minimum for meaningful solver gaps on small gains.
    floor = [min(v["mean_logp"] for v in r["actions"].values()) for r in rows]
    gain = np.array([v["mean_logp"] - floor[i] for i, v in arms])
    solution = milp(-gain, integrality=np.ones(count), bounds=Bounds(0,1), constraints=constraints,
                    options={"mip_rel_gap": 0.0})
    if not solution.success:
        raise RuntimeError(f"Could not solve latency-constrained oracle: {solution.message}")
    take = solution.x > .5
    used = float(latency[take].sum())
    if used > budget + 1e-5 or not np.all(choose_one @ take == 1):
        raise RuntimeError("Infeasible oracle selection")
    return dict(mean_logp=(sum(floor) + gain[take].sum())/len(rows), used_ms=used, budget_ms=budget)


def policy_stats(items):
    return dict(n=len(items), correct=sum(x["correct"] for x in items),
                mean_logp=mean(x["mean_logp"] for x in items),
                mean_generation_ms=mean(x["generation_ms"] for x in items))


def summarize(rows, constrained=False):
    router = [r["actions"][r["router"]] for r in rows]
    stop = [r["actions"]["stop"] for r in rows]
    # Actions unavailable in some states cannot be used as a single fixed policy on that set.
    local_fixed = {a: policy_stats([r["actions"][a] for r in rows]) for a in ACTIONS
                   if all(a in r["actions"] for r in rows)}
    best_lp = [max(a["mean_logp"] for a in r["actions"].values()) for r in rows]
    ceiling = sum(max(a["correct"] for a in r["actions"].values()) for r in rows)
    delta = [b - p["mean_logp"] for b, p in zip(best_lp, router)]
    result = dict(n=len(rows), router=policy_stats(router), stop=policy_stats(stop),
                local_fixed=local_fixed, oracle_correct=ceiling, oracle_mean_logp=mean(best_lp),
                oracle_logp_gain_over_router=mean(delta), oracle_logp_gain_over_best_fixed=mean(best_lp) -
                max(p["mean_logp"] for p in local_fixed.values()),
                matched_router_ms=matched_em(rows, sum(x["generation_ms"] for x in router)),
                router_actions=dict(Counter(r["router"] for r in rows)),
                legal_counts={a:sum(a in r["actions"] for r in rows) for a in ACTIONS},
                action_contrast={a:dict(n=len(items),mean_delta_logp=mean(v["delta_logp"] for v in items),
                    median_paired_delta_decision_ms=median(v["delta_decision_ms"] for v in items))
                    for a in ACTIONS if (items := [r["actions"][a] for r in rows if a in r["actions"]])})

    for base, alternative in (("stop","reuse"), ("reuse","rescore_warm"), ("reuse","global_refresh"),
                              ("rescore_warm","global_refresh")):
        pairs = [(r["actions"][base], r["actions"][alternative]) for r in rows
                 if base in r["actions"] and alternative in r["actions"]]
        if pairs:
            changes = [b["mean_logp"] - a["mean_logp"] for a,b in pairs]
            result.setdefault("paired_contrasts", {})[alternative+"_vs_"+base] = dict(
                n=len(pairs), mean_delta_logp=mean(changes), better_gt_0p0001=sum(v > 1e-4 for v in changes),
                worse_gt_0p0001=sum(v < -1e-4 for v in changes),
                changed_sources=sum(a["source_ids_at_exit"] != b["source_ids_at_exit"] for a,b in pairs),
                changed_generation=sum(a["generated_ids"] != b["generated_ids"] for a,b in pairs))
    if constrained:
        budget = sum(x["generation_ms"] for x in router)
        result["matched_logp"] = matched_logp(rows, budget)
        for a, metrics in local_fixed.items():
            fixed_budget = metrics["mean_generation_ms"] * len(rows)
            result.setdefault("matched_fixed", {})[a] = dict(em=matched_em(rows, fixed_budget),
                                                            logp=matched_logp(rows, fixed_budget))
        groups = defaultdict(list)
        for i, (row, gain) in enumerate(zip(rows, delta)):
            group = row["deployment_id"] if row.get("deployment_id") is not None else ("row", i)
            groups[group].append(gain)
        rng = random.Random(413)
        blocks = list(groups.values())
        draws = sorted(mean(x for block in rng.choices(blocks, k=len(blocks)) for x in block)
                       for _ in range(1000))
        result["gain_bootstrap_95"] = [draws[24], draws[974]]
        result["bootstrap_groups"] = len(groups)
    return result


def full_baseline(model, inputs, reference, tok, answer, cap, policy, budget_idx, repeats):
    question, delimiter, context, visibility = inputs
    def first():
        ctrl, past = prepare(model, *inputs, policy, budget_idx)
        return ctrl, decode(ctrl, delimiter[:, -1:], past, policy)
    ctrl, out = first()
    lp = score(ctrl, out, reference, policy)
    times = []
    for rep in range(repeats + 1):
        start = clock(question.device)
        ctrl, out = first()
        ids = complete(ctrl, out, cap, tok.eos_token_id, policy)
        end = clock(question.device)
        if rep:
            times.append((end-start)/1e6)
    return dict(mean_logp=mean(lp), correct=int(exact_match(tok.decode(ids, skip_special_tokens=True), answer)),
                generation_ms=median(times), generated_ids=ids)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="Validation JSONL with question, context, answer")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=str(ROOT/"model"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--budget-idx", type=int)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cache", choices=["cold","request","both"], default="both")
    p.add_argument("--seed", type=int, default=19)
    args = p.parse_args()
    if min(args.max_new_tokens,args.repeats) < 1:
        p.error("repeats and generation cap must be positive")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_x_for_eval(".", args.checkpoint, device=args.device)
    cfg = model.config
    if not (cfg.use_memory and cfg.use_retrieval and cfg.use_loop):
        p.error("Checkpoint must support memory, retrieval and recurrent depth")
    if args.budget_idx is not None and not 0 <= args.budget_idx < len(cfg.memory_budgets):
        p.error("budget index out of range")
    bind_x_tokenizer(model,tok)
    device = torch.device(args.device)
    rows, baselines = [], defaultdict(list)
    with open(args.data, encoding="utf-8") as data:
        for line_no, line in enumerate(data, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("split", "val") != "val":
                continue
            context, gold, visibility = encode_evidence(sample, tok, cfg.block_size)
            def ids(text):
                return tok(text, add_special_tokens=False).input_ids
            question = torch.tensor([[tok.bos_token_id, *ids(sample["question"])]], device=device)
            delimiter = torch.tensor([ids("答案：")], device=device)
            answer = str(sample["answer"])
            ref = torch.tensor(ids(answer), device=device, dtype=torch.long)
            if not ref.numel() or not delimiter.numel():
                raise ValueError(f"Empty reference or delimiter at line {line_no}")
            if question.numel()+delimiter.numel()+max(ref.numel(),args.max_new_tokens)-1 > cfg.max_position_embeddings:
                raise ValueError(f"Question and continuation exceed position limit at line {line_no}")
            inputs = question, delimiter, context.to(device), visibility.to(device)
            ctrl, past = prepare(model,*inputs,budget_idx=args.budget_idx)
            decision = capture(ctrl, delimiter[:, -1:], past, args.round)
            warm = ctrl.warm_candidates
            shortlist = warm["cand_idx"].flatten().tolist() if ctrl.has_warm_cache() else []
            state = ("no_annotated_gold" if gold < 0 else "resident" if gold in evidence(ctrl)
                     else "warm_only" if gold in shortlist else "outside_shortlist")
            for mode in (["request","cold"] if args.cache == "both" else [args.cache]):
                arms = evaluate_decision(decision,ref,tok,answer,args.max_new_tokens,args.repeats,
                                         mode,args.seed+line_no)
                rows.append(dict(line=line_no, family=sample.get("family","unspecified"), cache=mode,
                            deployment_id=sample.get("deployment_id"), question=sample["question"], answer=answer,
                            prefix_ids=[*question.flatten().tolist(),*delimiter.flatten().tolist()],
                            context_tokens=context.numel(), gold_block=gold, gold_location=state,
                            initial_sources=evidence(ctrl), warm_candidates=shortlist,
                            visible_round=ctrl.store.round, memory_version=ctrl.memory.memory_version,
                            router=decision.router, actions=arms))
            policies = {"router":Policy(), "initial_only":Policy(initial_only=True)}
            policies.update({f"fixed_R{r}":Policy(depth=r) for r in range(1,cfg.max_loop+1)})
            if args.budget_idx is not None:
                policies["native_router"] = Policy()
            for name, policy in policies.items():
                budget_idx = None if name == "native_router" else args.budget_idx
                result = full_baseline(model,inputs,ref,tok,answer,args.max_new_tokens,policy,budget_idx,args.repeats)
                baselines[name].append(dict(line=line_no, **result))
            if len(baselines["router"]) % 7 == 0:
                print(f"Evaluated {len(baselines['router'])} validation examples", flush=True)
    if not rows:
        raise ValueError("No validation records")
    summary = {}
    for mode in sorted({r["cache"] for r in rows}):
        subset = [r for r in rows if r["cache"] == mode]
        summary[mode] = summarize(subset, constrained=True)
        for field in ("family","gold_location"):
            summary[mode]["by_"+field] = {v:summarize([r for r in subset if r[field]==v])
                                          for v in sorted({r[field] for r in subset})}
    fingerprint = hashlib.sha256()
    with open(args.checkpoint, "rb") as weights:
        for block in iter(lambda: weights.read(1024 * 1024), b""):
            fingerprint.update(block)
    report = dict(config=vars(args), checkpoint_sha256=fingerprint.hexdigest(), checkpoint_id=getattr(model,"checkpoint_id",None),
                  data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                  architecture=cfg.architecture_version(), max_loop=cfg.max_loop, memory_budgets=cfg.memory_budgets,
                  summary=summary, full_policies={name:policy_stats(items) for name,items in baselines.items()},
                  full_policy_samples=baselines, samples=rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(f"{len(baselines['router'])} examples, {len(rows)} paired snapshots -> {out}")


if __name__ == "__main__":
    main()
