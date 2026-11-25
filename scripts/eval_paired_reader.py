"""Prompt versus pinned Reader on identical evidence tokens and fixed recurrent depth."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model_minimind import precompute_freqs_cis
from model.minimind_x.controller import AdaptiveController
from scripts.eval_counterfactual import Policy, complete, decode, fork_controller, score
from scripts.eval_fact_swap import (
    FAMILIES,
    PinnedEvidence,
    counterfactual,
    encode_text,
    exact_match,
    locate_facts,
    select_evidence,
)
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval


def extend_positions(model, needed):
    core, cfg = model.model, model.config
    if needed <= len(core.freqs_cos):
        return
    tables = precompute_freqs_cis(cfg.head_dim, needed, cfg.rope_theta, cfg.rope_scaling)
    for name, table in zip(("freqs_cos", "freqs_sin"), tables):
        old = getattr(core, name)
        table = table.to(old)
        if not torch.equal(table[: len(old)], old):
            raise ValueError("Extending positions would change existing rotary frequencies")
        setattr(core, name, table)


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def run_channel(
    model,
    tok,
    sample,
    source,
    blocks,
    release,
    token_ids,
    channel,
    depth,
    cap,
    targets=(),
    observer=None,
):
    """Use a common header. Late evidence becomes visible at the next input boundary."""
    reader = channel != "prompt"
    bare = channel == "reader_bare"
    header = [tok.bos_token_id] + ([] if bare else encode_text(tok, "资料："))
    question = ([] if bare else encode_text(tok, "\n问题：")) + encode_text(tok, sample["question"])
    delimiter = encode_text(tok, "答案：")
    suffix = question if reader else token_ids + question
    length = len(header) + len(suffix) + len(delimiter)
    extend_positions(model, length + cap - 1)
    ctrl = AdaptiveController(model)
    ctrl.begin(source[:0])
    policy = Policy(depth=depth, initial_only=True)
    past = None
    release_round = max(int(release[i]) for i in blocks)
    if release_round > depth:
        raise ValueError("Evidence release is after the common header computation")
    before_release = []
    if release_round:
        # Prompt tokens cannot be appended in the middle of a Core loop.
        warmup = decode(ctrl, source.new_tensor([header]), None, policy, prefill=True)
        before_release = header
        past = warmup.past_key_values
    if reader:
        ctrl = PinnedEvidence(model, block_visible_from=release)
        ctrl.begin(source)
        ctrl.gold_ids = blocks
        ctrl.store.advance_round(release_round)
        ctrl.pin_visible()
        memory = ctrl.memory
        if memory.source_ids[0].tolist() != blocks:
            raise AssertionError("Reader source blocks differ from the prescribed ordered blocks")
        ordered = ctrl.store._chunks[memory.source_ids[0]].reshape(-1)
        actual = ordered[memory.token_valid()[0]].tolist()
    ids = source.new_tensor([suffix if release_round else header + suffix])
    if not reader:
        start = 0 if release_round else len(header)
        actual = ids[0, start : start + len(token_ids)].tolist()
    if actual != token_ids:
        raise AssertionError("The evidence delivered to this channel differs from the canonical IDs")
    out = decode(ctrl, ids, past, policy, prefill=past is None)
    for t in delimiter:
        out = decode(ctrl, source.new_tensor([[t]]), out.past_key_values, policy)
    if observer is not None:
        observer(ctrl, out, policy)
    generated = complete(fork_controller(ctrl), out, cap, tok.eos_token_id, policy)
    candidate_logps = []
    for target_ids in targets:
        lp = score(fork_controller(ctrl), out, source.new_tensor(target_ids), policy)
        candidate_logps.append(sum(lp) / len(lp))
    prediction = tok.decode(generated, skip_special_tokens=True)
    target = str(sample["answer"])
    value = prediction.strip()
    mismatch = next((i + 1 for i, (a, b) in enumerate(zip(value, target)) if a != b), None)
    if mismatch is None and len(value) != len(target):
        mismatch = min(len(value), len(target)) + 1
    numbers = re.findall(r"[0-9]+", tok.decode(token_ids))
    return dict(
        prediction=prediction,
        generated_ids=generated,
        correct=exact_match(prediction, target),
        candidate_mean_logps=candidate_logps,
        first_mismatch=mismatch,
        matches_other_number=value != target and value in numbers,
        delivered_sha256=ids_hash(actual),
        delivered_tokens=len(actual),
        input_length=length,
        before_release_ids=before_release,
        release_after_rounds=depth if release_round else 0,
    )


def evaluate_world(model, tok, sample, depth, cap, targets):
    device = next(model.parameters()).device
    facts = locate_facts(sample)
    bs = model.config.block_size
    forms = {}
    for form, mode in (("sentence", "gold_sentences"), ("block", "gold_blocks")):
        source, blocks, release = select_evidence(tok, sample, facts, mode, bs, device)
        if not blocks or len(blocks) > model.config.budget_blocks(0):
            raise ValueError("Missing or over-budget gold evidence")
        tokens = [t for b in blocks for t in source[b * bs : (b + 1) * bs].tolist()]
        arms = {
            channel: run_channel(
                model, tok, sample, source, blocks, release, tokens, channel, depth, cap, targets
            )
            for channel in ("prompt", "reader", "reader_bare")
        }
        forms[form] = dict(
            block_ids=blocks, evidence_ids=tokens, release_rounds=release[blocks].tolist(), **arms
        )
    return forms


def summarize(rows):
    result = {}
    for form in ("sentence", "block"):
        tally = Counter()
        for row in rows:
            outcomes = {}
            for channel in ("prompt", "reader", "reader_bare"):
                pair = [row[w][form][channel] for w in ("original", "edited")]
                outcomes[channel] = all(x["correct"] for x in pair)
                tally[channel] += int(outcomes[channel])
                tally[channel + "_worlds"] += sum(x["correct"] for x in pair)
            category = {
                (True, True): "both",
                (True, False): "prompt_only",
                (False, True): "reader_only",
                (False, False): "neither",
            }
            tally[category[outcomes["prompt"], outcomes["reader"]]] += 1
        result[form] = dict(n=len(rows), **tally)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data", "reference", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--tokenizer", default=str(ROOT / "model"))
    args = p.parse_args()
    ref = json.loads(args.reference.read_text())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if sha(args.checkpoint) != ref["checkpoint_sha256"] or sha(args.data) != ref["data_sha256"]:
        p.error("Reference checkpoint or validation data SHA-256 differs")
    seed = ref["config"]["seed"]
    model = load_x_for_eval(".", str(args.checkpoint), device="cpu").eval().requires_grad_(False)
    if not (1 <= args.depth <= model.config.max_loop and args.max_new_tokens >= 1):
        p.error("Invalid depth or generation cap")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bind_x_tokenizer(model, tok)
    cfg = model.config
    digest = state_digest(model)
    rows = []
    depth_hist = Counter()

    def check_depth(module, inputs, output):
        if output.num_loops_executed != args.depth:
            raise AssertionError("Early exit in fixed-depth evaluation")
        depth_hist[output.num_loops_executed] += 1

    with torch.inference_mode(), model.register_forward_hook(check_depth):
        for line, raw in enumerate(args.data.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            sample = json.loads(raw)
            if sample.get("split", "val") != "val" or sample.get("family") not in FAMILIES:
                continue
            edit = counterfactual(sample, locate_facts(sample), tok, seed)
            targets = [encode_text(tok, s["answer"]) for s in (sample, edit)]
            worlds = [
                evaluate_world(model, tok, s, args.depth, args.max_new_tokens, targets)
                for s in (sample, edit)
            ]
            rows.append(
                dict(
                    line=line,
                    family=sample["family"],
                    question=sample["question"],
                    original_answer=sample["answer"],
                    edited_answer=edit["answer"],
                    original=worlds[0],
                    edited=worlds[1],
                    extrapolated=any(
                        w["block"]["prompt"]["input_length"] + args.max_new_tokens - 1
                        > cfg.max_position_embeddings
                        for w in worlds
                    ),
                )
            )
    identity = ("line", "family", "question", "original_answer", "edited_answer")
    if not rows or [[r[k] for k in identity] for r in rows] != [
        [r[k] for k in identity] for r in ref["samples"]
    ]:
        raise ValueError("Paired validation examples differ from the reference run")
    if state_digest(model) != digest:
        raise AssertionError("Frozen model state changed")
    groups = dict(
        all=rows,
        immediate=[r for r in rows if r["family"] != "new_evidence"],
        delayed=[r for r in rows if r["family"] == "new_evidence"],
        native_length=[r for r in rows if not r["extrapolated"]],
        extrapolated=[r for r in rows if r["extrapolated"]],
    )
    groups.update({f: [r for r in rows if r["family"] == f] for f in FAMILIES})
    report = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        seed=seed,
        checkpoint_sha256=sha(args.checkpoint),
        data_sha256=sha(args.data),
        reference_sha256=sha(args.reference),
        model_state_sha256=digest,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        depth_histogram=depth_hist,
        native_positions=cfg.max_position_embeddings,
        rotary_table_length=len(model.model.freqs_cos),
        summary={k: summarize(v) for k, v in groups.items()},
        samples=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
