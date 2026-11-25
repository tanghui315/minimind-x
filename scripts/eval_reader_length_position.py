"""Frozen paired QA: nested neutral prefixes, fixed-length positions and short stale-value controls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_fact_swap import encode_text
from scripts.eval_paired_reader import run_channel, state_digest
from scripts.eval_reader_distractors import CHANNELS, filler, treatments
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval

ADDED = (0, 2, 4, 8, 16, 32, 64, 128, 192)
TOTALS = (64, 128, 256)
SEEDS = (1701, 1702)
WORLDS = ("original", "edited")


def layouts(tok, fact_length, line, entity, decoy):
    """Yield (cell, before, after); preserve intact sentences at the middle split."""
    yield "clean", [], []
    for seed in SEEDS:
        rng = random.Random(f"length/{seed}/{line}")
        before = []
        for n in ADDED[1:]:
            if n <= 8:
                # Ensure both backgrounds really differ even when only 2 or 4 tokens fit.
                choices = {
                    2: ("正常。", "稳定。"),
                    4: ("完成。", "在线。"),
                    8: ("校验通过。", "自检通过。"),
                }
                chunk = encode_text(tok, choices[n][(line + SEEDS.index(seed)) % 2])
                if len(chunk) != n - len(before):
                    raise ValueError("Short telemetry phrase has unexpected token length")
            else:
                chunk = filler(tok, n - len(before), rng)
            before = chunk + before
            yield f"length/{seed}/{n}", before, []
        for total in TOTALS:
            free = total - fact_length
            if free < 0:
                raise ValueError("A support passage exceeds the position experiment length")
            rng = random.Random(f"position/{seed}/{line}/{total}")
            left = filler(tok, free // 2, rng)
            right = filler(tok, free - len(left), rng)
            for pos, a, b in (
                ("head", [], left + right),
                ("middle", left, right),
                ("tail", left + right, []),
            ):
                yield f"position/{seed}/{total}/{pos}", a, b
    slots, _ = treatments(tok, entity, decoy, random.Random(f"short-slot/{line}"))
    for position in ("before", "after"):
        for arm in ("neutral", "entity_count", "obsolete_threshold"):
            slot = slots[arm]
            yield (
                f"short/{position}/{arm}",
                slot if position == "before" else [],
                slot if position == "after" else [],
            )


def metrics(rows, cell, channel):
    pairs = correct = flips = cap = decoy_hits = 0
    for row in rows:
        a, b = [row["cells"][cell][w][channel] for w in WORLDS]
        la, lb = a["candidate_mean_logps"], b["candidate_mean_logps"]
        pairs += a["correct"] and b["correct"]
        flips += la[0] > la[1] and lb[1] > lb[0]
        for x in (a, b):
            correct += x["correct"]
            cap += len(x["generated_ids"]) == 32
            decoy_hits += x["prediction"].strip() == row["decoy"]
    return dict(
        n=len(rows), pairs=pairs, correct=correct, flips=flips, at_cap=cap, generated_decoy=decoy_hits
    )


def transitions(rows, control, changed, channel):
    """Preserve directions: a gain on a different item cannot cancel a loss."""
    losses, gains = [], []
    pair_losses, pair_gains = [], []
    base_correct = base_pairs = 0
    for row in rows:
        a, b = [
            [row["cells"][cell][w][channel]["correct"] for w in WORLDS] for cell in (control, changed)
        ]
        base_pairs += all(a)
        if all(a) and not all(b):
            pair_losses.append(row["line"])
        if all(b) and not all(a):
            pair_gains.append(row["line"])
        base_correct += sum(a)
        for w, x, y in zip(WORLDS, a, b):
            if x != y:
                (losses if x else gains).append([row["line"], w])
    return dict(
        base_correct=base_correct,
        losses=losses,
        gains=gains,
        base_pairs=base_pairs,
        pair_losses=pair_losses,
        pair_gains=pair_gains,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distractors", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    prev = json.loads(args.distractors.read_text())
    ref = Path(prev["reference"])

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    if sha(ref) != prev["reference_sha256"]:
        raise ValueError("Mismatched baseline")
    base = json.loads(ref.read_text())
    cfg = base["config"]
    for k in ("checkpoint", "data"):
        if sha(cfg[k]) != base[k + "_sha256"]:
            raise ValueError(f"Changed {k}")
    raw = Path(cfg["data"]).read_text().splitlines()
    model = load_x_for_eval(".", cfg["checkpoint"], device="cpu").eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    bind_x_tokenizer(model, tok)
    digest = state_digest(model)
    if digest != base["model_state_sha256"]:
        raise ValueError("Model state changed")
    rows = []
    depth = Counter()

    def observe(module, inputs, out):
        depth[out.num_loops_executed] += 1
        if out.num_loops_executed != 2:
            raise AssertionError("Depth changed")

    with torch.inference_mode(), model.register_forward_hook(observe):
        for prior, control in zip(base["samples"], prev["samples"], strict=True):
            keys = ("line", "question", "family", "original_answer", "edited_answer")
            if any(prior[k] != control[k] for k in keys):
                raise ValueError("Pair identity changed")
            sample = json.loads(raw[prior["line"] - 1])
            decoy = control["decoy"]
            targets = [
                encode_text(tok, s) for s in (prior["original_answer"], prior["edited_answer"], decoy)
            ]
            fact = {w: prior[w]["sentence"]["evidence_ids"] for w in WORLDS}
            n = len(fact["original"])
            if len(fact["edited"]) != n:
                raise ValueError("Fact edit changes token length")
            cells = {}
            for cell, before, after in layouts(tok, n, prior["line"], control["entity"], decoy):
                data = dict(before=before, after=after, fact_span=[len(before), len(before) + n])
                for w in WORLDS:
                    ids = before + fact[w] + after
                    if len(ids) > model.config.block_size:
                        raise ValueError("Would change Reader block count")
                    world = dict(sample, answer=prior[w + "_answer"])
                    release = torch.tensor(prior[w]["sentence"]["release_rounds"])
                    data[w] = {}
                    for ch in CHANNELS:
                        x = run_channel(
                            model, tok, world, torch.tensor(ids), [0], release, ids, ch, 2, 32, targets
                        )
                        if x["input_length"] + 31 > model.config.max_position_embeddings:
                            raise ValueError("Position extrapolation would confound scan")
                        baseline = prior[w]["sentence"][ch]
                        if x["before_release_ids"] != baseline["before_release_ids"]:
                            raise AssertionError("Evidence released at a different boundary")
                        if cell == "clean" and (
                            x["generated_ids"] != baseline["generated_ids"]
                            or x["candidate_mean_logps"][:2] != baseline["candidate_mean_logps"]
                        ):
                            raise AssertionError("Clean baseline failed to replay")
                        data[w][ch] = x
                cells[cell] = data
            rows.append(
                dict(
                    **{k: prior[k] for k in keys},
                    entity=control["entity"],
                    decoy=decoy,
                    fact_ids=fact,
                    cells=cells,
                )
            )
            print("Completed validation line", prior["line"], flush=True)
    if state_digest(model) != digest:
        raise AssertionError("State changed during evaluation")
    groups = dict(
        all=rows,
        immediate=[r for r in rows if r["family"] != "new_evidence"],
        delayed=[r for r in rows if r["family"] == "new_evidence"],
    )
    summary = {
        g: {cell: {ch: metrics(group, cell, ch) for ch in CHANNELS} for cell in rows[0]["cells"]}
        for g, group in groups.items()
    }
    changes = {
        g: {
            pos: {
                ch: transitions(
                    group, f"short/{pos}/entity_count", f"short/{pos}/obsolete_threshold", ch
                )
                for ch in CHANNELS
            }
            for pos in ("before", "after")
        }
        for g, group in groups.items()
    }
    result = dict(
        checkpoint_sha256=base["checkpoint_sha256"],
        data_sha256=base["data_sha256"],
        reference_sha256=sha(ref),
        distractors_sha256=sha(args.distractors),
        model_state_sha256=digest,
        depth_histogram=depth,
        native_positions=model.config.max_position_embeddings,
        added=ADDED,
        totals=TOTALS,
        seeds=SEEDS,
        summary=summary,
        short_transitions=changes,
        samples=rows,
    )
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
