"""Replay paired Reader answers and find where greedy decoding loses the gold prefix."""

import argparse
from collections import Counter
from pathlib import Path
import random
import json
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_counterfactual import decode, fork_controller
from scripts.eval_fact_swap import encode_text
from scripts.eval_paired_reader import run_channel
from scripts.eval_reader_distractors import filler
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval


class NumericTokens:
    def __init__(self, tok, size):
        self.ids_by_piece = {}
        self.pieces = {}
        self.leading_space = set()
        self.eos = tok.eos_token_id
        for token in range(min(len(tok), size)):
            raw = tok.decode([token], skip_special_tokens=True)
            piece = raw.lstrip()
            if piece.isascii() and piece.isdecimal():
                self.pieces[token] = piece
                if raw != piece:
                    self.leading_space.add(token)
                self.ids_by_piece.setdefault(piece, []).append(token)

    def allowed(self, suffix, *, first=False):
        if not suffix:
            return [self.eos]
        return [
            token
            for n in range(1, len(suffix) + 1)
            for token in self.ids_by_piece.get(suffix[:n], ())
            if first or token not in self.leading_space
        ]


def traces(ctrl, start, policy, tok, numeric, answer, old, gold):
    ctrl = fork_controller(ctrl)
    out = start
    first_bad = None
    greedy = []
    prefix = ""
    logps = []
    for step in range(32):
        score = out.logits[0, -1].float()
        logp = score.log_softmax(-1)
        winner = int(score.argmax())
        if winner != numeric.eos:
            greedy.append(winner)
            logps.append(float(logp[winner]))
        if first_bad is None:
            allowed = numeric.allowed(answer[len(prefix) :], first=not prefix)
            if winner not in allowed:
                good = max(allowed, key=lambda t: float(score[t]))
                old_compatible = old.startswith(prefix)
                old_allowed = (
                    numeric.allowed(old[len(prefix) :], first=not prefix) if old_compatible else []
                )
                # Beyond a shared prefix, a locally old-matching digit is an additional diagnostic.
                same_position = (
                    numeric.allowed(old[len(prefix) :], first=not prefix)
                    if len(prefix) < len(old)
                    else []
                )
                first_bad = dict(
                    digit_offset=len(prefix),
                    prefix=prefix,
                    step=step,
                    greedy_id=winner,
                    greedy_piece=tok.decode([winner]),
                    greedy_logp=float(logp[winner]),
                    best_valid_id=good,
                    best_valid_piece=tok.decode([good]),
                    best_valid_logp=float(logp[good]),
                    valid_rank=int((score > score[good]).sum()) + 1,
                    margin=float(score[winner] - score[good]),
                    old_compatible=bool(winner in old_allowed),
                    old_digit_here=bool(winner in same_position),
                    premature_eos=winner == numeric.eos and len(prefix) < len(answer),
                    missing_eos=len(prefix) == len(answer),
                )
            elif winner != numeric.eos:
                prefix += numeric.pieces[winner]
        if winner == numeric.eos or step == 31:
            break
        out = decode(ctrl, score.new_tensor([[winner]], dtype=torch.long), out.past_key_values, policy)
    ctrl = fork_controller(ctrl)
    out = start
    canonical = []
    for token in [*gold, numeric.eos]:
        score = out.logits[0, -1].float()
        logp = score.log_softmax(-1)
        canonical.append(
            dict(
                id=token,
                piece=tok.decode([token]),
                rank=int((score > score[token]).sum()) + 1,
                logp=float(logp[token]),
                greedy_id=int(score.argmax()),
            )
        )
        if token != numeric.eos:
            out = decode(
                ctrl, score.new_tensor([[token]], dtype=torch.long), out.past_key_values, policy
            )
    return dict(
        greedy=greedy,
        first_bad=first_bad,
        canonical=canonical,
        greedy_mean_logp=(sum(logps) / len(logps)) if logps else None,
    )


@torch.inference_mode()
def audit(model, tok, baseline):
    numeric = NumericTokens(tok, model.config.vocab_size)
    pairs = []
    for case in baseline:
        worlds = []
        for i, answer in enumerate(case["answers"]):
            before, after = case["cells"]["old_before"]
            ids = before + case["fact_ids"][i] + after
            targets = [encode_text(tok, v) for v in [*case["answers"], case["old"]]]
            trace = {}

            def observe(ctrl, out, policy):
                trace.update(traces(ctrl, out, policy, tok, numeric, answer, case["old"], targets[i]))

            result = run_channel(
                model,
                tok,
                dict(case, answer=answer),
                torch.tensor(ids),
                [0],
                torch.tensor([case["release"]]),
                ids,
                "reader",
                2,
                32,
                targets,
                observe,
            )
            prior = case["results"]["old_before"][i]
            if (
                trace["greedy"] != prior["generated_ids"]
                or result["generated_ids"] != prior["generated_ids"]
                or result["candidate_mean_logps"] != prior["candidate_mean_logps"]
            ):
                raise AssertionError("Audit failed to reproduce original decoding")
            if trace["first_bad"] is None and not result["correct"]:
                raise AssertionError("Missing first error on incorrect generation")
            worlds.append(dict(answer=answer, trace=trace, **result))
        pairs.append(
            dict(
                deployment=case["deployment"],
                family=case["family"],
                old=case["old"],
                question=case["question"],
                worlds=worlds,
            )
        )
    return pairs


def summarize(pairs):
    worlds = [(x, i) for p in pairs for i, x in enumerate(p["worlds"])]
    failed = [x["trace"]["first_bad"] for x, i in worlds if not x["correct"]]
    return dict(
        pairs=len(pairs),
        correct_worlds=sum(x["correct"] for x, i in worlds),
        both_correct=sum(all(x["correct"] for x in p["worlds"]) for p in pairs),
        flipped=sum(
            all(
                x["candidate_mean_logps"][i] > x["candidate_mean_logps"][1 - i]
                for i, x in enumerate(p["worlds"])
            )
            for p in pairs
        ),
        truth_ranked_above_both=sum(
            x["candidate_mean_logps"][i]
            > max(x["candidate_mean_logps"][1 - i], x["candidate_mean_logps"][2])
            for x, i in worlds
        ),
        greedy_mean_beats_truth=sum(
            x["trace"]["greedy_mean_logp"] is not None
            and x["trace"]["greedy_mean_logp"] > x["candidate_mean_logps"][i]
            for x, i in worlds
        ),
        first_bad_token_start=dict(Counter(x["digit_offset"] + 1 for x in failed)),
        first_wrong_character=dict(Counter(x["first_mismatch"] for x, _ in worlds if not x["correct"])),
        early_eos=sum(x["premature_eos"] for x in failed),
        after_four_digits=sum(x["missing_eos"] for x in failed),
        follows_old_prefix=sum(x["old_compatible"] for x in failed),
        matches_old_digit_at_error=sum(x["old_digit_here"] for x in failed),
    )


@torch.inference_mode()
def matched_controls(model, tok, baseline):
    """Keep the fact at its original offset; match the count's number to the old value."""
    rows = []
    for case in baseline:
        before, after = case["cells"]["old_before"]
        if len(before) != 16 or after:
            raise ValueError("Expected a 16-token prefix")
        rng = random.Random(f"diagnosis/{case['deployment']}")
        count_record = encode_text(tok, f"{case['entity']}采样数{case['old']}。")
        old_value = encode_text(tok, case["old"] + "。")
        if before[-len(old_value) :] != old_value or count_record[-len(old_value) :] != old_value:
            raise ValueError("Place the identical tokenized value at the same offset")
        neutral = filler(tok, 16, rng)
        count = filler(tok, 16 - len(count_record), rng) + count_record
        worlds = {}
        for name, prefix in (("neutral_before", neutral), ("count_before", count)):
            outcomes = []
            targets = [encode_text(tok, answer) for answer in [*case["answers"], case["old"]]]
            for answer, fact in zip(case["answers"], case["fact_ids"]):
                ids = prefix + fact
                outcomes.append(
                    run_channel(
                        model,
                        tok,
                        dict(case, answer=answer),
                        torch.tensor(ids),
                        [0],
                        torch.tensor([case["release"]]),
                        ids,
                        "reader",
                        2,
                        32,
                        targets,
                    )
                )
            worlds[name] = dict(prefix=prefix, outcomes=outcomes)
        rows.append(dict(deployment=case["deployment"], **worlds))
    summary = {}
    for name in ("neutral_before", "count_before"):
        results = [r[name]["outcomes"] for r in rows]
        summary[name] = dict(
            pairs=sum(all(x["correct"] for x in worlds) for worlds in results),
            worlds=sum(x["correct"] for worlds in results for x in worlds),
            rescued_worlds=sum(
                not old["correct"] and new["correct"]
                for case, worlds in zip(baseline, results)
                for old, new in zip(case["results"]["old_before"], worlds)
            ),
            lost_worlds=sum(
                old["correct"] and not new["correct"]
                for case, worlds in zip(baseline, results)
                for old, new in zip(case["results"]["old_before"], worlds)
            ),
        )
    return dict(summary=summary, samples=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controls", action="store_true")
    args = p.parse_args()
    base = json.loads(args.baseline.read_text())
    model = load_x_for_eval(".", base["checkpoint"])
    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    bind_x_tokenizer(model, tok)
    rows = audit(model, tok, base["after"]["samples"])
    info = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(summary=info, samples=rows)
    if args.controls:
        report["matched_controls"] = matched_controls(model, tok, base["after"]["samples"])
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(info)


if __name__ == "__main__":
    main()
