"""Equal-length distractors at fixed fact and number positions on paired Reader QA."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.retrieval_qa_dataset import token_span
from scripts.eval_fact_swap import encode_text, locate_facts, evidence_blocks
from scripts.eval_paired_reader import run_channel, state_digest
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval

ARMS = ("neutral", "unrelated_count", "entity_count", "obsolete_threshold")
CHANNELS = ("prompt", "reader")


def filler(tok, length, rng):
    phrases = [
        encode_text(tok, s)
        for s in (
            "设备自检通过。",
            "链路心跳正常。",
            "例行遥测正常。",
            "报告已归档。",
            "固件校验通过。",
            "状态稳定。",
        )
    ]
    result = []
    while length:
        options = [p for p in phrases if len(p) <= length]
        ids = rng.choice(options) if options else encode_text(tok, "正常。" if length >= 2 else "\n")
        if not 0 < len(ids) <= length:
            raise ValueError("Tokenizer cannot fill residual token budget")
        result.extend(ids)
        length -= len(ids)
    return result


def layout(tok, sample, reference, bs):
    """Map full-document annotations to positions in the ordered gold-block payload."""
    facts = locate_facts(sample)
    enc = tok(sample["context"], add_special_tokens=False, return_offsets_mapping=True)
    blocks = evidence_blocks(enc.offset_mapping, facts.spans, bs)
    pos = [i for b in blocks for i in range(b * bs, min((b + 1) * bs, len(enc.input_ids)))]
    tokens = [enc.input_ids[i] for i in pos]
    if blocks != reference["block_ids"] or tokens != reference["evidence_ids"]:
        raise ValueError("Source block IDs or evidence tokens differ from prior evaluation")

    def local(span):
        lo, hi = token_span(enc.offset_mapping, span)
        start = pos.index(lo)
        if pos[start : start + hi - lo] != list(range(lo, hi)):
            raise ValueError("Fact is not contiguous in the delivered evidence")
        return start, start + hi - lo

    return sorted(local(s) for s in facts.spans), local(facts.value_span)


def treatments(tok, entity, decoy, rng):
    prefix = [
        "日志：本地设备的采样计数为 ",
        f"日志：{entity} 的采样计数为 ",
        f"日志：{entity} 的旧阈值为 ",
    ]
    generic, named, stale = [encode_text(tok, p) for p in prefix]
    if len(named) != len(stale):
        raise ValueError("Named counter and obsolete threshold need equal-length prefixes")
    width = max(len(generic), len(named))
    generic = filler(tok, width - len(generic), rng) + generic
    pad = filler(tok, width - len(named), rng)
    value = encode_text(tok, decoy)
    missing = next(
        (
            encode_text(tok, s)
            for s in ("未知", "未记录", "尚未记录", "暂未记录")
            if len(encode_text(tok, s)) == len(value)
        ),
        None,
    )
    if missing is None:
        raise ValueError("Missing an equal-width nonnumeric control")
    end = encode_text(tok, "。")
    return dict(
        neutral=generic + missing + end,
        unrelated_count=generic + value + end,
        entity_count=pad + named + value + end,
        obsolete_threshold=pad + stale + value + end,
    ), width


def make_pair(tok, sample, prior, bs):
    """Construct all treatments once per pair, sharing the distractor across both worlds."""
    orig, edit = prior["original_answer"], prior["edited_answer"]
    facts = locate_facts(sample)
    a, b = facts.value_span
    edited = dict(sample, context=sample["context"][:a] + edit + sample["context"][b:], answer=edit)
    original = prior["original"]["block"]
    alternative = prior["edited"]["block"]
    spans, value_span = layout(tok, sample, original, bs)
    if layout(tok, edited, alternative, bs) != (spans, value_span):
        raise ValueError("Edited facts moved")
    lo, hi = value_span
    left, right = original["evidence_ids"], alternative["evidence_ids"]
    if left[:lo] != right[:lo] or left[hi:] != right[hi:] or left[lo:hi] == right[lo:hi]:
        raise ValueError("Reference edit must change only the target value")
    # The entity bearing the value is the linked device on multi-hop questions.
    sentence = sample["context"][slice(*facts.spans[-1])]
    entity = re.search(r"：([^ ：]+) ", sentence).group(1)
    rng = random.Random(f"reader-distractor-v1/{prior['line']}")
    for _ in range(10000):
        decoy = str(rng.randrange(1000, 10000))
        if (
            all(d not in (x, y) for d, x, y in zip(decoy, orig, edit))
            and decoy not in sample["context"]
            and decoy not in sample["question"]
            and len(encode_text(tok, decoy)) == len(encode_text(tok, orig))
        ):
            break
    else:
        raise ValueError("No distinct token-length-matched distractor")
    slots, value_offset = treatments(tok, entity, decoy, rng)
    size = len(slots["neutral"])
    if any(len(v) != size for v in slots.values()):
        raise AssertionError("Unequal treatment lengths")
    gaps = list(zip([0] + [b for _, b in spans], [a for a, _ in spans] + [len(left)]))
    eligible = [(a, b) for a, b in gaps if b - a >= size]
    prefer_before = prior["line"] % 2 == 1
    preferred = [(a, b) for a, b in eligible if (b <= lo) == prefer_before]
    start, end = max(preferred or eligible, key=lambda ab: ab[1] - ab[0])
    insert = (start + end - size) // 2
    background = [0] * len(left)
    for a, b in gaps:
        if a == start and b == end:
            background[a:insert] = filler(tok, insert - a, rng)
            background[insert + size : b] = filler(tok, b - insert - size, rng)
        else:
            background[a:b] = filler(tok, b - a, rng)
    for a, b in spans:
        background[a:b] = left[a:b]
    designs = {}
    for name, slot in slots.items():
        ids = background.copy()
        ids[insert : insert + size] = slot
        alt = ids.copy()
        alt[lo:hi] = right[lo:hi]
        designs[name] = (ids, alt)
    return dict(
        entity=entity,
        decoy=decoy,
        fact_spans=spans,
        target_span=value_span,
        slot_span=[insert, insert + size],
        decoy_start=insert + value_offset,
    ), designs


def summarize(rows):
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for channel in CHANNELS:
            counts = Counter()
            decoy_margins = []
            for row in rows:
                a, b = [row["arms"][arm][w][channel] for w in ("original", "edited")]
                la, lb = a["candidate_mean_logps"], b["candidate_mean_logps"]
                counts.update(
                    pairs=int(a["correct"] and b["correct"]),
                    correct=a["correct"] + b["correct"],
                    flips=int(la[0] > la[1] and lb[1] > lb[0]),
                    changed=int(a["prediction"] != b["prediction"]),
                )
                for idx, x in enumerate((a, b)):
                    lp = x["candidate_mean_logps"]
                    counts["decoy_above_true"] += lp[2] > lp[idx]
                    decoy_margins.append(lp[2] - lp[idx])
                    counts["generated_decoy"] += x["prediction"].strip() == row["decoy"]
                    counts["four_digits"] += bool(re.fullmatch(r"[0-9]{4}", x["prediction"].strip()))
                    counts["at_cap"] += len(x["generated_ids"]) == 32
            out[arm][channel] = dict(
                counts, n=len(rows), mean_decoy_minus_true=sum(decoy_margins) / len(decoy_margins)
            )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    previous = json.loads(args.reference.read_text())
    config = previous["config"]
    if config["depth"] != 2 or config["max_new_tokens"] != 32:
        p.error("Expected the fixed-R2, 32-token reference")
    for key in ("checkpoint", "data"):
        if hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() != previous[key + "_sha256"]:
            raise ValueError(f"Reference {key} checksum differs")
    model = load_x_for_eval(".", config["checkpoint"], device="cpu").eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(config["tokenizer"])
    bind_x_tokenizer(model, tok)
    digest = state_digest(model)
    if digest != previous["model_state_sha256"]:
        raise ValueError("Reference model state differs")
    lines = Path(config["data"]).read_text().splitlines()
    rows = []
    forwards = Counter()

    def check(module, inputs, out):
        forwards[out.num_loops_executed] += 1
        if out.num_loops_executed != 2:
            raise AssertionError("Unexpected executed depth")

    with torch.inference_mode(), model.register_forward_hook(check):
        for prior in previous["samples"]:
            sample = json.loads(lines[prior["line"] - 1])
            info, designs = make_pair(tok, sample, prior, model.config.block_size)
            targets = [
                encode_text(tok, t)
                for t in (prior["original_answer"], prior["edited_answer"], info["decoy"])
            ]
            refs = prior["original"]["block"]
            # Local block numbers preserve source boundaries, partial tail and release times.
            blocks = list(range(len(refs["block_ids"])))
            release = torch.tensor(refs["release_rounds"])
            arms = {}
            for arm, pair in designs.items():
                arms[arm] = {}
                for idx, w in enumerate(("original", "edited")):
                    world = dict(sample, answer=prior[w + "_answer"])
                    ids = pair[idx]
                    source = torch.tensor(ids)
                    arms[arm][w] = dict(evidence_ids=ids, evidence_text=tok.decode(ids))
                    for c in CHANNELS:
                        outcome = run_channel(
                            model, tok, world, source, blocks, release, ids, c, 2, 32, targets
                        )
                        if (
                            outcome["release_after_rounds"]
                            != prior[w]["block"][c]["release_after_rounds"]
                        ):
                            raise AssertionError("Release protocol differs")
                        arms[arm][w][c] = outcome
            rows.append(
                dict(
                    **{
                        k: prior[k]
                        for k in (
                            "line",
                            "family",
                            "question",
                            "original_answer",
                            "edited_answer",
                            "extrapolated",
                        )
                    },
                    **info,
                    source_blocks=refs["block_ids"],
                    release_rounds=release.tolist(),
                    arms=arms,
                )
            )
    if state_digest(model) != digest:
        raise AssertionError("Evaluation changed model state")
    groups = {
        "all": rows,
        "immediate": [r for r in rows if r["family"] != "new_evidence"],
        "delayed": [r for r in rows if r["family"] == "new_evidence"],
        "native": [r for r in rows if not r["extrapolated"]],
        "extended": [r for r in rows if r["extrapolated"]],
    }
    result = dict(
        reference=str(args.reference),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        checkpoint_sha256=previous["checkpoint_sha256"],
        data_sha256=previous["data_sha256"],
        model_state_sha256=digest,
        depth_histogram=dict(forwards),
        max_new_tokens=32,
        summary={k: summarize(v) for k, v in groups.items()},
        samples=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["summary"]["all"], indent=2))


if __name__ == "__main__":
    main()
