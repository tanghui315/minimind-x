"""Train the Reader against incompatible next-token winners with stale thresholds in front."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_old_threshold_reader import NumericTokens, audit, summarize
from scripts.eval_fact_swap import FAMILIES, encode_text
from trainer.train_numeric_reader import freeze_for_reader
from trainer.train_short_reader import (
    background,
    batch_scores,
    encode_batch,
    eval_examples,
    evaluate,
    make_cases,
    sha,
    split_data,
    write_json,
)
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval, save_x_checkpoint

MODES = ("old_before", "clean", "neutral_16")


def allowed_prefixes(numeric, target):
    """Exclude EOS from the margin and accept any strictly correct next digit piece."""
    pieces = [numeric.pieces[t] for t in target[:-1]]
    if target[-1] != numeric.eos or len("".join(pieces)) != 4:
        raise ValueError("Expected a four-digit target with EOS")
    return [numeric.allowed("".join(pieces[i:]), first=i == 0) for i in range(len(pieces))] + [[]]


def prefix_hinge(logits, compatible, weight_rows, margin=1.0):
    """Compute a margin against every incompatible next token in the model vocab."""
    if not compatible:
        return logits.new_zeros(()), 0
    idx = torch.tensor(weight_rows)
    values = logits.index_select(0, idx).float()
    cols = pad_sequence([torch.tensor(ids) for ids in compatible], batch_first=True, padding_value=-1)
    cols = torch.where(cols < 0, cols[:, :1], cols)
    mask = torch.zeros_like(values, dtype=torch.bool).scatter_(1, cols, True)
    good = values.gather(1, cols).amax(1)
    bad = values.masked_fill(mask, -torch.inf).amax(1)
    return (margin + bad - good).relu().sum(), len(idx)


def objective(model, batch, past, weight, memory_width=None):
    ce, hinge, positions, supervised = 0.0, 0.0, 0, 0
    for late, rows in batch.items():
        if not rows:
            continue
        encoded, allowances = zip(*rows)
        logits, targets = batch_scores(model, encoded, late, past, memory_width=memory_width)
        ce = ce + F.cross_entropy(logits, targets, reduction="sum")
        valid_rows = [
            i for i, ids in enumerate(token_ids for ex in allowances for token_ids in ex) if ids
        ]
        compatible = [ids for ex in allowances for ids in ex if ids]
        if weight:
            penalty, count = prefix_hinge(logits, compatible, valid_rows)
            hinge = hinge + penalty
            positions += count
        supervised += targets.numel()
    return ce / supervised, weight * hinge / max(1, positions)


def prepare_splits(ancestor, initial_reader, distractors, previous, seed):
    first, historical = split_data(ancestor, initial_reader, distractors, previous["config"]["seed"])
    heldout_old = {v for split in ("dev", "test") for v in previous["splits"][split]["values"]}
    seen = historical | heldout_old | set(previous["splits"]["train"]["values"])
    fresh = [str(i) for i in range(1000, 10000) if str(i) not in seen]
    rng = random.Random(seed)
    rng.shuffle(fresh)
    if len(fresh) < 350:
        raise ValueError("Too few unseen four-digit values for the new splits")
    withheld = heldout_old | set(fresh)
    sites = list(range(500, 680))
    rng.shuffle(sites)
    return dict(
        train=dict(
            sites=first["train"]["sites"],
            values=[v for v in first["train"]["values"] if v not in withheld],
        ),
        dev=dict(sites=sites[:48], values=fresh[:120]),
        test=dict(sites=sites[48:], values=fresh[120:]),
    ), seen


def make_training(tok, rng, cases):
    for i, case in enumerate(cases):
        arm = i // len(FAMILIES) % 8
        if arm < 6:
            mode, pos, length = "old", "head", rng.randint(len(case["old_ids"]), 16)
        elif arm == 6:
            mode, pos, length = "clean", "head", 0
        else:
            mode, pos, length = "neutral", ("head", "middle", "tail")[i // 24 % 3], 16
        before, after = background(tok, rng, case, mode, pos, length)
        yield dict(case, mode=mode, before=before, after=after)


def rank(row):
    return (
        row["old_before"]["pairs"],
        row["clean"]["pairs"] + row["neutral_16"]["pairs"],
        row["old_before"]["correct"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "extraction", "initial-reader", "distractors", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--steps", type=int, default=6500)
    p.add_argument("--check-every", type=int, default=500)
    p.add_argument("--batch-pairs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--margin-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=261126)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "training.json").exists():
        p.error("Use a new training output directory")
    old_tr = json.loads((args.previous / "training.json").read_text())
    old_results = json.loads((args.previous / "results.json").read_text())
    parent = old_results["checkpoint"]
    if sha(parent) != old_results["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint changed")
    source = [
        json.loads(path.read_text()) for path in (args.extraction, args.initial_reader, args.distractors)
    ]
    splits, seen = prepare_splits(*source, old_tr, args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    model = load_x_for_eval(".", parent)
    bind_x_tokenizer(model, tok)
    reader = freeze_for_reader(model)
    model.checkpoint_parent = dict(path=str(Path(parent).resolve()), id=model.checkpoint_id)
    numeric = NumericTokens(tok, model.config.vocab_size)
    cases = {}
    for i, (key, space) in enumerate(splits.items()):
        split_rng = random.Random(args.seed + i + 1)
        count = {"train": 6144, "dev": 36, "test": 96}[key]
        gen = make_cases(tok, split_rng, **space, count=count)
        cases[key] = list((make_training if key == "train" else eval_examples)(tok, split_rng, gen))
    catalogs = {
        key: dict(
            sites=sorted({r["deployment"] for r in rows}),
            values=sorted({v for r in rows for v in [*r["answers"], r["old"]]}),
        )
        for key, rows in cases.items()
    }
    for key in ("dev", "test"):
        if set(catalogs[key]["values"]) & seen:
            raise ValueError("Held-out values occur in an ancestor")
        if set(catalogs[key]["sites"]) & {
            s for split in old_tr["splits"].values() for s in split["sites"]
        }:
            raise ValueError("Held-out site occurs in an ancestor")
        write_json(args.out / (key + "_cases.json"), cases[key])
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if any(set(catalogs[a][k]) & set(catalogs[b][k]) for k in ("sites", "values")):
            raise ValueError("Split overlap")
    report = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        parent=model.checkpoint_parent,
        parent_sha256=sha(parent),
        splits=catalogs,
        seen_values=len(seen),
        train_layouts=dict(Counter(r["mode"] for r in cases["train"])),
        history=[],
        selected_step=0,
        trainable_parameters=sum(p.numel() for p in reader.parameters()),
    )
    write_json(args.out / "training.json", report)
    rows = []
    for start in range(0, len(cases["train"]), 128):
        chunk = cases["train"][start : start + 128]
        enc = encode_batch(model, tok, chunk)
        for i, row in enumerate(chunk):
            pair = enc[i * 2 : i * 2 + 2]
            rows.append(
                (
                    bool(row["release"]),
                    [
                        (
                            x,
                            allowed_prefixes(numeric, x[0][-1])
                            if row["mode"] == "old"
                            else [[] for _ in x[0][-1]],
                        )
                        for x in pair
                    ],
                )
            )
    header = torch.tensor([[tok.bos_token_id, *encode_text(tok, "资料：")]])
    with torch.no_grad():
        past = model(header, use_cache=True, num_loops=2, collect_routers=False).past_key_values
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=0)
    metrics, _ = evaluate(model, tok, cases["dev"], modes=MODES)
    report["history"].append(dict(step=0, metrics=metrics))
    print(report["history"][-1], flush=True)
    selected = parent
    best = rank(metrics)
    running_ce = running_hinge = 0.0
    for step in range(1, args.steps + 1):
        batch = {False: [], True: []}
        for late, pair in rng.choices(rows, k=args.batch_pairs):
            batch[late].extend(pair)
        opt.zero_grad(set_to_none=True)
        ce, penalty = objective(model, batch, past, args.margin_weight)
        (ce + penalty).backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        opt.step()
        running_ce += float(ce.detach())
        running_hinge += float(torch.as_tensor(penalty).detach())
        if step % 100 == 0:
            print(
                f"step={step} ce={running_ce / 100:.5f} weighted_margin={running_hinge / 100:.5f}",
                flush=True,
            )
            running_ce = running_hinge = 0.0
        if step % args.check_every == 0 or step == args.steps:
            with torch.no_grad():
                original = [p.clone() for p in reader.parameters()]
                for p in reader.parameters():
                    p.copy_(p.half())
            metrics, _ = evaluate(model, tok, cases["dev"], modes=MODES)
            report["history"].append(dict(step=step, metrics=metrics))
            print(report["history"][-1], flush=True)
            if rank(metrics) > best:
                best = rank(metrics)
                selected = save_x_checkpoint(
                    model,
                    args.out,
                    "old_before_reader",
                    model.config.hidden_size,
                    extra=dict(training_step=step, dev=metrics),
                )
                report["selected_step"] = step
            with torch.no_grad():
                for p, x in zip(reader.parameters(), original):
                    p.copy_(x)
            report["selected_checkpoint"] = selected
            write_json(args.out / "training.json", report)
    a, b = [load_x_for_eval(".", w) for w in (parent, selected)]
    for net in (a, b):
        bind_x_tokenizer(net, tok)
    aa, bb = a.state_dict(), b.state_dict()
    changed = [k for k in aa if not torch.equal(aa[k], bb[k])]
    if any(not k.startswith("model.thinking_core.memory_reader.") for k in changed):
        raise AssertionError("Frozen state changed")
    dev, _ = evaluate(b, tok, cases["dev"], modes=MODES)
    if dev != next(h["metrics"] for h in report["history"] if h["step"] == report["selected_step"]):
        raise AssertionError("Saved checkpoint differs from selected development results")
    out = dict(
        checkpoint=selected,
        checkpoint_sha256=sha(selected),
        changed_tensors=changed,
        unchanged_tensors=len(aa) - len(changed),
        reloaded_dev=dev,
        test_cases_sha256=sha(args.out / "test_cases.json"),
    )
    prior = old_results["after"]["samples"]
    for label, net in (("before", a), ("after", b)):
        summary, items = evaluate(net, tok, cases["test"], True, modes=MODES)
        replay, replay_rows = evaluate(net, tok, prior, True, modes=MODES)
        if label == "before":
            for old, new in zip(prior, replay_rows, strict=True):
                for mode in MODES:
                    if old["results"][mode] != new["results"][mode]:
                        raise AssertionError("Earlier baseline failed to replay")
        new_audit = audit(net, tok, items)
        trace_summary = summarize(new_audit)
        write_json(
            args.out / (label + "_new_audit.json"), dict(summary=trace_summary, samples=new_audit)
        )
        out[label] = dict(
            summary=summary,
            samples=items,
            old_benchmark=replay,
            old_benchmark_samples=replay_rows,
            new_audit=trace_summary,
        )
    write_json(args.out / "results.json", out)
    print({k: out[k]["summary"] for k in ("before", "after")}, flush=True)


if __name__ == "__main__":
    main()
