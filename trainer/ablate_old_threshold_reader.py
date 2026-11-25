"""Factorial ablation of old-threshold sample share and next-token margin loss."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_old_threshold_reader import NumericTokens
from scripts.eval_fact_swap import encode_text
from trainer.train_numeric_reader import freeze_for_reader
from trainer.train_old_threshold_reader import MODES, allowed_prefixes, objective, prepare_splits, rank
from trainer.train_short_reader import (
    background,
    encode_batch,
    eval_examples,
    evaluate,
    make_cases,
    sha,
    write_json,
)
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval, save_x_checkpoint

ARMS = {
    "low_ce": ("low", 0.0),
    "low_margin": ("low", 0.5),
    "high_ce": ("high", 0.0),
    "high_margin": ("high", 0.5),
}


def read(path):
    return json.loads(Path(path).read_text())


def assignment(rng, cases):
    """Block-randomize nested layouts separately within each question family."""
    slots = [(True, "old")] * 6 + [(True, "clean")] * 9 + [(True, "neutral")] * 9
    # The same 6 old cases and the two 4-case background subsets survive both ratios.
    slots += [(False, "clean")] * 4 + [(False, "neutral")] * 4
    schedules = {}
    layouts = []
    for case in cases:
        family = case["family"]
        if not schedules.get(family):
            schedules[family] = slots.copy()
            rng.shuffle(schedules[family])
        in_high, low = schedules[family].pop()
        layouts.append(dict(case, low=low, high="old" if in_high else low))
    return layouts


def intermediate_shares(cases, seed):
    """Add nested old-before layouts between the two existing 32-case allocations."""
    by_family = defaultdict(list)
    for idx, case in enumerate(cases):
        by_family[case["family"]].append(idx)
    output = [dict(case) for case in cases]
    for family, indices in sorted(by_family.items()):
        if len(indices) % 32:
            raise ValueError("Expected complete 32-case family blocks")
        for block_start in range(0, len(indices), 32):
            block = indices[block_start : block_start + 32]
            rng = random.Random(f"share-sweep/{seed}/{family}/{block_start // 32}")
            promotion_rank = {}
            for baseline_mode in ("clean", "neutral"):
                eligible = [
                    i for i in block if cases[i]["low"] == baseline_mode and cases[i]["high"] == "old"
                ]
                if len(eligible) != 9:
                    raise ValueError("Reference block must have nine promotable cases per baseline")
                rng.shuffle(eligible)
                promotion_rank.update({i: rank for rank, i in enumerate(eligible)})
            for i in block:
                for key, count in (("share_375", 3), ("share_500", 5), ("share_625", 7)):
                    output[i][key] = "old" if promotion_rank.get(i, 9) < count else cases[i]["low"]
    return output


def layouts_for(tok, cases, ratio, seed):
    for idx, case in enumerate(cases):
        mode = case[ratio]
        # For a pair that shares a layout, neither the ratio nor other rows alter its tokens.
        rng = random.Random(f"ablation/{seed}/{idx}")
        position, length = "head", 0
        if mode == "old":
            length = rng.randint(len(case["old_ids"]), 16)
        elif mode == "neutral":
            position, length = rng.choice(("head", "middle", "tail")), 16
        before, after = background(tok, rng, case, mode, position, length)
        yield dict(case, mode=mode, before=before, after=after)


def prepare(args):
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    if (root / "design.json").exists():
        raise ValueError("Use a new experiment directory")
    baseline = read(args.previous / "results.json")
    parent = Path(baseline["checkpoint"]).resolve()
    if sha(parent) != baseline["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint differs")
    history = [read(path) for path in (args.extraction, args.initial_reader, args.distractors)]
    previous = read(args.previous / "training.json")
    targeted = read(args.targeted / "training.json")
    pool, seen = prepare_splits(*history, previous, args.seed)
    seen.update(v for s in targeted["splits"].values() for v in s["values"])
    rng = random.Random(args.seed)
    new_values = [str(v) for v in range(1000, 10000) if str(v) not in seen]
    rng.shuffle(new_values)
    if len(new_values) < 160:
        raise ValueError("Insufficient new four-digit values")
    sites = list(range(700, 880))
    rng.shuffle(sites)
    pool["dev"] = dict(sites=sites[:48], values=new_values[:72])
    pool["test"] = dict(sites=sites[48:], values=new_values[72:])
    past_eval = {
        v
        for source in (previous, targeted)
        for key in ("dev", "test")
        for v in source["splits"][key]["values"]
    }
    pool["train"]["values"] = [
        v for v in pool["train"]["values"] if v not in set(new_values) | past_eval
    ]
    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    catalogs, digests, counts = {}, {}, {}
    for key, n in (("train", 6144), ("dev", 36), ("test", 96)):
        split_rng = random.Random(f"{args.seed}/{key}")
        cases = list(make_cases(tok, split_rng, **pool[key], count=n))
        cases = (
            assignment(split_rng, cases)
            if key == "train"
            else list(eval_examples(tok, split_rng, cases))
        )
        counts[key] = len(cases)
        catalogs[key] = dict(
            sites=sorted({r["deployment"] for r in cases}),
            values=sorted({v for r in cases for v in [*r["answers"], r["old"]]}),
        )
        if key == "train":
            max_width = max(len(ids) for r in cases for ids in r["fact_ids"]) + 16
            layout_counts = {ratio: dict(Counter(x[ratio] for x in cases)) for ratio in ("low", "high")}
            for ratio, expected in (
                ("low", {"old": 1152, "clean": 2496, "neutral": 2496}),
                ("high", {"old": 4608, "clean": 768, "neutral": 768}),
            ):
                if layout_counts[ratio] != expected:
                    raise AssertionError("Unexpected treatment allocation")
        elif set(catalogs[key]["values"]) & seen or len(catalogs[key]["sites"]) != n:
            raise AssertionError("Held-out case is not from a new site/value pool")
        path = root / (key + "_cases.json")
        write_json(path, cases)
        digests[key] = sha(path)
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if any(set(catalogs[a][k]) & set(catalogs[b][k]) for k in ("sites", "values")):
            raise AssertionError("Split overlap")
    budget = dict(
        steps=args.steps,
        batch_pairs=args.batch_pairs,
        check_every=args.check_every,
        learning_rate=0.0015,
        depth=2,
        memory_width=max_width,
        seed=args.seed,
    )
    design = dict(
        parent=str(parent),
        parent_sha256=sha(parent),
        budget=budget,
        groups=ARMS,
        layouts=layout_counts,
        case_counts=counts,
        case_sha256=digests,
        split=catalogs,
        excluded_historical_values=len(seen),
    )
    write_json(root / "design.json", design)
    print(counts, layout_counts, budget, flush=True)


def dataset(root, design, split):
    path = root / (split + "_cases.json")
    if sha(path) != design["case_sha256"][split]:
        raise ValueError(f"{split} data differ from pre-registered design")
    return read(path)


def model_for(tok, path):
    model = load_x_for_eval(".", str(path))
    bind_x_tokenizer(model, tok)
    return model


def train(args):
    root = args.out
    design = read(root / "design.json")
    budget = design["budget"]
    ratio, weight = design["groups"][args.arm]
    parent = design["parent"]
    if sha(parent) != design["parent_sha256"]:
        raise ValueError("Parent checkpoint changed")
    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    model = model_for(tok, parent)
    reader = freeze_for_reader(model)
    model.checkpoint_parent = dict(path=parent, id=model.checkpoint_id)
    torch.manual_seed(budget["seed"])
    rng = random.Random(budget["seed"])
    data_dir = root / args.arm
    data_dir.mkdir(exist_ok=True)
    if (data_dir / "training.json").exists():
        raise ValueError("Arm output exists")
    layout_seed = budget.get("layout_seed", budget["seed"])
    train_cases = list(layouts_for(tok, dataset(root, design, "train"), ratio, layout_seed))
    layouts = [[case["mode"], case["before"], case["after"]] for case in train_cases]
    layout_sha256 = hashlib.sha256(json.dumps(layouts).encode()).hexdigest()
    dev = dataset(root, design, "dev")
    numeric = NumericTokens(tok, model.config.vocab_size) if weight else None
    training = []
    for start in range(0, len(train_cases), 128):
        cases = train_cases[start : start + 128]
        encoded = encode_batch(model, tok, cases)
        for idx, case in enumerate(cases):
            worlds = []
            for example in encoded[2 * idx : 2 * idx + 2]:
                target = example[0][-1]
                allowed = (
                    allowed_prefixes(numeric, target)
                    if weight and case["mode"] == "old"
                    else [[]] * len(target)
                )
                worlds.append((example, allowed))
            training.append((bool(case["release"]), case["mode"], worlds))
    header = torch.tensor([[tok.bos_token_id, *encode_text(tok, "资料：")]])
    with torch.no_grad():
        past = model(header, use_cache=True, num_loops=2, collect_routers=False).past_key_values
    optimizer = torch.optim.AdamW(reader.parameters(), lr=budget["learning_rate"], weight_decay=0)
    baseline, _ = evaluate(model, tok, dev, modes=MODES)
    report = dict(
        arm=args.arm,
        ratio=ratio,
        margin_weight=weight,
        budget=budget,
        layout_sha256=layout_sha256,
        parent_sha256=sha(parent),
        dev_sha256=design["case_sha256"]["dev"],
        train_sha256=design["case_sha256"]["train"],
        history=[dict(step=0, metrics=baseline)],
        selected_step=0,
        selected_checkpoint=parent,
    )
    best = rank(baseline)
    samples = hashlib.sha256()
    seen = Counter()
    ce_running = hinge_running = 0.0
    for step in range(1, budget["steps"] + 1):
        indices = rng.choices(range(len(training)), k=budget["batch_pairs"])
        samples.update(json.dumps(indices).encode())
        batch = {False: [], True: []}
        for idx in indices:
            late, mode, pair = training[idx]
            seen[mode] += 1
            batch[late].extend(pair)
        optimizer.zero_grad(set_to_none=True)
        ce, hinge = objective(model, batch, past, weight, memory_width=budget["memory_width"])
        (ce + hinge).backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        optimizer.step()
        ce_running += float(ce.detach())
        hinge_running += float(torch.as_tensor(hinge).detach())
        if step % 250 == 0:
            print(step, f"ce={ce_running / 250:.4f} margin={hinge_running / 250:.4f}", flush=True)
            ce_running = hinge_running = 0.0
        if step % budget["check_every"] and step != budget["steps"]:
            continue
        with torch.no_grad():
            original = [p.clone() for p in reader.parameters()]
            for p in reader.parameters():
                p.copy_(p.half())
        metrics, _ = evaluate(model, tok, dev, modes=MODES)
        report["history"].append(dict(step=step, metrics=metrics))
        print(report["history"][-1], flush=True)
        if rank(metrics) > best:
            best = rank(metrics)
            report["selected_step"] = step
            report["selected_checkpoint"] = save_x_checkpoint(
                model,
                data_dir,
                "reader_selected",
                model.config.hidden_size,
                extra=dict(training_step=step, dev=metrics),
            )
        if step == budget["steps"]:
            report["final_checkpoint"] = save_x_checkpoint(
                model,
                data_dir,
                "reader_final",
                model.config.hidden_size,
                extra=dict(training_step=step, dev=metrics),
            )
        with torch.no_grad():
            for p, old in zip(reader.parameters(), original):
                p.copy_(old)
        report.update(
            executed_steps=step, batch_schedule_sha256=samples.hexdigest(), sampled_pairs=dict(seen)
        )
        write_json(data_dir / "training.json", report)


def selected_runs(root, design):
    reports = {arm: read(root / arm / "training.json") for arm in design["groups"]}
    schedules = {r["batch_schedule_sha256"] for r in reports.values()}
    if len(schedules) != 1 or any(
        r["executed_steps"] != design["budget"]["steps"] for r in reports.values()
    ):
        raise ValueError("Training budget and shared sampling schedule are not complete")
    return reports


def score_arm(args):
    root = args.out
    design = read(root / "design.json")
    reports = selected_runs(root, design)
    test = dataset(root, design, "test")
    dev = dataset(root, design, "dev")
    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    parent = model_for(tok, design["parent"])
    parent_state = parent.state_dict()
    if args.arm == "parent":
        paths = {"parent": design["parent"]}
        data_dir = root
    else:
        report = reports[args.arm]
        paths = {label: report[label + "_checkpoint"] for label in ("selected", "final")}
        data_dir = root / args.arm
    results = {}
    for label, path in paths.items():
        model = parent if label == "parent" else model_for(tok, path)
        updated = [k for k, w in model.state_dict().items() if not torch.equal(w, parent_state[k])]
        if any(not k.startswith("model.thinking_core.memory_reader.") for k in updated):
            raise AssertionError("Frozen weights changed")
        if label != "parent":
            step = report["selected_step"] if label == "selected" else report["executed_steps"]
            saved_dev, _ = evaluate(model, tok, dev, modes=MODES)
            if saved_dev != next(x["metrics"] for x in report["history"] if x["step"] == step):
                raise AssertionError("Reloaded development metric differs")
        summary, rows = evaluate(model, tok, test, with_outputs=True, modes=MODES)
        results[label] = dict(
            checkpoint=path,
            checkpoint_sha256=sha(path),
            summary=summary,
            samples=rows,
            updated_tensors=updated,
        )
        print(label, summary, flush=True)
    write_json(data_dir / "results.json", results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "train", "evaluate"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm")
    for name in ("previous", "targeted", "extraction", "initial-reader", "distractors"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--seed", type=int, default=261127)
    p.add_argument("--steps", type=int, default=6500)
    p.add_argument("--batch-pairs", type=int, default=12)
    p.add_argument("--check-every", type=int, default=500)
    args = p.parse_args()
    if args.action == "prepare":
        if not all(
            (args.previous, args.targeted, args.extraction, args.initial_reader, args.distractors)
        ):
            p.error("prepare requires the previous runs and ancestry records")
        if min(args.steps, args.batch_pairs, args.check_every) < 1:
            p.error("Training sizes must be positive")
        prepare(args)
    else:
        arms = set(read(args.out / "design.json")["groups"])
        if args.action == "evaluate":
            arms.add("parent")
        if args.arm not in arms:
            p.error(f"Select one of the configured groups: {', '.join(sorted(arms))}")
        if args.action == "train":
            train(args)
        else:
            score_arm(args)


if __name__ == "__main__":
    main()
