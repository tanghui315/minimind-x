"""Train only the Reader on paired short evidence and evaluate using reserved sites and values."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.data_gen.gen_evidence_families import DEVICES, gen_sample
from scripts.eval_fact_swap import FAMILIES, encode_text, locate_facts, support_sentences
from scripts.eval_paired_reader import run_channel
from scripts.eval_reader_distractors import filler
from trainer.train_numeric_extraction import answer_logits, paired_fact_samples, split_values
from trainer.train_numeric_reader import freeze_for_reader
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval, save_x_checkpoint

MODES = ("clean", "neutral_02", "neutral_08", "neutral_16", "old_before", "old_after")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_data(extraction, reader, distractors, seed):
    reserved = set(extraction["reserved_values"])
    previous_eval = reserved | {r["decoy"] for r in distractors["samples"]}
    seen = previous_eval.copy()
    train_pool, dev_pool = split_values(random.Random(extraction["config"]["seed"]), reserved)
    for run in (extraction, reader):
        rng = random.Random(run["config"]["seed"])
        train = {
            s["answer"]
            for pair in paired_fact_samples(rng, run["train_pairs"], run["train_sites"], train_pool)
            for s in pair
        }
        dev = {
            s["answer"]
            for pair in paired_fact_samples(rng, extraction["dev_pairs"], run["dev_sites"], dev_pool)
            for s in pair
        }
        if run is reader and (len(train), len(dev)) != (
            reader["unique_train_values"],
            reader["unique_dev_values"],
        ):
            raise ValueError("Ancestor value pools could not be reproduced")
        seen.update(train | dev)
    fresh = [str(v) for v in range(1000, 10000) if str(v) not in seen]
    rng = random.Random(seed)
    rng.shuffle(fresh)
    if len(fresh) < 640:
        raise ValueError("Too few values outside ancestor target pools")
    withheld = set(fresh[:640]) | previous_eval
    sites = list(range(300, 480))
    rng.shuffle(sites)
    return {
        "train": dict(
            sites=extraction["train_sites"], values=[v for v in train_pool if v not in withheld]
        ),
        "dev": dict(sites=sites[:48], values=fresh[:240]),
        "test": dict(sites=sites[48:], values=fresh[240:640]),
    }, seen


def make_cases(tok, rng, sites, values, count):
    bins = {}
    for value in values:
        bins.setdefault(len(encode_text(tok, value)), []).append(value)
    eligible = [v for group in bins.values() if len(group) >= 3 for v in group]
    sites = list(sites)
    rng.shuffle(sites)
    for idx in range(count):
        family = FAMILIES[idx % len(FAMILIES)]
        site = sites[idx % len(sites)]
        sample = gen_sample(rng, family, [f"{base}-{site:03d}" for base in DEVICES])
        sample["family"] = family
        facts = locate_facts(sample)
        passage = support_sentences(sample, facts)
        entity = re.search(r"：([^ ：]+) ", sample["context"][slice(*facts.spans[-1])]).group(1)
        start, end = facts.value_span
        # Replace only the target span in its last supporting sentence.
        suffix_len = facts.spans[-1][1] - end
        offset = len(passage) - suffix_len - (end - start)
        match = passage[offset : offset + 4]
        if match != sample["answer"]:
            raise AssertionError("Target span in the support passage differs")
        for _ in range(512):
            first = rng.choice(eligible)
            second, old = rng.sample(bins[len(encode_text(tok, first))], 2)
            if any(a == b or a == c or b == c for a, b, c in zip(first, second, old)):
                continue
            if any(
                v in sample["question"] or v in passage[:offset] + passage[offset + 4 :]
                for v in (first, second, old)
            ):
                continue
            evidence = [
                encode_text(tok, passage[:offset] + v + passage[offset + 4 :]) for v in (first, second)
            ]
            old_ids = encode_text(tok, f"{entity}旧阈值{old}。")
            if len(old_ids) <= 16 and len(evidence[0]) == len(evidence[1]):
                break
        else:
            raise ValueError("Cannot construct matched numerical pair within the background budget")
        yield dict(
            family=family,
            deployment=site,
            question=sample["question"],
            answers=[first, second],
            fact_ids=evidence,
            old=old,
            old_ids=old_ids,
            entity=entity,
            release=sample.get("evidence_visible_from_round", 0),
        )


def background(tok, rng, case, mode, position, length):
    if mode == "clean":
        return [], []
    if mode == "old":
        distractor = case["old_ids"]
        extra = filler(tok, length - len(distractor), rng)
        return (extra + distractor, []) if position == "head" else ([], distractor + extra)
    before = {"head": length, "middle": length // 2, "tail": 0}[position]
    return filler(tok, before, rng), filler(tok, length - before, rng)


def train_examples(tok, rng, cases):
    for idx, case in enumerate(cases):
        slot = idx // len(FAMILIES) % 8
        if slot < 2:
            mode, position, length = "clean", "head", 0
        elif slot < 5:
            mode, position, length = "neutral", ("head", "middle", "tail")[slot - 2], rng.randint(1, 16)
        else:
            mode, position = "old", ("head", "tail", "head" if idx % 2 else "tail")[slot - 5]
            length = rng.randint(len(case["old_ids"]), 16)
        before, after = background(tok, rng, case, mode, position, length)
        yield dict(case, mode=mode, position=position, before=before, after=after)


def eval_examples(tok, rng, cases):
    for idx, case in enumerate(cases):
        cells = {}
        for mode in MODES:
            if mode.startswith("neutral"):
                pos = ("head", "middle", "tail")[idx // len(FAMILIES) % 3]
                kind, length = "neutral", int(mode[-2:])
            elif mode.startswith("old"):
                pos = "head" if mode.endswith("before") else "tail"
                kind, length = "old", 16
            else:
                pos, kind, length = "head", "clean", 0
            cells[mode] = background(tok, rng, case, kind, pos, length)
        yield dict(case, cells=cells)


def encode_batch(model, tok, rows):
    examples, ids = [], []
    header = [tok.bos_token_id, *encode_text(tok, "资料：")]
    delimiter = encode_text(tok, "答案：")
    for row in rows:
        question = encode_text(tok, "\n问题：") + encode_text(tok, row["question"])
        for answer, fact in zip(row["answers"], row["fact_ids"]):
            evidence = row["before"] + fact + row["after"]
            if len(evidence) > model.config.block_size:
                raise ValueError("Training evidence exceeds a single Reader block")
            ids.append(torch.tensor(evidence))
            prefix = question if row["release"] else header + question
            examples.append((prefix, delimiter, encode_text(tok, answer) + [tok.eos_token_id]))
    hidden = model.model.prelude_hidden(pad_sequence(ids, batch_first=True))
    return [(ex, h[: len(m)].clone()) for ex, h, m in zip(examples, hidden, ids)]


def delayed_logits(model, examples, payload, past):
    # Each sample uses the same separately cached evidence-free header.
    batch = len(examples)

    def expand(entry):
        if isinstance(entry, list):
            return [expand(e) for e in entry]
        return tuple(t.expand(batch, *t.shape[1:]) for t in entry)

    inputs, positions, targets = [], [], []
    for idx, (question, delimiter, answer) in enumerate(examples):
        prefix = question + delimiter
        inputs.append(torch.tensor(prefix + answer[:-1]))
        positions.extend((idx, len(prefix) - 1 + i) for i in range(len(answer)))
        targets.extend(answer)
    ids = pad_sequence(inputs, batch_first=True)
    out = model.model(
        ids,
        past_key_values=[expand(entry) for entry in past],
        num_loops=2,
        collect_routers=False,
        memory_payload=payload,
    )
    rows, columns = torch.tensor(positions).T
    return model.lm_head(out["hidden_states"][rows, columns]), torch.tensor(targets)


def batch_scores(model, selected, late, past, memory_width=None):
    examples, memory = zip(*selected)
    rows = pad_sequence(memory, batch_first=True)
    if memory_width is not None:
        if memory_width < rows.size(1):
            raise ValueError("Fixed memory width cannot truncate evidence")
        rows = torch.nn.functional.pad(rows, (0, 0, 0, memory_width - rows.size(1)))
    valid = torch.arange(rows.size(1))[None, :] < torch.tensor([len(m) for m in memory])[:, None]
    payload = (rows, valid)
    if late:
        return delayed_logits(model, examples, payload, past)
    return answer_logits(model, examples, "cpu", 2, memory_payload=payload)


def train_loss(model, batch, past):
    logits, targets = zip(
        *(batch_scores(model, selected, late, past) for late, selected in batch.items() if selected)
    )
    return torch.nn.functional.cross_entropy(torch.cat(logits), torch.cat(targets))


@torch.inference_mode()
def evaluate(model, tok, cases, with_outputs=False, modes=MODES):
    totals = {mode: Counter() for mode in modes}
    outputs = []
    for case in cases:
        results = {}
        candidates = (
            [encode_text(tok, v) for v in [*case["answers"], case["old"]]] if with_outputs else ()
        )
        for mode in modes:
            before, after = case["cells"][mode]
            worlds = []
            for answer, fact in zip(case["answers"], case["fact_ids"]):
                ids = before + fact + after
                x = run_channel(
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
                    candidates,
                )
                worlds.append(x)
            hits = sum(x["correct"] for x in worlds)
            totals[mode].update(pairs=int(hits == 2), correct=hits, n=1)
            if with_outputs:
                results[mode] = worlds
        if with_outputs:
            outputs.append(dict(case, results=results))
    return {mode: dict(counts) for mode, counts in totals.items()}, outputs


def rank(metrics):
    aug = [m for mode, m in metrics.items() if mode != "clean"]
    return sum(m["pairs"] for m in aug), metrics["clean"]["pairs"], sum(m["correct"] for m in aug)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "extraction", "reader", "distractors", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--steps", type=int, default=6500)
    p.add_argument("--check-every", type=int, default=500)
    p.add_argument("--batch-pairs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=261125)
    args = p.parse_args()
    if min(args.steps, args.check_every, args.batch_pairs) < 1 or args.lr <= 0:
        p.error("Positive training sizes and learning rate required")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Use a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    exclusion = json.loads(args.extraction.read_text())
    parent_run = json.loads(args.reader.read_text())
    dist = json.loads(args.distractors.read_text())
    if sha(args.checkpoint) != dist["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the prior experiment")
    splits, seen = split_data(exclusion, parent_run, dist, args.seed)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    model = load_x_for_eval(".", str(args.checkpoint))
    bind_x_tokenizer(model, tok)
    reader = freeze_for_reader(model)
    model.checkpoint_parent = dict(path=str(args.checkpoint.resolve()), id=model.checkpoint_id)
    cases = {}
    for idx, (split, space) in enumerate(splits.items()):
        split_rng = random.Random(args.seed + idx + 1)
        source = list(
            make_cases(tok, split_rng, **space, count={"train": 6144, "dev": 36, "test": 96}[split])
        )
        cases[split] = list(
            (train_examples if split == "train" else eval_examples)(tok, split_rng, source)
        )
    catalogs = {
        key: dict(
            sites=sorted({r["deployment"] for r in rows}),
            values=sorted({v for r in rows for v in [*r["answers"], r["old"]]}),
        )
        for key, rows in cases.items()
    }
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if any(set(catalogs[a][k]) & set(catalogs[b][k]) for k in ("sites", "values")):
            raise AssertionError("Split overlap")
    if any(set(catalogs[k]["values"]) & seen for k in ("dev", "test")):
        raise AssertionError("Ancestor value reuse in held-out pairs")
    report = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        parent=model.checkpoint_parent,
        parent_sha256=sha(args.checkpoint),
        trainable_parameters=sum(x.numel() for x in reader.parameters()),
        ancestor_values=len(seen),
        splits=catalogs,
        train_layouts=dict(Counter(r["mode"] for r in cases["train"])),
        train_background_tokens=dict(
            Counter(len(r["before"]) + len(r["after"]) for r in cases["train"])
        ),
        history=[],
    )
    for k in ("dev", "test"):
        write_json(args.out / (k + "_cases.json"), cases[k])
    write_json(args.out / "training.json", report)
    train = []
    for start in range(0, len(cases["train"]), 128):
        rows = cases["train"][start : start + 128]
        encoded = encode_batch(model, tok, rows)
        train.extend((bool(row["release"]), encoded[2 * i : 2 * i + 2]) for i, row in enumerate(rows))
    header = torch.tensor([[tok.bos_token_id, *encode_text(tok, "资料：")]])
    with torch.no_grad():
        past = model(header, use_cache=True, num_loops=2, collect_routers=False).past_key_values
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=0.0)
    baseline, _ = evaluate(model, tok, cases["dev"])
    report["history"].append(dict(step=0, metrics=baseline))
    print(report["history"][-1], flush=True)
    best = rank(baseline)
    report["selected_step"] = 0
    selected = str(args.checkpoint)
    running = 0.0
    for step in range(1, args.steps + 1):
        batch = {False: [], True: []}
        for late, pair in rng.choices(train, k=args.batch_pairs):
            batch[late].extend(pair)
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(model, batch, past)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        optimizer.step()
        running += float(loss.detach())
        if step % 100 == 0:
            print(f"step={step} ce={running / 100:.5f}", flush=True)
            running = 0.0
        if step % args.check_every == 0 or step == args.steps:
            # Evaluate the exact precision that will be available after checkpoint reload.
            with torch.no_grad():
                saved_params = [p.clone() for p in reader.parameters()]
                for p in reader.parameters():
                    p.copy_(p.half())
            metrics, _ = evaluate(model, tok, cases["dev"])
            report["history"].append(dict(step=step, metrics=metrics))
            print(report["history"][-1], flush=True)
            if rank(metrics) > best:
                best = rank(metrics)
                selected = save_x_checkpoint(
                    model,
                    args.out,
                    "short_reader",
                    model.config.hidden_size,
                    extra=dict(training_step=step, dev=metrics),
                )
                report["selected_step"] = step
            with torch.no_grad():
                for p, original in zip(reader.parameters(), saved_params):
                    p.copy_(original)
            report["selected_checkpoint"] = selected
            write_json(args.out / "training.json", report)
    parent = load_x_for_eval(".", str(args.checkpoint))
    loaded = load_x_for_eval(".", selected)
    bind_x_tokenizer(parent, tok)
    bind_x_tokenizer(loaded, tok)
    before, after = parent.state_dict(), loaded.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    if any(not k.startswith("model.thinking_core.memory_reader.") for k in changed):
        raise AssertionError("A frozen state tensor changed")
    saved_dev, _ = evaluate(loaded, tok, cases["dev"])
    selected_dev = next(
        rec["metrics"] for rec in report["history"] if rec["step"] == report["selected_step"]
    )
    if selected_dev != saved_dev:
        raise AssertionError("Selected development results changed after reload")
    results = dict(
        checkpoint=selected,
        checkpoint_sha256=sha(selected),
        changed_tensors=changed,
        frozen_tensors=len(before) - len(changed),
        reloaded_dev=saved_dev,
        test_cases_sha256=sha(args.out / "test_cases.json"),
    )
    for name, net in (("before", parent), ("after", loaded)):
        summary, outputs = evaluate(net, tok, cases["test"], with_outputs=True)
        results[name] = dict(summary=summary, samples=outputs)
    old_rows = json.loads(Path(dist["reference"]).read_text())["samples"]
    old_data = json.loads(Path(dist["reference"]).read_text())["config"]["data"]
    old_lines = Path(old_data).read_text().splitlines()
    clean = []
    for row in old_rows:
        item = json.loads(old_lines[row["line"] - 1])
        ids = [row[w]["sentence"]["evidence_ids"] for w in ("original", "edited")]
        answers = [row["original_answer"], row["edited_answer"]]
        scores = []
        for net in (parent, loaded):
            outcomes = [
                run_channel(
                    net,
                    tok,
                    dict(item, answer=v),
                    torch.tensor(x),
                    [0],
                    torch.tensor(row["original"]["sentence"]["release_rounds"]),
                    x,
                    "reader",
                    2,
                    32,
                )
                for v, x in zip(answers, ids)
            ]
            scores.append(outcomes)
        clean.append(
            dict(
                line=row["line"],
                family=row["family"],
                answers=answers,
                before=scores[0],
                after=scores[1],
            )
        )
    if any(
        x["generated_ids"] != row[w]["sentence"]["reader"]["generated_ids"]
        for result, row in zip(clean, old_rows)
        for x, w in zip(result["before"], ("original", "edited"))
    ):
        raise AssertionError("Original clean benchmark failed to replay")
    results["original_clean"] = clean
    write_json(args.out / "results.json", results)
    print({k: results[k]["summary"] for k in ("before", "after")}, flush=True)


if __name__ == "__main__":
    main()
