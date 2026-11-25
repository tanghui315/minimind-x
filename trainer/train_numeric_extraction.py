"""Prompt-only numerical extraction SFT and paired, free-generation development checks."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.data_gen.gen_evidence_families import DEVICES, gen_sample
from scripts.eval_fact_swap import FAMILIES, encode_text, evidence_prompt, locate_facts
from scripts.eval_counterfactual import Policy, complete, decode
from scripts.eval_needle import exact_match
from model.minimind_x.controller import AdaptiveController
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval, save_x_checkpoint


def split_values(rng, reserved):
    values = [str(v) for v in range(1000, 10000) if str(v) not in reserved]
    rng.shuffle(values)
    return values[512:], values[:512]


def paired_fact_samples(rng, rows, sites, values):
    for i in range(rows):
        family = FAMILIES[i % len(FAMILIES)]
        site = rng.choice(sites)
        sample = gen_sample(rng, family, [f"{base}-{site:03d}" for base in DEVICES])
        sample["family"] = family
        facts = locate_facts(sample)
        start, end = facts.value_span
        original = sample["context"]
        pair = []
        for value in rng.sample(values, 2):
            pair.append(dict(sample, context=original[:start] + value + original[end:], answer=value))
        yield pair


def generate_examples(rng, tok, rows, sites, values):
    examples = []
    delimiter = encode_text(tok, "答案：")
    for samples in paired_fact_samples(rng, rows, sites, values):
        pair = []
        for sample in samples:
            prefix = [tok.bos_token_id, *encode_text(tok, evidence_prompt(sample, locate_facts(sample)))]
            target = encode_text(tok, sample["answer"]) + [tok.eos_token_id]
            pair.append((prefix, delimiter, target))
        examples.append(pair)
    return examples


def answer_logits(model, examples, device, loops, memory_payload=None):
    inputs, positions, targets = [], [], []
    for row, (prompt, delimiter, target) in enumerate(examples):
        prefix = prompt + delimiter
        inputs.append(torch.tensor(prefix + target[:-1]))
        positions.extend((row, len(prefix) - 1 + i) for i in range(len(target)))
        targets.extend(target)
    # Right padding is strictly after every supervised position in this causal LM.
    ids = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True).to(device)
    if ids.shape[1] > model.config.max_position_embeddings:
        raise ValueError("Numerical QA example exceeds the checkpoint's position limit")
    out = model.model(ids, num_loops=loops, collect_routers=False, memory_payload=memory_payload)
    rows, columns = torch.tensor(positions, device=device).T
    logits = model.lm_head(out["hidden_states"][rows, columns])
    return logits, torch.tensor(targets, device=device)


def batch_loss(model, examples, device, loops, memory_payload=None):
    logits, targets = answer_logits(model, examples, device, loops, memory_payload)
    return F.cross_entropy(logits, targets)


@torch.inference_mode()
def dev_check(model, tok, pairs, device):
    model.eval()
    correct, tracked = 0, 0
    for pair in pairs:
        hits = []
        for prompt, delimiter, target in pair:
            ctrl = AdaptiveController(model)
            ctrl.begin(torch.empty(0, device=device, dtype=torch.long))
            policy = Policy(initial_only=True)
            out = decode(ctrl, torch.tensor([prompt], device=device), None, policy, prefill=True)
            for token in delimiter:
                out = decode(ctrl, torch.tensor([[token]], device=device), out.past_key_values, policy)
            predicted = complete(ctrl, out, 32, tok.eos_token_id, policy)
            hits.append(
                exact_match(tok.decode(predicted, skip_special_tokens=True), tok.decode(target[:-1]))
            )
        correct += sum(hits)
        tracked += all(hits)
    return dict(correct=correct, worlds=2 * len(pairs), tracked=tracked, pairs=len(pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--reserve-report", required=True, help="Previous fact-swap JSON, for split exclusions only"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2843)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.check_every) < 1 or args.lr <= 0:
        parser.error("Training sizes and learning rate must be positive")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(str(ROOT / "model"))
    model = load_x_for_eval(".", args.checkpoint, device=args.device)
    bind_x_tokenizer(model, tok)
    model.checkpoint_parent = dict(id=model.checkpoint_id, path=str(Path(args.checkpoint).resolve()))
    heldout = json.loads(Path(args.reserve_report).read_text())["samples"]
    reserved = {str(row[k]) for row in heldout for k in ("original_answer", "edited_answer")}
    sites = list(range(256))
    random.Random(0).shuffle(sites)  # Last 64 sites are validation sites in gen_evidence_families.
    train_values, dev_values = split_values(rng, reserved)
    train = generate_examples(rng, tok, 6144, sites[:160], train_values)
    dev = generate_examples(rng, tok, 36, sites[160:192], dev_values)
    flat = [world for pair in train for world in pair]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    history = []
    report = dict(
        config=vars(args),
        parent=model.checkpoint_parent,
        reserved_values=sorted(reserved),
        train_sites=sites[:160],
        dev_sites=sites[160:192],
        train_pairs=len(train),
        dev_pairs=len(dev),
        history=history,
    )
    total_loss = 0.0
    best = (-1, -1)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(
            model,
            rng.choices(flat, k=args.batch_size),
            args.device,
            model.config.max_loop if step % 4 == 0 else 1,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if step % 50 == 0:
            print(f"step={step} train_ce={total_loss / 50:.4f}", flush=True)
            total_loss = 0.0
        if step % args.check_every == 0 or step == args.steps:
            metrics = dev_check(model, tok, dev, args.device)
            history.append(dict(step=step, **metrics))
            print(history[-1], flush=True)
            rank = (metrics["tracked"], metrics["correct"])
            if rank > best:
                best = rank
                path = save_x_checkpoint(
                    model,
                    out_dir,
                    "numeric_sft",
                    model.config.hidden_size,
                    extra=dict(training_step=step, dev=metrics),
                )
                report.update(selected_step=step, checkpoint=path)
            (out_dir / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if metrics["tracked"] >= 0.90 * len(dev):
                break


if __name__ == "__main__":
    main()
