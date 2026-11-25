"""Reader-only SFT on paired facts with a frozen numerical extraction backbone."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_fact_swap import (
    encode_text,
    evidence_blocks,
    locate_facts,
    run_world,
    support_sentences,
    pair_metrics,
)
from trainer.train_numeric_extraction import answer_logits, batch_loss, paired_fact_samples, split_values
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval, save_x_checkpoint


def freeze_for_reader(model):
    if not (model.config.residual_v2 and model.config.use_memory and model.config.use_retrieval):
        raise ValueError("Reader training requires a retrieval-enabled V2 checkpoint")
    model.eval().requires_grad_(False)
    reader = model.model.thinking_core.memory_reader
    reader.requires_grad_(True)
    return reader


@torch.no_grad()
def encode_examples(model, tok, samples, evidence_mode="gold_sentences"):
    """Use the same block boundaries, Prelude, and valid-token payload as the controller."""
    device = next(model.parameters()).device
    bs = model.config.block_size
    delimiter = encode_text(tok, "答案：")
    examples, padded_docs, lengths = [], [], []
    for sample in samples:
        facts = locate_facts(sample)
        if evidence_mode == "gold_blocks":
            encoded = tok(sample["context"], add_special_tokens=False, return_offsets_mapping=True)
            blocks = evidence_blocks(encoded.offset_mapping, facts.spans, bs)
            if len(blocks) > model.config.budget_blocks(0):
                raise ValueError("Gold evidence exceeds the evaluation memory budget")
            # Each source block gets its own Prelude positions. Only the last selected
            # block can be short, so concatenation preserves the encoder boundaries.
            ids = torch.tensor(
                [
                    token
                    for block in blocks
                    for token in encoded.input_ids[block * bs : (block + 1) * bs]
                ],
                device=device,
            )
        elif evidence_mode == "gold_sentences":
            ids = torch.tensor(encode_text(tok, support_sentences(sample, facts)), device=device)
        else:
            raise ValueError(f"Unknown training evidence mode {evidence_mode}")
        padded_docs.append(torch.nn.functional.pad(ids, (0, -len(ids) % bs)))
        lengths.append(len(ids))
        prefix = [tok.bos_token_id, *encode_text(tok, sample["question"])]
        target = encode_text(tok, sample["answer"]) + [tok.eos_token_id]
        examples.append((prefix, delimiter, target))
    flat = model.model.encode_context_blocks(torch.cat(padded_docs)).flatten(0, 1)
    docs = flat.split([len(doc) for doc in padded_docs])
    return [
        (example, doc[:length].cpu().clone()) for example, doc, length in zip(examples, docs, lengths)
    ]


class DigitSupervision:
    """Teacher-force preceding single digits. Credit any numeric token starting with the next digit."""

    def __init__(self, tok, vocab_size, device):
        self.eos = tok.eos_token_id
        self.digit_ids = [encode_text(tok, str(i)) for i in range(10)]
        if not all(len(ids) == 1 for ids in self.digit_ids):
            raise ValueError("Digit supervision needs an individual token for every decimal digit")
        self.numeric_tokens = {}
        self.groups = torch.zeros(10, vocab_size, dtype=torch.bool)
        for token in range(min(vocab_size, len(tok))):
            piece = tok.decode([token], clean_up_tokenization_spaces=False)
            if piece.isascii() and piece.isdecimal():
                self.numeric_tokens[token] = piece
                self.groups[int(piece[0]), token] = True
        self.groups = self.groups.to(device)
        self.token_classes = torch.full((vocab_size,), -1, device=device, dtype=torch.long)
        for digit, (token,) in enumerate(self.digit_ids):
            if self.numeric_tokens.get(token) != str(digit):
                raise ValueError("Decimal digit token does not round-trip")
            self.token_classes[token] = digit

    def digit_example(self, example):
        prefix, delimiter, target = example
        if target[-1] != self.eos or any(i not in self.numeric_tokens for i in target[:-1]):
            raise ValueError("Expected a numerical answer followed by EOS")
        value = "".join(self.numeric_tokens[i] for i in target[:-1])
        if len(value) != 4:
            raise ValueError("Expected a four-digit numerical answer")
        return prefix, delimiter, [self.digit_ids[int(char)][0] for char in value]

    def loss(self, logits, digit_ids):
        classes = self.token_classes[digit_ids]
        log_probs = logits.float().log_softmax(dim=-1)
        return -log_probs.masked_fill(~self.groups[classes], -torch.inf).logsumexp(dim=-1).mean()


def reader_loss(model, batch, device, loops, digit_supervision=None, digit_weight=0.0):
    examples, memories = zip(*batch)
    payload = pad_sequence(memories, batch_first=True).to(device)
    lengths = torch.tensor([len(mem) for mem in memories], device=device)
    valid = torch.arange(payload.size(1), device=device)[None, :] < lengths[:, None]
    memory = (payload, valid)
    loss = batch_loss(model, examples, device, loops, memory_payload=memory)
    if digit_weight:
        digit_examples = [digit_supervision.digit_example(example) for example in examples]
        logits, targets = answer_logits(model, digit_examples, device, loops, memory_payload=memory)
        loss = loss + digit_weight * digit_supervision.loss(logits, targets)
    return loss


def evaluate_dev(model, tok, pairs, evidence_mode="gold_sentences"):
    correct, tracked, flipped = 0, 0, 0
    for pair in pairs:
        targets = [encode_text(tok, sample["answer"]) for sample in pair]
        a, b = [run_world(model, tok, s, locate_facts(s), evidence_mode, targets, 32, 0) for s in pair]
        metrics = pair_metrics(a, b, *(s["answer"] for s in pair))
        correct += metrics["original_correct"] + metrics["edited_correct"]
        tracked += metrics["tracked"]
        flipped += metrics["preference_flipped"]
    return dict(
        correct=correct, worlds=2 * len(pairs), tracked=tracked, pairs=len(pairs), flipped=flipped
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--extraction-run",
        required=True,
        help="Parent numerical SFT training.json with split and reservations",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=6103)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--check-every", type=int, default=500)
    p.add_argument("--batch-pairs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument(
        "--evidence-mode",
        choices=("gold_sentences", "gold_blocks", "mixed"),
        default="gold_sentences",
        help="Evidence used for Reader SFT and development checkpoint selection",
    )
    p.add_argument(
        "--digit-weight", type=float, default=0.0, help="Weight of the auxiliary per-digit loss"
    )
    args = p.parse_args()
    if (
        min(args.steps, args.batch_pairs, args.check_every) < 1
        or args.lr <= 0
        or not 0 <= args.digit_weight < float("inf")
    ):
        p.error("Step counts, batch size and learning rate must be positive")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads(Path(args.extraction_run).read_text())
    reserved = set(previous["reserved_values"])
    train_pool, dev_pool = split_values(random.Random(previous["config"]["seed"]), reserved)
    if set(previous["train_sites"]) & set(previous["dev_sites"]):
        p.error("Training and development deployment groups overlap")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model = load_x_for_eval(".", args.checkpoint, device=args.device)
    tok = AutoTokenizer.from_pretrained(str(ROOT / "model"))
    bind_x_tokenizer(model, tok)
    reader = freeze_for_reader(model)
    model.checkpoint_parent = dict(path=str(Path(args.checkpoint).resolve()), id=model.checkpoint_id)
    modes = ("gold_sentences", "gold_blocks") if args.evidence_mode == "mixed" else (args.evidence_mode,)
    train = {mode: [] for mode in modes}
    train_values = set()
    block_batch = []
    for pair in paired_fact_samples(rng, 6144, previous["train_sites"], train_pool):
        block_batch.extend(pair)
        train_values.update(s["answer"] for s in pair)
        if len(block_batch) == 32:
            for mode in modes:
                train[mode].extend(encode_examples(model, tok, block_batch, mode))
            block_batch.clear()
    if block_batch:
        for mode in modes:
            train[mode].extend(encode_examples(model, tok, block_batch, mode))
    num_pairs = len(train[modes[0]]) // 2
    dev = list(paired_fact_samples(rng, 36, previous["dev_sites"], dev_pool))
    dev_values = {s["answer"] for pair in dev for s in pair}
    assert not (train_values & dev_values or (train_values | dev_values) & reserved)
    history = []
    report = dict(
        config=vars(args),
        parent=model.checkpoint_parent,
        train_pairs=num_pairs,
        unique_train_values=len(train_values),
        unique_dev_values=len(dev_values),
        reserved_values=sorted(reserved),
        train_sites=previous["train_sites"],
        dev_sites=previous["dev_sites"],
        trainable=[n for n, t in model.named_parameters() if t.requires_grad],
        trainable_parameters=sum(t.numel() for t in reader.parameters()),
        history=history,
    )
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=0.0)
    digits = DigitSupervision(tok, model.config.vocab_size, args.device) if args.digit_weight else None
    best = ()
    loss_sum = 0.0
    for step in range(1, args.steps + 1):
        indices = rng.choices(range(num_pairs), k=args.batch_pairs)
        batch = [train[mode][2 * i + j] for mode in modes for i in indices for j in range(2)]
        optimizer.zero_grad(set_to_none=True)
        loss = reader_loss(
            model,
            batch,
            args.device,
            model.config.max_loop if step % 4 == 0 else 1,
            digits,
            args.digit_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        if step % 100 == 0:
            print(f"step={step} train_loss={loss_sum / 100:.4f}", flush=True)
            loss_sum = 0.0
        if step % args.check_every == 0 or step == args.steps:
            by_mode = {mode: evaluate_dev(model, tok, dev, mode) for mode in modes}
            metrics = (
                by_mode[modes[0]]
                if len(modes) == 1
                else {
                    **{key: sum(row[key] for row in by_mode.values()) for key in by_mode[modes[0]]},
                    "by_mode": by_mode,
                }
            )
            history.append(dict(step=step, **metrics))
            print(history[-1], flush=True)
            worst_tracked = min(v["tracked"] for v in by_mode.values())
            rank = (worst_tracked, metrics["tracked"], metrics["correct"], metrics["flipped"])
            if rank > best:
                best = rank
                path = save_x_checkpoint(
                    model,
                    output,
                    "reader_sft",
                    model.config.hidden_size,
                    extra=dict(training_step=step, dev=metrics),
                )
                report.update(selected_step=step, checkpoint=path)
            (output / "training.json").write_text(json.dumps(report, indent=2))
            if worst_tracked >= 0.9 * len(dev):
                break


if __name__ == "__main__":
    main()
