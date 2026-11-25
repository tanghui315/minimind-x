"""Paired fact substitution on the numerical retrieval-QA families."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.retrieval_qa_dataset import encode_evidence, token_span
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.memory.working_memory import WorkingMemory
from scripts.eval_counterfactual import Policy, complete, decode, evidence, fork_controller, score
from scripts.eval_needle import exact_match
from trainer.x_utils import bind_x_tokenizer, load_x_for_eval

FAMILIES = ("evidence_needed", "new_evidence", "multi_hop")
ARMS = ("no_memory", "retrieval", "gold_blocks", "gold_sentences", "prompt_sentences")


@dataclass(frozen=True)
class Facts:
    spans: tuple
    value_span: tuple


def locate_facts(sample):
    text, answer = sample["context"], str(sample["answer"])
    start, end = sample["gold_span"]
    if not 0 <= start < end <= len(text) or not re.fullmatch(r"[1-9][0-9]*", answer):
        raise ValueError("Expected a valid fact annotation and positive integer answer")
    spans = [(start, end)]
    if sample["family"] == "multi_hop":
        bridge = re.fullmatch(r"链路表：[^。]+ 的上行通道绑定到 ([^。]+)。", text[start:end])
        if bridge is None:
            raise ValueError("Unrecognized bridge annotation")
        target = re.escape(bridge.group(1))
        matches = list(re.finditer(rf"配置记录：{target} 的阈值设定为 [0-9]+。", text))
        if len(matches) != 1:
            raise ValueError("Expected exactly one value fact for the linked entity")
        start, end = matches[0].span()
        spans.append((start, end))
    match = re.search(r"(?:阈值设定为|实际阈值为) ([0-9]+)(?=[。，])", text[start:end])
    if match is None or match.group(1) != answer:
        raise ValueError("Annotated value disagrees with reference answer")
    return Facts(tuple(spans), (start + match.start(1), start + match.end(1)))


def encode_text(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def support_sentences(sample, facts):
    return "\n".join(sample["context"][start:end] for start, end in facts.spans)


def evidence_prompt(sample, facts):
    return f"资料：{support_sentences(sample, facts)}\n问题：{sample['question']}"


def evidence_blocks(offsets, spans, block_size):
    blocks = set()
    for span in spans:
        bounds = token_span(offsets, span)
        if bounds is None:
            raise ValueError("Evidence span is outside the tokenized document")
        first, last = bounds
        blocks.update(range(first // block_size, (last - 1) // block_size + 1))
    return sorted(blocks)


def counterfactual(sample, facts, tok, seed):
    """Edit one field. Keep the textual length, token layout and unrelated IDs fixed."""
    context, original = sample["context"], str(sample["answer"])
    start, end = facts.value_span
    if original in sample["question"]:
        raise ValueError("Original answer already appears in question")
    encoded = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    first, last = token_span(encoded.offset_mapping, facts.value_span)
    old_answer_ids = encode_text(tok, original)
    rng = random.Random(f"{seed}/{context[start:end]}/{sample['question']}")
    for _ in range(1000):
        new = "".join(
            rng.choice([d for d in ("123456789" if i == 0 else "0123456789") if d != c])
            for i, c in enumerate(original)
        )
        if new in context or new in sample["question"]:
            continue
        target = encode_text(tok, new)
        if target == old_answer_ids or len(target) != len(old_answer_ids):
            continue
        changed = context[:start] + new + context[end:]
        alt = tok(changed, add_special_tokens=False, return_offsets_mapping=True)
        if (
            alt.offset_mapping == encoded.offset_mapping
            and alt.input_ids[:first] == encoded.input_ids[:first]
            and alt.input_ids[last:] == encoded.input_ids[last:]
            and alt.input_ids[first:last] != encoded.input_ids[first:last]
        ):
            return dict(sample, context=changed, answer=new)
    raise ValueError("Could not generate a distinct tokenizer-matched counterfactual")


class PinnedEvidence(AdaptiveController):
    """Supply selected document blocks directly, at their annotated release round."""

    def pin_visible(self):
        wanted = [i for i in self.gold_ids if int(self.store.block_visible_from[i]) <= self.store.round]
        if wanted == evidence(self):
            return None
        idx = self.store.context_ids.new_tensor([wanted])
        hidden = self.store.get_block_hidden(self.raw, idx)
        version = 1 if self.memory is None else self.memory.memory_version + 1
        self.memory = WorkingMemory(
            hidden,
            self.store.block_keys[idx],
            source_ids=idx,
            memory_version=version,
            metadata={"context_tokens": self.store.context_ids.numel()},
        )
        self._record_memory(self.memory)
        return self._memory_input()

    def make_loop_callback(self, **kwargs):
        def callback(r, hidden):
            self.store.advance_round(r + 1)
            return self.pin_visible()

        return callback


def select_evidence(tok, sample, facts, mode, bs, device):
    if mode == "gold_sentences":
        ctx = torch.tensor(encode_text(tok, support_sentences(sample, facts)), device=device)
        gold = list(range((ctx.numel() + bs - 1) // bs))
        # Only the single-fact new_evidence family delays the answer.
        release = torch.full(
            (len(gold),), int(sample.get("evidence_visible_from_round", 0)), device=device
        )
    else:
        ctx, _, release = encode_evidence(sample, tok, bs)
        ctx, release = ctx.to(device), release.to(device)
        offsets = tok(sample["context"], add_special_tokens=False, return_offsets_mapping=True).offset_mapping
        gold = evidence_blocks(offsets, facts.spans, bs)
    return ctx, gold, release


def build_arm(model, tok, sample, facts, mode, budget_idx):
    device = next(model.parameters()).device
    bs = model.config.block_size
    prompt = sample["question"]
    if mode in ("no_memory", "prompt_sentences"):
        if mode == "prompt_sentences":
            prompt = evidence_prompt(sample, facts)
        ctrl = AdaptiveController(model)
        ctrl.begin(torch.empty(0, device=device, dtype=torch.long))
        return ctrl, prompt, [], Policy(initial_only=True)
    if mode not in ("gold_blocks", "gold_sentences", "retrieval"):
        raise ValueError(f"Unknown evidence mode {mode}")
    pinned = mode != "retrieval"
    ctx, gold, release = select_evidence(tok, sample, facts, mode, bs, device)
    if pinned and len(gold) > model.config.budget_blocks(budget_idx):
        raise ValueError("Oracle evidence exceeds the configured budget")
    cls = PinnedEvidence if pinned else AdaptiveController
    ctrl = cls(model, block_visible_from=release, fixed_budget_idx=budget_idx)
    ctrl.begin(ctx)
    if pinned:
        ctrl.gold_ids = gold
        ctrl.pin_visible()
    else:
        ids = torch.tensor([[tok.bos_token_id, *encode_text(tok, prompt)]], device=device)
        hidden = model.model.prelude_hidden(ids)
        ctrl.memory = ctrl.retrieve_into_memory(hidden, budget_idx)
    return ctrl, prompt, gold, Policy()


@torch.no_grad()
def run_world(model, tok, sample, facts, mode, targets, cap, budget_idx):
    ctrl, prompt, gold, policy = build_arm(model, tok, sample, facts, mode, budget_idx)
    device = next(model.parameters()).device
    q = torch.tensor([[tok.bos_token_id, *encode_text(tok, prompt)]], device=device)
    delim = encode_text(tok, "答案：")
    if (
        not delim
        or q.numel() + len(delim) + max(cap, *(len(t) for t in targets)) - 1
        > model.config.max_position_embeddings
    ):
        raise ValueError("Prompt and continuation exceed model position limit or lack delimiter")
    initial = evidence(ctrl)
    out = decode(ctrl, q, None, policy, prefill=True)
    for token in delim[:-1]:
        out = decode(ctrl, q.new_tensor([[token]]), out.past_key_values, policy)
    reader_updates = []

    def observe_reader(module, inputs, outputs):
        _, kv = inputs[:2]
        delta = outputs[1]["delta"].float()
        hidden = inputs[0].float()
        reader_updates.append(
            dict(
                valid_tokens=int(kv.valid.sum()) if kv is not None else 0,
                relative_l2=float(delta.norm() / hidden.norm().clamp_min(1e-12)),
            )
        )

    reader = model.model.thinking_core.memory_reader
    with reader.register_forward_hook(observe_reader):
        out = decode(ctrl, q.new_tensor([[delim[-1]]]), out.past_key_values, policy)
    resident = evidence(ctrl)
    payload_hash = None
    if ctrl.memory is not None and resident:
        payload = (
            ctrl.memory.token_hidden()[ctrl.memory.token_valid()].detach().float().cpu().contiguous()
        )
        payload_hash = hashlib.sha256(payload.numpy().tobytes()).hexdigest()
    # Gold scoring must not change retrieval state for the other candidate or for free generation.
    logps = [score(fork_controller(ctrl), out, q.new_tensor(ids), policy) for ids in targets]
    gen = complete(fork_controller(ctrl), out, cap, tok.eos_token_id, policy)
    return dict(
        generated_ids=gen,
        prediction=tok.decode(gen, skip_special_tokens=True),
        token_logps=logps,
        mean_logps=[mean(lp) for lp in logps],
        gold_blocks=gold,
        initial_blocks=initial,
        first_answer_blocks=resident,
        all_gold_present=bool(gold) and set(gold).issubset(resident),
        valid_payload_sha256=payload_hash,
        first_answer_reader_calls=reader_updates,
    )


def pair_metrics(a, b, old, new):
    ma = a["mean_logps"][1] - a["mean_logps"][0]
    mb = b["mean_logps"][1] - b["mean_logps"][0]
    return dict(
        output_changed=a["prediction"] != b["prediction"],
        tokens_changed=a["generated_ids"] != b["generated_ids"],
        original_correct=exact_match(a["prediction"], old),
        edited_correct=exact_match(b["prediction"], new),
        tracked=exact_match(a["prediction"], old) and exact_match(b["prediction"], new),
        preference_flipped=(ma < 0 < mb),
        margin_old_world=ma,
        margin_new_world=mb,
        margin_shift=mb - ma,
        payload_changed=a["valid_payload_sha256"] != b["valid_payload_sha256"],
    )


def summarize(rows):
    result = {}
    for mode in ARMS:
        pairs = [r["arms"][mode]["metrics"] for r in rows]
        counts = {
            k: sum(p[k] for p in pairs)
            for k in (
                "output_changed",
                "tokens_changed",
                "original_correct",
                "edited_correct",
                "tracked",
                "preference_flipped",
                "payload_changed",
            )
        }
        shifts = [p["margin_shift"] for p in pairs]
        counts.update(
            n=len(rows),
            mean_margin_shift=mean(shifts),
            max_abs_shift=max(map(abs, shifts)),
            positive_above_0p0001=sum(d > 1e-4 for d in shifts),
            negative_below_minus_0p0001=sum(d < -1e-4 for d in shifts),
            reader_nonzero_in_both=sum(
                all(
                    any(
                        c["valid_tokens"] and c["relative_l2"] > 0
                        for c in r["arms"][mode][w]["first_answer_reader_calls"]
                    )
                    for w in ("original", "edited")
                )
                for r in rows
            ),
            all_gold_in_both=sum(
                all(r["arms"][mode][w]["all_gold_present"] for w in ("original", "edited")) for r in rows
            ),
        )
        result[mode] = counts
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=str(ROOT / "model"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=773)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--budget-idx", type=int, default=0)
    p.add_argument(
        "--zero-reader-scale",
        action="store_true",
        help="Ablate the loaded Reader residual during evaluation",
    )
    args = p.parse_args()
    model = load_x_for_eval(".", args.checkpoint, device=args.device)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bind_x_tokenizer(model, tok)
    if not (model.config.use_memory and model.config.use_retrieval and model.config.residual_v2):
        p.error("This Reader experiment requires a retrieval-enabled residual V2 checkpoint")
    if args.zero_reader_scale:
        with torch.no_grad():
            model.model.thinking_core.memory_reader.scale_logit.zero_()
    if args.max_new_tokens < 1 or not 0 <= args.budget_idx < len(model.config.memory_budgets):
        p.error("Invalid generation cap or budget index")
    rows, skipped = [], Counter()
    with open(args.data, encoding="utf-8") as data:
        for i, line in enumerate(data, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("split", "val") != "val":
                skipped["other_split"] += 1
                continue
            family = sample.get("family", "unspecified")
            if family not in FAMILIES:
                skipped[family] += 1
                continue
            facts = locate_facts(sample)
            edited = counterfactual(sample, facts, tok, args.seed)
            targets = [encode_text(tok, str(r["answer"])) for r in (sample, edited)]
            if len(targets[0]) > args.max_new_tokens:
                raise ValueError("Generation cap is too short for the paired answers")
            arms = {}
            for mode in ARMS:
                a, b = [
                    run_world(model, tok, r, facts, mode, targets, args.max_new_tokens, args.budget_idx)
                    for r in (sample, edited)
                ]
                metrics = pair_metrics(a, b, str(sample["answer"]), edited["answer"])
                if mode == "no_memory" and (
                    metrics["tokens_changed"] or a["token_logps"] != b["token_logps"]
                ):
                    raise RuntimeError("Fact leaked into the no-memory control")
                arms[mode] = dict(original=a, edited=b, metrics=metrics)
            rows.append(
                dict(
                    line=i,
                    family=family,
                    question=sample["question"],
                    original_answer=sample["answer"],
                    edited_answer=edited["answer"],
                    evidence_spans=facts.spans,
                    value_span=facts.value_span,
                    original_facts=[sample["context"][s:e] for s, e in facts.spans],
                    edited_facts=[edited["context"][s:e] for s, e in facts.spans],
                    arms=arms,
                )
            )
    if not rows:
        p.error("No eligible numerical validation pairs")
    with open(args.checkpoint, "rb") as weights:
        digest = hashlib.sha256()
        for chunk in iter(lambda: weights.read(1024 * 1024), b""):
            digest.update(chunk)
        ckpt = digest.hexdigest()
    report = dict(
        config=vars(args),
        checkpoint_sha256=ckpt,
        checkpoint_id=getattr(model, "checkpoint_id", None),
        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        skipped=dict(skipped),
        summary=summarize(rows),
        by_family={
            f: summarize([r for r in rows if r["family"] == f])
            for f in sorted({r["family"] for r in rows})
        },
        samples=rows,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
