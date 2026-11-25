import random

import pytest
import torch
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from scripts import eval_fact_swap as fs
from trainer.train_numeric_extraction import paired_fact_samples
from trainer.train_numeric_reader import encode_examples, freeze_for_reader, reader_loss


@pytest.mark.parametrize("mode", ["gold_sentences", "gold_blocks"])
def test_reader_training_payload_matches_pinned_blocks_and_keeps_values_out_of_question(mode):
    tok = AutoTokenizer.from_pretrained(fs.ROOT / "model")
    config = MiniMindXConfig(
        variant="minimind_x",
        residual_v2=True,
        hidden_size=32,
        vocab_size=len(tok),
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
        block_size=8,
        max_loop=2,
        memory_budgets=[128, 256, 512],
    )
    model = MiniMindXForCausalLM(config).eval()
    pairs = list(paired_fact_samples(random.Random(4), 3, [211], ["6397", "7381", "5219"]))
    samples = [s for pair in pairs for s in pair]
    examples = encode_examples(model, tok, samples, mode)
    checked_gap = checked_boundary = False
    for s, ((prefix, delim, target), hidden) in zip(samples, examples):
        assert prefix == [tok.bos_token_id, *fs.encode_text(tok, s["question"])]
        assert s["answer"] not in s["question"]
        assert tok.decode(delim) == "答案：" and tok.decode(target[:-1]) == s["answer"]
        facts = fs.locate_facts(s)
        ctrl, _, blocks, _ = fs.build_arm(model, tok, s, facts, mode, 0)
        ctrl.make_loop_callback()(0, None)  # Release delayed facts.
        payload, valid = ctrl._memory_input()
        torch.testing.assert_close(hidden, payload[valid])
        assert not hidden.requires_grad
        if mode == "gold_blocks":
            offsets = tok(
                s["context"], add_special_tokens=False, return_offsets_mapping=True
            ).offset_mapping
            fact_tokens = {
                i
                for start, end in facts.spans
                for i, (a, b) in enumerate(offsets)
                if start < b and a < end
            }
            selected = {i for i in range(len(offsets)) if i // config.block_size in blocks}
            assert fact_tokens <= selected
            assert len(hidden) == len(selected)
            checked_boundary |= len(selected) > len(fact_tokens)
            checked_gap |= any(b > a + 1 for a, b in zip(blocks, blocks[1:]))
    if mode == "gold_blocks":
        assert checked_gap and checked_boundary
    for i in range(0, len(samples), 2):
        assert examples[i][0][0] == examples[i + 1][0][0]
        assert not torch.equal(examples[i][1], examples[i + 1][1])


def test_reader_only_step_preserves_backbone_and_masks_memory_padding():
    torch.manual_seed(11)
    config = MiniMindXConfig(
        variant="minimind_x",
        residual_v2=True,
        hidden_size=32,
        vocab_size=80,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        max_loop=2,
        dropout=0.0,
    )
    model = MiniMindXForCausalLM(config)
    reader = freeze_for_reader(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    batch = [
        (([1, 4, 5], [6], [13, 15, 2]), torch.randn(11, 32)),
        (([1, 4, 5], [6], [12, 2]), torch.randn(3, 32)),
    ]
    loss = reader_loss(model, batch, "cpu", 2)
    individual = [reader_loss(model, [row], "cpu", 2) for row in batch]
    torch.testing.assert_close(loss, (individual[0] * 3 + individual[1] * 2) / 5)
    loss.backward()
    assert all(p.grad is not None for p in reader.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    torch.optim.AdamW(reader.parameters(), lr=0.01, weight_decay=0.0).step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(before[n], p)}
    prefix = "model.thinking_core.memory_reader."
    assert {prefix + n for n in ("scale_logit", "k_proj.weight", "v_proj.weight")} <= changed
    assert changed <= {prefix + n for n, _ in reader.named_parameters()}


def test_digit_supervision_scores_four_decimal_positions_and_credits_merged_tokens():
    from trainer.train_numeric_reader import DigitSupervision

    tok = AutoTokenizer.from_pretrained(fs.ROOT / "model")
    digits = DigitSupervision(tok, len(tok), "cpu")
    canonical = fs.encode_text(tok, "9458")
    assert len(canonical) < 4
    prefix, delim = [tok.bos_token_id], fs.encode_text(tok, "答案：")
    example = (prefix, delim, canonical + [tok.eos_token_id])
    q, d, labels = digits.digit_example(example)
    assert q == prefix and d == delim
    assert [tok.decode([token]) for token in labels] == list("9458")
    # At the second decimal position both "4" and "45" lead with the right digit.
    logits = torch.full((1, len(tok)), -torch.inf)
    logits[0, labels[1]] = 0.0
    logits[0, canonical[1]] = 0.0
    logits[0, labels[2]] = 0.0
    logits[0, tok.eos_token_id] = 0.0
    loss = digits.loss(logits, torch.tensor([labels[1]]))
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
    with pytest.raises(ValueError):
        digits.digit_example((prefix, delim, labels))


def test_auxiliary_digit_loss_adds_reader_gradients_without_unfreezing_backbone():
    from trainer.train_numeric_reader import DigitSupervision

    tok = AutoTokenizer.from_pretrained(fs.ROOT / "model")
    digits = DigitSupervision(tok, len(tok), "cpu")
    torch.manual_seed(47)
    model = MiniMindXForCausalLM(
        MiniMindXConfig(
            variant="minimind_x",
            residual_v2=True,
            hidden_size=32,
            vocab_size=len(tok),
            num_hidden_layers=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            max_loop=2,
            dropout=0.0,
        )
    )
    reader = freeze_for_reader(model)
    target = fs.encode_text(tok, "9458") + [tok.eos_token_id]
    batch = [(([1, 4, 5], [6], target), torch.randn(7, 32))]
    primary = reader_loss(model, batch, "cpu", 1)
    combined = reader_loss(model, batch, "cpu", 1, digits, 0.5)
    assert combined > primary
    (combined - primary).backward()
    assert reader.v_proj.weight.grad.norm().item() > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_delayed_short_reader_loss_matches_real_release_boundary():
    from model.minimind_x.controller import AdaptiveController
    from scripts.eval_counterfactual import Policy, decode
    from trainer.train_short_reader import encode_batch, delayed_logits

    tok = AutoTokenizer.from_pretrained(fs.ROOT / "model")
    torch.manual_seed(22)
    model = MiniMindXForCausalLM(MiniMindXConfig(
        variant="minimind_x", residual_v2=True, hidden_size=32, vocab_size=len(tok),
        num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, block_size=64, max_loop=2,
        memory_budgets=[64, 128, 256])).eval()
    row = dict(question="有效阈值是多少？", answers=["9458", "3296"],
               fact_ids=[fs.encode_text(tok, f"记录：当前阈值{value}。") for value in ["9458", "3296"]],
               before=fs.encode_text(tok, "旧阈值2730。"), after=[], release=1)
    examples, memories = zip(*encode_batch(model, tok, [row]))
    header = torch.tensor([[tok.bos_token_id, *fs.encode_text(tok, "资料：")]])
    with torch.no_grad():
        past = model(header, use_cache=True, num_loops=2, collect_routers=False).past_key_values
        hidden = torch.nn.utils.rnn.pad_sequence(memories, batch_first=True)
        valid = torch.arange(hidden.size(1))[None, :] < torch.tensor([len(x) for x in memories])[:, None]
        logits, labels = delayed_logits(model, examples, (hidden, valid), past)
        chunks = logits.split([len(x[-1]) for x in examples])
        assert labels.tolist() == examples[0][-1] + examples[1][-1]
        for row_idx, (question, delimiter, answer) in enumerate(examples):
            ids = torch.tensor(row["before"] + row["fact_ids"][row_idx])
            ctrl = fs.PinnedEvidence(model, block_visible_from=[1])
            ctrl.begin(ids)
            ctrl.gold_ids = [0]
            assert ctrl.pin_visible() is None
            policy = Policy(depth=2, initial_only=True)
            prefix = decode(AdaptiveController(model), header, None, policy, prefill=True)
            ctrl.store.advance_round(1)
            ctrl.pin_visible()
            actual, valid = ctrl._memory_input()
            torch.testing.assert_close(memories[row_idx], actual[valid])
            out = decode(ctrl, torch.tensor([question]), prefix.past_key_values, policy)
            for token in delimiter:
                out = decode(ctrl, torch.tensor([[token]]), out.past_key_values, policy)
            scores = [out.logits[0, -1]]
            for token in answer[:-1]:
                out = decode(ctrl, torch.tensor([[token]]), out.past_key_values, policy)
                scores.append(out.logits[0, -1])
            torch.testing.assert_close(chunks[row_idx], torch.stack(scores), rtol=2e-5, atol=2e-5)


def test_old_threshold_margin_compares_best_compatible_piece_with_full_vocabulary():
    from scripts.audit_old_threshold_reader import NumericTokens
    from trainer.train_old_threshold_reader import allowed_prefixes, prefix_hinge

    tok = AutoTokenizer.from_pretrained(fs.ROOT / "model")
    numeric = NumericTokens(tok, len(tok))
    prefix = fs.encode_text(tok, "9458") + [tok.eos_token_id]
    allowed = allowed_prefixes(numeric, prefix)
    assert allowed[-1] == []
    assert all(t in choices for t, choices in zip(prefix[:-1], allowed[:-1]))
    spaced = next(i for i in numeric.allowed("9458", first=True) if i in numeric.leading_space)
    assert spaced not in numeric.allowed("9458")
    assert tok.eos_token_id not in allowed[0]
    logits = torch.tensor([[0., 1., 4., 5., 0.], [0., 9., 0., 0., 0.]], requires_grad=True)
    loss, n = prefix_hinge(logits, [[1, 2]], [0])
    assert n == 1
    torch.testing.assert_close(loss, torch.tensor(2.))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0., 0., -1., 1., 0.], [0., 0., 0., 0., 0.]]))


def test_factorial_old_sample_layouts_are_nested_and_family_balanced():
    from collections import Counter
    from trainer.ablate_old_threshold_reader import assignment, layouts_for

    tok = AutoTokenizer.from_pretrained(fs.ROOT / "model")
    cases = [{"family": family, "old_ids": fs.encode_text(tok, "网关A-105旧阈值9458。")}
             for family in fs.FAMILIES for _ in range(64)]
    assigned = assignment(random.Random(4), cases)
    for family in fs.FAMILIES:
        sample = [row for row in assigned if row["family"] == family]
        assert Counter(row["low"] for row in sample) == dict(old=12, clean=26, neutral=26)
        assert Counter(row["high"] for row in sample) == dict(old=48, clean=8, neutral=8)
    low, high = [list(layouts_for(tok, assigned, ratio, 4)) for ratio in ("low", "high")]
    for left, right in zip(low, high, strict=True):
        if left["mode"] == right["mode"]:
            assert left["before"] == right["before"] and left["after"] == right["after"]
        else:
            assert right["mode"] == "old" and left["mode"] in ("clean", "neutral")
        if right["mode"] == "old":
            assert right["before"][-len(right["old_ids"]):] == right["old_ids"]
            assert not right["after"] and len(right["before"]) <= 16


def test_factorial_fixed_reader_width_keeps_valid_scores_and_gradients():
    from trainer.train_short_reader import batch_scores

    torch.manual_seed(42)
    model = MiniMindXForCausalLM(MiniMindXConfig(
        variant="minimind_x", residual_v2=True, hidden_size=32, vocab_size=80,
        num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, max_loop=2, dropout=0.0,
    ))
    reader = freeze_for_reader(model)
    rows = [(([1, 4, 5], [6], [13, 15, 2]), torch.randn(11, 32)),
            (([1, 4, 5], [6], [12, 2]), torch.randn(3, 32))]
    short, labels = batch_scores(model, rows, False, None)
    fixed, fixed_labels = batch_scores(model, rows, False, None, memory_width=28)
    assert torch.equal(labels, fixed_labels)
    torch.testing.assert_close(short, fixed)
    loss = torch.nn.functional.cross_entropy
    for left,right in zip(torch.autograd.grad(loss(short,labels),tuple(reader.parameters())),
                          torch.autograd.grad(loss(fixed,labels),tuple(reader.parameters()))):
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError, match="truncate"):
        batch_scores(model, rows, False, None, memory_width=8)


def test_intermediate_old_threshold_shares_preserve_and_nest_endpoint_assignments():
    from collections import Counter
    from trainer.ablate_old_threshold_reader import assignment, intermediate_shares

    cases = [{"family": family} for _ in range(64) for family in fs.FAMILIES]
    endpoints = assignment(random.Random(44), cases)
    expanded = intermediate_shares(endpoints, 44)
    middle = ("share_375", "share_500", "share_625")
    assert expanded == intermediate_shares(endpoints, 44)
    for base, row in zip(endpoints, expanded, strict=True):
        assert {k: v for k, v in row.items() if k not in middle} == base
        ordered = [row[k] for k in ("low", *middle, "high")]
        assert ordered == sorted(ordered, key=lambda mode: mode == "old")
        assert all(mode == base["low"] or mode == "old" for mode in ordered)
    for family in fs.FAMILIES:
        group = [r for r in expanded if r["family"] == family]
        for name, old in zip(middle, (24, 32, 40), strict=True):
            assert Counter(row[name] for row in group) == dict(
                old=old, clean=(64-old)//2, neutral=(64-old)//2)
