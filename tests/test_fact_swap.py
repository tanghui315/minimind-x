import json
import sys

import pytest
import torch
from transformers import AutoTokenizer

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from scripts import eval_fact_swap as fs
from trainer.x_utils import save_x_checkpoint


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(fs.ROOT / "model")


def record(family="evidence_needed"):
    if family == "multi_hop":
        fact = "链路表：网关A 的上行通道绑定到 网关B。"
        other = "配置记录：网关C 的阈值设定为 1382。配置记录：网关B 的阈值设定为 1382。"
    elif family == "new_evidence":
        fact = "补充约束（生效中）：网关A 的实际阈值为 1382，覆盖早期文档值。"
        other = "早期文档：网关A 的阈值为 49（已废弃）。"
    else:
        fact = "配置记录：网关A 本次部署的阈值设定为 1382。"
        other = ""
    return dict(
        question="当前阈值？",
        answer="1382",
        family=family,
        split="val",
        context=other + fact,
        gold_span=[len(other), len(other) + len(fact)],
        evidence_visible_from_round=int(family == "new_evidence"),
    )


def tiny(tok):
    torch.manual_seed(551)
    return MiniMindXForCausalLM(
        MiniMindXConfig(
            variant="minimind_x",
            residual_v2=True,
            vocab_size=len(tok),
            hidden_size=32,
            num_hidden_layers=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            block_size=8,
            max_loop=2,
            memory_budgets=[128, 256, 512],
            max_position_embeddings=256,
        )
    ).eval()


@pytest.mark.parametrize("family", fs.FAMILIES)
def test_numeric_edit_is_local_and_tokenizer_matched(tok, family):
    sample = record(family)
    facts = fs.locate_facts(sample)
    edited = fs.counterfactual(sample, facts, tok, 21)
    s, e = facts.value_span
    assert edited["context"] == sample["context"][:s] + edited["answer"] + sample["context"][e:]
    assert edited["question"] == sample["question"] and len(
        fs.encode_text(tok, edited["answer"])
    ) == len(fs.encode_text(tok, sample["answer"]))
    assert edited == fs.counterfactual(sample, facts, tok, 21)
    assert all(a != b for a, b in zip(edited["answer"], sample["answer"]))
    if family == "multi_hop":
        assert len(facts.spans) == 2 and "网关C 的阈值设定为 1382" in edited["context"]
        assert "网关B 的阈值设定为 " + edited["answer"] in edited["context"]
        ambiguous = dict(sample, context=sample["context"] + "配置记录：网关B 的阈值设定为 6666。")
        with pytest.raises(ValueError, match="exactly one"):
            fs.locate_facts(ambiguous)
    with pytest.raises(ValueError, match="disagrees"):
        fs.locate_facts(dict(sample, answer="9999"))


@torch.no_grad()
def test_pinned_blocks_wait_for_release_and_preserve_tail(tok):
    model = tiny(tok)
    s = record("new_evidence")
    ctrl, _, gold, _ = fs.build_arm(model, tok, s, fs.locate_facts(s), "gold_blocks", 0)
    assert len(gold) > 1 and fs.evidence(ctrl) == [] and ctrl._memory_input() is None
    cb = ctrl.make_loop_callback()
    payload = cb(0, None)
    assert fs.evidence(ctrl) == gold and ctrl.memory.memory_version == 1
    assert int(payload.valid_mask.sum()) == sum(
        min(8, ctrl.store.context_ids.numel() - idx * 8) for idx in gold
    )
    assert cb(1, None) is None and ctrl.memory.memory_version == 1


@torch.no_grad()
def test_reader_intervention_and_zero_scale_negative_control(tok):
    m = tiny(tok)
    a = record("multi_hop")
    facts = fs.locate_facts(a)
    b = fs.counterfactual(a, facts, tok, 29)
    targets = [fs.encode_text(tok, r["answer"]) for r in (a, b)]

    def worlds(mode):
        return [fs.run_world(m, tok, r, facts, mode, targets, 4, 0) for r in (a, b)]

    original, edited = worlds("gold_blocks")
    for w in (original, edited):
        assert w["all_gold_present"]
        assert any(c["relative_l2"] > 0 for c in w["first_answer_reader_calls"])
    assert original["valid_payload_sha256"] != edited["valid_payload_sha256"]
    m.model.thinking_core.memory_reader.scale_logit.zero_()
    no_a, no_b = worlds("no_memory")
    gold_a, gold_b = worlds("gold_sentences")
    assert no_a["token_logps"] == no_b["token_logps"] == gold_a["token_logps"] == gold_b["token_logps"]
    assert gold_a["generated_ids"] == gold_b["generated_ids"]
    assert gold_a["valid_payload_sha256"] != gold_b["valid_payload_sha256"]
    assert all(c["relative_l2"] == 0 for c in gold_a["first_answer_reader_calls"])


def test_tracking_requires_both_answers_and_paired_preference():
    def outcome(prediction, lp):
        return dict(
            prediction=prediction,
            mean_logps=lp,
            valid_payload_sha256=None,
            generated_ids=[ord(prediction[0])],
        )

    a = outcome("1382", [-1.0, -2.0])
    b = outcome("9999", [-3.0, -1.0])
    tracked = fs.pair_metrics(a, b, "1382", "9999")
    assert tracked["tracked"] and tracked["preference_flipped"] and tracked["margin_shift"] == 3.0
    result = fs.pair_metrics(a, outcome("99", [-4.0, -5.0]), "1382", "9999")
    assert result["output_changed"] and not result["tracked"] and not result["preference_flipped"]


@pytest.mark.parametrize("zero_scale", [False, True])
def test_cli_evaluates_only_eligible_validation_pairs(tok, tmp_path, monkeypatch, zero_scale):
    path = save_x_checkpoint(tiny(tok), tmp_path, "test", 32)
    data = tmp_path / "qa.jsonl"
    data.write_text(
        "\n".join(map(json.dumps, [dict(split="train"), dict(family="no_evidence"), record()]))
    )
    result = tmp_path / "results.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swap",
            "--checkpoint",
            str(path),
            "--data",
            str(data),
            "--device",
            "cpu",
            "--out",
            str(result),
            "--max-new-tokens",
            "6",
            *(["--zero-reader-scale"] if zero_scale else []),
        ],
    )
    fs.main()
    r = json.loads(result.read_text())
    assert r["skipped"] == dict(other_split=1, no_evidence=1)
    assert len(r["samples"]) == 1 and set(r["summary"]) == set(fs.ARMS)
    assert r["summary"]["gold_blocks"]["reader_nonzero_in_both"] == int(not zero_scale)
    if zero_scale:
        assert r["summary"]["gold_blocks"]["payload_changed"] == 1
        assert r["summary"]["gold_blocks"]["max_abs_shift"] == 0
    assert r["summary"]["no_memory"]["max_abs_shift"] == 0


@pytest.mark.parametrize("late", [False, True])
@torch.no_grad()
def test_matched_channels_deliver_same_ids_and_observe_release(tok, late):
    from scripts.eval_paired_reader import run_channel

    net = tiny(tok).requires_grad_(False)
    source = torch.arange(100, 120)
    blocks = [0, 2]  # non-adjacent; the second block has four valid tokens
    visible = torch.tensor([int(late), 0, int(late)])
    ids = source[:8].tolist() + source[16:].tolist()
    expected = net.model.encode_context_blocks(source, block_idxs=blocks)
    expected = torch.cat([expected[0], expected[1, :4]])
    sample = record()
    header = [tok.bos_token_id, *fs.encode_text(tok, "资料：")]
    suffix = fs.encode_text(tok, "\n问题：") + fs.encode_text(tok, sample["question"])
    records = {}
    for channel in ("reader", "prompt"):
        calls = []

        def observe(module, args, kw):
            assert kw["num_loops"] == 2 and not kw["halt"] and kw["loop_callback"] is None
            memory = kw["memory_payload"]
            payload = None if memory is None else memory[0][memory[1]].clone()
            calls.append((args[0].flatten().tolist(), payload))

        with net.register_forward_pre_hook(observe, with_kwargs=True):
            records[channel] = run_channel(
                net, tok, sample, source, blocks, visible, ids, channel, 2, 3)
        if late:
            first, memory = calls.pop(0)
            assert first == header and memory is None
        prefix = [] if late else header
        assert calls[0][0] == prefix + (ids if channel == "prompt" else []) + suffix
        for _, payload in calls:
            if channel == "reader":
                torch.testing.assert_close(payload, expected, rtol=0, atol=0)
            else:
                assert payload is None
    assert records["prompt"]["delivered_sha256"] == records["reader"]["delivered_sha256"]


def test_rotary_extension_preserves_native_outputs_and_parameters(tok):
    from scripts.eval_paired_reader import extend_positions, state_digest

    net = tiny(tok).eval().requires_grad_(False)
    digest = state_digest(net)
    prefix = net.model.freqs_cos.clone(), net.model.freqs_sin.clone()
    ids = torch.arange(4, 13)[None]
    with torch.inference_mode():
        before = net(ids, num_loops=2).logits
        extend_positions(net, 512)
        torch.testing.assert_close(before, net(ids, num_loops=2).logits, rtol=0, atol=0)
    for actual, old in zip((net.model.freqs_cos, net.model.freqs_sin), prefix):
        assert len(actual) == 512 and torch.equal(actual[:len(old)], old)
    assert state_digest(net) == digest


def test_channel_summary_splits_both_worlds_and_discordant_pairs():
    from scripts.eval_paired_reader import summarize

    def row(p, r):
        def world(i):
            form = {name: {"correct": values[i]} for name, values in
                    (("prompt", p), ("reader", r), ("reader_bare", r))}
            return {"sentence": form, "block": form}
        return {"original": world(0), "edited": world(1)}

    rows = [row((1, 1), (1, 1)), row((1, 1), (1, 0)), row((1, 0), (1, 1)), row((1, 0), (0, 1))]
    for counts in summarize(rows).values():
        assert counts["prompt"] == counts["reader"] == 2
        assert [counts[k] for k in ("both", "prompt_only", "reader_only", "neither")] == [1, 1, 1, 1]


@pytest.mark.parametrize("family", fs.FAMILIES)
def test_distractor_pairs_preserve_fact_positions_and_isolate_slot(tok, family):
    from scripts.eval_reader_distractors import make_pair

    sample = record(family)
    pad = "遥测稳定。" * 7
    sample["context"] = pad + sample["context"] + pad
    sample["gold_span"] = [i + len(pad) for i in sample["gold_span"]]
    facts = fs.locate_facts(sample)
    edited = fs.counterfactual(sample, facts, tok, 773)
    prior = {"original_answer": sample["answer"], "edited_answer": edited["answer"], "line": 7}
    for world, rec in (("original", sample), ("edited", edited)):
        ctx, blocks, _ = fs.select_evidence(tok, rec, facts, "gold_blocks", 128, "cpu")
        prior[world] = {"block": {"block_ids": blocks, "evidence_ids": [
            t for b in blocks for t in ctx[b*128:(b+1)*128].tolist()]}}
    info, designs = make_pair(tok, sample, prior, 128)
    assert info["entity"] == ("网关B" if family == "multi_hop" else "网关A")
    assert (info, designs) == make_pair(tok, sample, prior, 128)
    l, h = info["slot_span"]
    lo, hi = info["target_span"]
    neutral = designs["neutral"][0]
    number = fs.encode_text(tok, info["decoy"])
    pos = info["decoy_start"]
    for arm, (orig, edit) in designs.items():
        for ids, world in [(orig, "original"), (edit, "edited")]:
            ref = prior[world]["block"]["evidence_ids"]
            assert len(ids) == len(ref)
            for a,b in info["fact_spans"]:
                assert ids[a:b] == ref[a:b] and (h <= a or b <= l)
        assert orig[:lo] == edit[:lo] and orig[hi:] == edit[hi:] and orig[lo:hi] != edit[lo:hi]
        assert orig[:l] == neutral[:l] and orig[h:] == neutral[h:]
        if arm == "neutral":
            assert not any(c.isdigit() for c in tok.decode(orig[l:h]))
        else:
            assert orig[pos:pos+len(number)] == number and tok.decode(orig).count(info["decoy"]) == 1
    numeric = designs["unrelated_count"][0]
    assert neutral[:pos] == numeric[:pos] and neutral[pos+len(number):] == numeric[pos+len(number):]
    assert info["entity"] in tok.decode(designs["obsolete_threshold"][0][l:h])
    prior["original"]["block"]["evidence_ids"][0] += 1
    with pytest.raises(ValueError, match="prior evaluation"):
        make_pair(tok, sample, prior, 128)


@pytest.mark.parametrize("fact_length", [24, 31, 53])
def test_length_position_layouts_hold_the_other_axis_constant(tok, fact_length):
    from scripts.eval_reader_length_position import ADDED, SEEDS, TOTALS, layouts

    cells = {cell:(left,right) for cell,left,right in layouts(tok,fact_length,31,"网关B-143","1232")}
    assert cells["clean"] == ([], [])
    for n in ADDED[1:]:
        assert cells[f"length/{SEEDS[0]}/{n}"] != cells[f"length/{SEEDS[1]}/{n}"]
    for seed in SEEDS:
        prev = []
        for n in ADDED[1:]:
            left,right = cells[f"length/{seed}/{n}"]
            assert not right and len(left) == n and left[n-len(prev):] == prev
            assert not any(c.isdigit() for c in tok.decode(left))
            prev = left
        for total in TOTALS:
            head,mid,tail = [cells[f"position/{seed}/{total}/{pos}"] for pos in ("head","middle","tail")]
            assert not head[0] and not tail[1] and head[1] == mid[0]+mid[1] == tail[0]
            assert len(head[1])+fact_length == total
            assert abs(len(mid[0])-len(mid[1])) <= 1
            assert not any(c.isdigit() for c in tok.decode(head[1]))
    for pos,idx in (("before",0),("after",1)):
        count = cells[f"short/{pos}/entity_count"]
        old = cells[f"short/{pos}/obsolete_threshold"]
        assert count[1-idx] == old[1-idx] == []
        assert len(count[idx]) == len(old[idx])
        for tokens in (count[idx],old[idx]):
            assert "网关B-143" in tok.decode(tokens) and "1232" in tok.decode(tokens)
        assert tok.decode(count[idx]).replace("采样计数","旧阈值") == tok.decode(old[idx])


def test_position_scan_counts_loss_and_gain_on_different_worlds():
    from scripts.eval_reader_length_position import transitions

    def case(line,a,b):
        return dict(line=line,cells={arm:{w:{"reader":{"correct":v}} for w,v in
                                         zip(("original","edited"),pair)} for arm,pair in (("c",a),("d",b))})
    rows = [case(7,[True,True],[True,False]),case(8,[False,True],[True,True])]
    result = transitions(rows,"c","d","reader")
    assert result == dict(base_correct=3, losses=[[7,"edited"]], gains=[[8,"original"]], base_pairs=1,
                          pair_losses=[7], pair_gains=[8])
