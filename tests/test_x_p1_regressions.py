"""P1 contracts on data, memory, trajectories, training semantics and evaluation."""
import copy
import json
import random
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from dataset.long_context_dataset import LongContextPretrainDataset
from dataset.retrieval_qa_dataset import RetrievalQADataset, encode_evidence
from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM, estimate_flops
from model.minimind_x.controller import AdaptiveController
from scripts.data_gen.gen_needle import gen_sample
from scripts.eval_needle import exact_match
from trainer.grpo_trajectory import trajectory, replay_logps
from trainer.oracle_labels import sweep_retrieval_action

ROOT = Path(__file__).resolve().parents[1]


def model(block=8, vocab=64):
    return MiniMindXForCausalLM(MiniMindXConfig(
        variant="minimind_x", residual_v2=True, vocab_size=vocab, hidden_size=32,
        num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
        block_size=block, max_loop=3, topk_coarse=4, memory_budgets=[block, block*2, block*3],
        max_position_embeddings=256, flash_attn=False)).eval()


def test_packed_stream_preserves_boundaries_tails_and_rank_ownership(tmp_path):
    tok = AutoTokenizer.from_pretrained(ROOT/'model')
    texts = [f"编号{i}。{'机器学习'*(1+i)}" for i in range(23)]
    file = tmp_path/'corpus'
    file.write_text(''.join(json.dumps(dict(text=t))+'\n' for t in texts))
    ds = LongContextPretrainDataset(file, tok, max_length=32, needle_ratio=0)
    want = [x for t in texts for x in [tok.bos_token_id, *tok(t, add_special_tokens=False).input_ids,
                                         tok.eos_token_id]]
    packed = [ds[i][1] for i in range(len(ds))]
    assert torch.cat([x[x != -100] for x in packed]).tolist() == want
    assert all(x.ne(-100).all() for x in packed[:-1])
    shards = [list(DistributedSampler(ds, num_replicas=2, rank=r, drop_last=True)) for r in (0, 1)]
    assert len(shards[0]) == len(shards[1]) and set(shards[0]).isdisjoint(shards[1])
    # Deterministic for a given (epoch, index), independent of fetch order / number of workers.
    loader = DataLoader(ds, batch_size=1, num_workers=1)
    for i, (_, labels, _) in enumerate(loader):
        assert torch.equal(labels[0], packed[i])


def test_tail_mask_survives_ranking_eviction_and_embedding_refresh():
    m = model()
    x = torch.arange(3, 21)
    ctrl = AdaptiveController(m)
    store = ctrl.begin(x)
    first = store.metadata['key_encoder_hash']
    q = m.model.prelude_hidden(x[:4][None])
    ctrl.memory = ctrl.retrieve_into_memory(q, 2)
    ids = ctrl.memory.source_ids[0]
    assert ctrl.stats['actual_valid_tokens'] == 18 and ctrl.stats['memory_tokens'] == 24
    torch.testing.assert_close(ctrl.memory.token_valid().reshape(3, 8), store.token_mask(ids))
    k = m.model.retriever.key_proj(m.model.embed_tokens(x[-2:]).mean(0))
    torch.testing.assert_close(store.block_keys[-1], k)
    # Padding attracted to the query must still lose the fine max-sim comparison.
    vec = m.model.retriever.project_query(q).detach()
    store._hidden_cache[2][:2] = -vec
    store._hidden_cache[2][2:] = vec
    res = m.model.retriever.retrieve(q, store, m)
    assert res.fine_scores[res.cand_idx == 2].item() == pytest.approx(-1)
    ctrl.retrieve_into_memory(q, 2)
    ctrl.memory = ctrl.rescore_warm_into_memory(q, 0, ctrl.memory)
    assert ctrl.stats['resident_source_ids'] == ctrl.memory.source_ids.flatten().tolist()
    assert ctrl.stats['unique_blocks'] == 1
    with torch.no_grad():
        m.model.embed_tokens.weight[x[-1]].add_(1)
    ctrl.refresh_index()
    assert store.metadata['key_encoder_hash'] != first
    assert not store._hidden_cache and ctrl.memory is None and not ctrl.has_warm_cache()
    assert store.metadata['tokenizer'] and store.metadata['model_version']


def test_sft_real_tokenized_sample_matches_append_only_decoding(tmp_path):
    tok = AutoTokenizer.from_pretrained(ROOT/'model')
    s = dict(context='旧记录。'*8+'当前编号37。', question='编号？', answer='37')
    f = tmp_path/'qa'
    f.write_text(json.dumps(s))
    row = RetrievalQADataset(f, tok, block_size=8)[0]
    torch.manual_seed(18)
    m = model(vocab=len(tok))
    ids, ctx = row['input_ids'][None], row['context_ids']
    prefix = int(row['question_len'])
    entry = AdaptiveController(m)
    out, ex = entry.forward_with_oracles(ids, None, row['labels'][None], ctx,
                                          query_len=prefix, oracle_retrieve=[2, 1], append_only=True)
    reference = AdaptiveController(m)
    reference.begin(ctx)
    query = m.model.prelude_hidden(ids[:, :prefix])
    reference.memory = reference.retrieve_into_memory(query, reference._budget_idx(query))
    past = None
    logits = []
    for start in [0, *range(prefix, ids.size(1))]:
        end = prefix if start == 0 else start + 1
        actions = [2, 1] if start == 0 else None
        step = m(ids[:, start:end], use_cache=True, past_key_values=past, num_loops=3,
                 loop_callback=reference.make_loop_callback(actions), memory_payload=reference._memory_input())
        logits.append(step.logits)
        past = step.past_key_values
    gold = row['labels'][1:]
    valid = gold != -100
    def scored(logits):
        return logits[0, :-1][valid].log_softmax(-1).gather(-1, gold[valid, None])
    torch.testing.assert_close(scored(out.logits), scored(torch.cat(logits, 1)), atol=1e-5, rtol=1e-5)
    assert ex['action_counts']['global_refresh'] >= 2
    actions, matrix = sweep_retrieval_action(m, ids, row['labels'][None], ctx, prefix,
                                              lambda: AdaptiveController(m), num_loops=1)
    assert actions.shape == (1, 0) and matrix.shape == (1, 0, 3)


def test_joint_replay_matches_each_sampled_action_and_updates_all_heads():
    torch.manual_seed(10)
    m = model()
    ids, ctx = torch.arange(3, 9)[None], torch.arange(3, 45)
    with torch.no_grad():
        records, old, reads, stats = trajectory(m, ids, ctx, 16, .65, None)
    kinds = {k for step in records for k, _, _ in step['events']}
    assert kinds == {'token','retrieval','budget','halt'}
    assert len({s['depth'] for s in records}) > 1 and reads and stats['scored_blocks'] > 0
    replayed = replay_logps(m, ids, ctx, records, .65, verify_behavior=True)
    torch.testing.assert_close(old, replayed)
    (-replayed.mean()).backward()
    for head in [m.lm_head, m.model.retrieve_router.action_head, m.model.budget_router, m.model.halt_router]:
        assert any(p.grad is not None and p.grad.norm() > 0 for p in head.parameters())
    broken = copy.deepcopy(records)
    broken[1]['depth'] = 10
    with pytest.raises(ValueError, match='depth'):
        replay_logps(m, ids, ctx, broken, .65)
    with pytest.raises(AssertionError):
        replay_logps(m, ids, ctx, records, 2., verify_behavior=True)


def test_cost_counts_real_reader_rounds_batch_tokens_and_cache():
    cfg = model().config
    empty = estimate_flops(cfg, 7, 3)
    assert empty['reader_attn'] == empty['reader_proj'] == empty['reader_gate'] == 0
    f = estimate_flops(cfg, 7, 3, memory_blocks=2, batch_size=2, indexed_blocks=5, encoded_blocks=3)
    assert f['memory_attn'] == 0 and f['reader_attn'] == 4*2*7*17*32*3
    assert f['candidate_encoding'] and f['index_build']
    assert estimate_flops(cfg, 14, 3)['update_gate'] == 2 * empty['update_gate']
    assert estimate_flops(cfg, 1, past_tokens=16)['backbone'] > estimate_flops(cfg, 1)['backbone']
    cfg.thinking_layers += 1
    assert estimate_flops(cfg, 7, 3, memory_blocks=2, batch_size=2)['reader_attn'] == f['reader_attn']


def test_multineedle_measures_requested_target_and_complete_answer():
    assert exact_match('３７', '37') and not exact_match('137', '37')
    assert not exact_match('37 或 34', '37')
    tok = AutoTokenizer.from_pretrained(ROOT/'model')
    starts = []
    for depth in (.1, .9):
        sample = gen_sample(random.Random(9), 512, [depth,.5], 2, 8, tok, target_needle=0)
        ids, gold, _ = encode_evidence(sample, tok, 8)
        starts.append(gold * 8 / len(ids))
    assert starts[0] < .2 and starts[1] > .75


def test_fixed_budget_baseline_and_explicit_external_cap(tmp_path):
    m = model()
    m.config.use_budget_router = False
    m.config.default_budget_idx = 1
    with torch.no_grad():
        m.model.budget_router.proj.weight.zero_()
        m.model.budget_router.proj.bias.copy_(torch.tensor([5., 0., 0.]))
    ctrl = AdaptiveController(m)
    query = torch.zeros(1, 1, m.config.hidden_size)
    assert ctrl._budget_idx(query) == 1
    m.config.use_budget_router = True
    assert ctrl._budget_idx(query) == 0
    tok = AutoTokenizer.from_pretrained(ROOT/'model')
    record = gen_sample(random.Random(3), 512, [.9], 1, 8, tok)
    path = tmp_path/'records'
    path.write_text(json.dumps(record))
    full = RetrievalQADataset(path, tok, block_size=8)[0]
    short = RetrievalQADataset(path, tok, block_size=8, max_context_tokens=128)[0]
    assert full['context_ids'].numel() > m.config.max_position_embeddings
    assert full['gold_block'] >= 0 and not full['gold_truncated']
    assert short['gold_block'] == -1 and short['gold_truncated']


def test_append_only_halt_loss_reaches_only_scored_decisions(monkeypatch):
    from types import SimpleNamespace
    from trainer import train_sft_x as sft

    torch.manual_seed(40)
    net = model()
    ids = torch.tensor([[3, 4, 5, 6, 7, 40, 41, 2]])
    # Two delimiter tokens, two answer tokens, then EOS.
    labels = ids.clone()
    labels[:, :5] = -100
    batch = dict(input_ids=ids, labels=labels, context_ids=torch.arange(8, 32)[None],
                 question_len=torch.tensor([3]), oracle_r=torch.tensor([3]),
                 oracle_retrieve=torch.tensor([[0, 0]]), gold_block=torch.tensor([1]))
    args = SimpleNamespace(device="cpu", oracle_rate=0., train_halt_router=True,
                           train_budget_router=False, lambda_distill=0.)
    calls, captured = [], {}
    joint = sft.joint_loss

    def capture_loss(*args, **kwargs):
        parts = joint(*args, **kwargs)
        captured.update(parts)
        return parts

    monkeypatch.setattr(sft, "joint_loss", capture_loss)
    with net.model.halt_router.register_forward_hook(lambda m, inputs, out: calls.append(out)):
        sft._sft_c_loss(net, batch, args)
    gradients = torch.autograd.grad(captured["halt"], calls, allow_unused=True)
    active = torch.tensor([g is not None and bool(g.abs().any()) for g in gradients])
    # Each column is a round; each row predicts the next token after that chunk.
    assert active.reshape(-1, 3).all(1).tolist() == [False, False, True, True, True, False]
    assert active.sum().item() == 9


@pytest.mark.parametrize("prefill_actions,visible", [([0, 0], [0, 0, 0]), ([2, 0], [0, 1, 1])])
def test_retrieval_losses_stay_with_question_event_across_decode_refreshes(prefill_actions, visible):
    torch.manual_seed(41)
    net = model()
    with torch.no_grad():
        net.model.retrieve_router.action_head.weight.zero_()
        net.model.retrieve_router.action_head.bias.copy_(torch.tensor([0., 0., 20.]))
    question, context = torch.tensor([[3, 4, 5]]), torch.arange(8, 32)

    def run(suffix):
        ctrl = AdaptiveController(net, block_visible_from=visible)
        ids = torch.cat([question, torch.tensor([suffix], dtype=torch.long)], 1)
        _, extras = ctrl.forward_with_oracles(ids, None, ids, context, gold_block_idx=torch.tensor([2]),
                                              oracle_retrieve=prefill_actions, query_len=3, append_only=True)
        return extras

    reference = run([])
    for answer in ([43, 44, 2], [45, 46, 2]):
        actual = run(answer)
        assert actual["action_counts"]["global_refresh"] > reference["action_counts"]["global_refresh"]
        for name in ("retrieval_ce", "distill_loss"):
            torch.testing.assert_close(actual[name], reference[name], atol=0, rtol=0)
            embed, proj = torch.autograd.grad(actual[name], (net.model.embed_tokens.weight,
                                                            net.model.retriever.query_proj.weight),
                                              retain_graph=True)
            assert proj.abs().sum() > 0 and embed[question].abs().sum() > 0
            assert embed[answer].count_nonzero() == 0


@pytest.mark.parametrize("policy,positions", [("fixed_cap", [3, 5, 6]), ("adaptive", [2, 3, 5, 6])])
def test_online_halt_targets_follow_the_next_token_at_each_prefix(policy, positions):
    from types import SimpleNamespace
    from trainer.train_sft_x import _halt_supervision

    # Position 4 predicts an ignored target. At position 6 no exit predicts the target.
    labels = torch.tensor([[-100, -100, -100, 1, 2, -100, 0, 2]])
    logits = [torch.zeros(1, 8, 3) for _ in range(3)]
    for pos, best_round in [(2, 2), (3, 1), (5, 3)]:
        logits[best_round - 1][0, pos, labels[0, pos + 1]] = 8.
    steps = [torch.arange(2, 8)[None] + 10*r for r in range(3)]
    out = SimpleNamespace(exit_logits=logits, router_probs={"halt_by_step": steps})
    cfg = SimpleNamespace(prefill_policy=policy, oracle_epsilon=0.05)
    probs, depth, valid = _halt_supervision(out, labels, 3, cfg, file_depth=0)
    for r, p in enumerate(probs):
        assert p.tolist() == [pos + 10*r for pos in positions]
    expected_depth = {2: 2, 3: 1, 5: 3, 6: 1}
    assert depth.tolist() == [expected_depth[pos] for pos in positions]
    assert valid.tolist() == [pos != 6 for pos in positions]
    assert _halt_supervision(out, torch.full_like(labels, -100), 3, cfg, 0) == (None, None, None)


def test_decode_only_evidence_does_not_create_question_retrieval_loss():
    net = model()
    with torch.no_grad():
        net.model.retrieve_router.action_head.weight.zero_()
        net.model.retrieve_router.action_head.bias.copy_(torch.tensor([0., 0., 20.]))
    ids = torch.tensor([[3, 4, 5, 40, 41, 2]])
    ctrl = AdaptiveController(net, block_visible_from=[1, 1, 1])
    _, extra = ctrl.forward_with_oracles(ids, None, ids, torch.arange(8, 32), query_len=3,
                                         gold_block_idx=torch.tensor([1]), append_only=True,
                                         oracle_retrieve=[0, 0])
    assert ctrl.has_warm_cache() and ctrl.memory.size > 0
    assert "retrieval_ce" not in extra and "distill_loss" not in extra


def test_sft_b_full_unroll_returns_per_exit_metrics(monkeypatch):
    from types import SimpleNamespace
    from trainer import train_sft_x as sft

    net = model()
    monkeypatch.setattr(sft, "model", net, raising=False)
    monkeypatch.setattr(sft, "args", SimpleNamespace(full_unroll_ratio=1., alpha_consistency=.1),
                        raising=False)
    ids = torch.arange(3, 10)[None]
    loss, info = sft.sft_b_step(ids, ids)
    assert torch.isfinite(loss) and len(info["per_exit"]) == net.config.max_loop


def test_sft_c_reports_online_depth_per_token_even_with_halt_training_disabled():
    from types import SimpleNamespace
    from trainer.train_sft_x import _sft_c_loss

    torch.manual_seed(42)
    net = model()
    ids, ctx = torch.arange(3, 11)[None], torch.arange(8, 32)
    # At two decision positions at least one exit is correct; at the third all exits miss.
    labels = torch.full_like(ids, -100)
    ctrl = AdaptiveController(net)
    with torch.no_grad():
        out, _ = ctrl.forward_with_oracles(ids, None, labels, ctx, oracle_retrieve=[0, 0],
                                           query_len=3, append_only=True)
        labels[:, 5] = out.exit_logits[0][:, 4].argmax(-1)
        labels[:, 6] = out.exit_logits[-1][:, 5].argmax(-1)
        predictions = {int(lg[:, 6].argmax(-1)) for lg in out.exit_logits}
        labels[:, 7] = next(i for i in range(net.config.vocab_size) if i not in predictions)
    batch = dict(input_ids=ids, labels=labels, context_ids=ctx[None], gold_block=torch.tensor([-1]),
                 question_len=torch.tensor([3]), oracle_r=torch.tensor([0]),
                 oracle_retrieve=torch.tensor([[0, 0]]))
    args = SimpleNamespace(device="cpu", train_budget_router=False, oracle_rate=1., lambda_distill=0.)
    for enabled in (True, False):
        args.train_halt_router = enabled
        loss, info = _sft_c_loss(net, batch, args)
        grad, = torch.autograd.grad(loss, net.model.halt_router.proj.weight, allow_unused=True)
        assert (grad is not None and bool(grad.abs().any())) == enabled
        assert (info["depth_assessed"], info["unresolved"]) == (3, 1)
