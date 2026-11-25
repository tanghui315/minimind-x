import json
import random
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from dataset.retrieval_qa_dataset import encode_evidence
from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from scripts.data_gen.gen_needle import gen_sample
from trainer.x_utils import load_x_for_eval, save_x_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def tiny_model():
    return MiniMindXForCausalLM(MiniMindXConfig(
        residual_v2=True, hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=64, max_loop=3, block_size=8, topk_coarse=8,
        memory_budgets=[8, 16, 32], max_position_embeddings=128))


def test_halt_stats_logits_and_decisions_use_visible_tokens():
    torch.manual_seed(7)
    model = tiny_model().eval()
    with torch.no_grad():
        model.model.halt_router.stats_proj.weight.fill_(1)
    x = torch.randint(3, 64, (2, 12))
    y = x.clone()
    y[:, 5:] = 3
    y[:, 0] = 4
    mask = torch.ones_like(x)
    mask[:, 0] = 0
    for pool in ("mean", "last"):
        model.model.halt_router.pool = pool
        def run(ids, halt=False):
            return model(ids, attention_mask=mask, query_len=5, collect_exits=True, halt=halt)
        a, b = run(x), run(y)
        for left, right in zip(a.residual_stats, b.residual_stats):
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=1e-6)
        for left, right in zip(a.halt_logits, b.halt_logits):
            torch.testing.assert_close(left, right, rtol=0, atol=1e-6)
        torch.testing.assert_close(run(x, True).exit_idx, run(y, True).exit_idx)


def test_delayed_evidence_excluded_before_topk_and_from_supervision():
    torch.manual_seed(5)
    model = tiny_model().eval()
    x = torch.arange(4, 12).unsqueeze(0)
    context = torch.arange(3, 27)
    changed = context.clone()
    changed[8:] = 44
    rounds = [0, 1, 1]
    def forward(ctx, actions):
        ctrl = AdaptiveController(model, block_visible_from=rounds)
        out, extras = ctrl.forward_with_oracles(x, None, x, ctx, gold_block_idx=torch.tensor([2]),
                                                query_len=4, oracle_retrieve=actions)
        return ctrl, out, extras
    early, a, extras = forward(context, [0, 0])
    _, b, _ = forward(changed, [0, 0])
    assert early.memory.source_ids.tolist() == [[0]]
    assert "retrieval_ce" not in extras
    torch.testing.assert_close(a.logits, b.logits)
    later, _, extras = forward(context, [2, 0])
    assert sorted(later.warm_candidates["cand_idx"][0].tolist()) == [0, 1, 2]
    assert torch.isfinite(extras["retrieval_ce"])
    # Initial retrieval can legally see no blocks at all.
    empty = AdaptiveController(model, block_visible_from=[1, 1, 1])
    out, _ = empty.forward_with_oracles(x, None, x, context, query_len=4, oracle_retrieve=[2, 0])
    assert out.router_trace[0]["valid_tokens"] == 0 and empty.memory.size > 0


def test_needle_lengths_spans_and_legacy_recovery():
    tok = AutoTokenizer.from_pretrained(ROOT / "model")
    for depth, count in [(0.1, 1), (0.5, 3), (0.9, 1), (1.0, 2)]:
        sample = gen_sample(random.Random(7), 8192, [depth], count, 256, tok)
        ids, gold, _ = encode_evidence(sample, tok, 256)
        assert len(ids) == 8192 and gold == sample["gold_block"]
        start, end = sample["gold_span"]
        assert sample["answer"] in sample["context"][start:end]
        enc = tok(sample["context"], add_special_tokens=False, return_offsets_mapping=True)
        hits = [i for i, (a, b) in enumerate(enc.offset_mapping) if a < end and b > start]
        assert gold == ((hits[0] + hits[-1]) // 2) // 256
        legacy = dict(sample, gold_block=999)
        del legacy["gold_span"]
        assert encode_evidence(legacy, tok, 256)[1] == gold
        _, truncated_gold, rounds = encode_evidence(dict(sample, evidence_visible_from_round=2),
                                                    tok, 256, max_tokens=hits[0]+1)
        assert truncated_gold == -1 and rounds[-1] == 2


def test_checkpoint_config_strict_load_and_context_decoding(tmp_path):
    torch.manual_seed(1)
    model = tiny_model().eval()
    path = save_x_checkpoint(model, tmp_path, "roundtrip", 32)
    restored = load_x_for_eval(tmp_path, "roundtrip")
    x = torch.arange(3, 11).unsqueeze(0)
    ctx = torch.arange(5, 53)
    def run(m):
        ctrl = AdaptiveController(m)
        out = ctrl.generate_with_context(x, context_ids=ctx, max_new_tokens=2, do_sample=False,
                                         fixed_loops=2, eos_token_id=None, return_stats=True)
        return out["generated_ids"], ctrl.memory.source_ids, m(x, num_loops=2).logits
    a, b = run(model), run(restored)
    for left, right in zip(a[:2], b[:2]):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(a[2], b[2], rtol=1e-2, atol=2e-3)  # FP16 checkpoint
    payload = torch.load(path, weights_only=False)
    del payload["model.thinking_core.update_gate.fc2.weight"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="update_gate.fc2.weight"):
        load_x_for_eval(tmp_path, "roundtrip")


def distributed_worker(rank, rendezvous):
    from trainer.train_sft_x import _sft_c_loss
    from trainer import train_grpo_x as grpo
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(51)
        model = tiny_model()
        opts = SimpleNamespace(phase="c", device="cpu", oracle_rate=1., train_budget_router=1,
                               train_halt_router=1, lambda_distill=.05)
        train = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        before = model.model.retriever.query_proj.weight.detach().clone()
        for step in range(2):
            ids = torch.arange(5, 15).unsqueeze(0) + rank
            has_context = rank == step
            batch = dict(input_ids=ids, labels=ids, question_len=torch.tensor([4]),
                         context_ids=torch.arange(32).view(1, -1) if has_context else ids[:, :0],
                         oracle_r=torch.tensor([0]), oracle_retrieve=torch.empty(1, 0, dtype=torch.long),
                         gold_block=torch.tensor([2 if has_context else -1]))
            loss, _ = _sft_c_loss(train, batch, opts)
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            grad = model.model.retriever.query_proj.weight.grad
            assert grad is not None and grad.norm() > 0
            other = grad.clone()
            dist.broadcast(other, 0)
            torch.testing.assert_close(other, grad)
            opt.step()
        assert not torch.equal(before, model.model.retriever.query_proj.weight)
        # New trainer stage, as in independent torchrun launches.
        policy = tiny_model()
        grpo.model = DDP(policy, find_unused_parameters=True, broadcast_buffers=False)
        from trainer.oracle_labels import sweep_depth
        q = torch.tensor([[3, 4, 5]])
        assert sweep_depth(grpo.model, q, q)[0].shape == (1, policy.config.max_loop)
        with_memory = AdaptiveController(grpo.model).generate_with_context(
            q, context_ids=q.flatten(), max_new_tokens=1, do_sample=False)
        assert with_memory["generated_ids"].shape == (1, 4)
        grpo.lm_config = policy.config
        grpo.args = SimpleNamespace(mode_diversity=1, adaptive_rollout=0, device="cpu", temperature=1.0)
        class Tokenizer:
            eos_token_id = None
            def __call__(self, text, **kwargs):
                return {"input_ids": [3, 4 + rank, 5]}
            def decode(self, ids, **kwargs):
                return str(ids)
        grpo.tokenizer = Tokenizer()
        roll = grpo.rollout_group("q", 2, 2)
        assert roll["loop"] == [1, 2]
        for tokens, r in zip(roll["ids"], roll["loop"]):
            trajectory = torch.cat((torch.tensor([3, 4 + rank, 5]), tokens)).unsqueeze(0)
            lp = grpo.compute_per_token_logps(grpo.model, trajectory, 2, num_loops=int(r))
            (-lp.mean()).backward()
    finally:
        dist.destroy_process_group()


def test_two_rank_sft_and_grpo(tmp_path):
    mp.spawn(distributed_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2)


def test_ppl_uses_exponentiated_token_nll(tmp_path):
    from scripts.eval_suite import eval_ppl
    class Uniform(torch.nn.Module):
        def forward(self, ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros(*ids.shape, 64))
    corpus = tmp_path / "text.jsonl"
    corpus.write_text('{"text":"abcdefgh"}')
    def tokenize(text, **kwargs):
        return SimpleNamespace(input_ids=list(range(len(text))))
    assert eval_ppl(Uniform(), tokenize, corpus) == pytest.approx(64, rel=1e-6)


def test_explicit_x_checkpoint_serves_chat(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from fastapi.testclient import TestClient
    from scripts import serve_openai_api as server
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, residual_v2=True, max_loop=2)
    path = save_x_checkpoint(MiniMindXForCausalLM(cfg), tmp_path, "api", 32)
    monkeypatch.setattr(server, "device", "cpu", raising=False)
    network, tok = server.init_model(SimpleNamespace(load_from=str(ROOT / "model"), x_checkpoint=path,
                                                      save_dir=tmp_path, lora_weight="None"))
    assert network.config.residual_v2
    monkeypatch.setattr(server, "model", network, raising=False)
    monkeypatch.setattr(server, "tokenizer", tok, raising=False)
    client = TestClient(server.app)
    req = {"model": "minimind-x", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 2, "stream": False}
    response = client.post("/v1/chat/completions", json=req)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["role"] == "assistant"
    response = client.post("/v1/chat/completions", json=dict(req, stream=True))
    assert response.status_code == 200 and '"finish_reason": "stop"' in response.text and '"error"' not in response.text


def test_evidence_splits_are_disjoint_even_with_identical_seeds(tmp_path, monkeypatch):
    from scripts.data_gen import gen_evidence_families
    path = tmp_path / "families.jsonl"
    monkeypatch.setattr(sys, "argv", ["gen", "--out", str(path), "--n_per_family", "5",
                                      "--seed_train", "12", "--seed_val", "12"])
    gen_evidence_families.main()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for key in ("deployment_id", "question", "context"):
        train = {r[key] for r in rows if r["split"] == "train" and r[key] != ""}
        val = {r[key] for r in rows if r["split"] == "val" and r[key] != ""}
        assert train and val and train.isdisjoint(val)
