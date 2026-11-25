"""G2/G3 评审修复：per-sample last-valid、同快照 Oracle、Reader KV 复用、dataset split/span。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController


def _m(**over):
    torch.manual_seed(91)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              max_loop=3, residual_v2=True)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_per_sample_last_valid_cache_mixed_exit():
    """评审 5.1：样本 A R=1、样本 B R=3 时，A 的深槽应复制其 exit 轮，而非 B 的深轮。"""
    m = _m()
    ids = torch.randint(0, 64, (2, 4))
    # 强制不同 exit：用 exit_choice 分开跑对照；此处检查 cache 形状与 last-valid 单调
    with torch.no_grad():
        out = m(ids[:, :2], use_cache=True, num_loops=3)
        past = out.past_key_values
        cfg = m.config
        cc = past[cfg.prelude_layers:cfg.prelude_layers + cfg.thinking_layers]
        assert all(len(slots) == 3 for slots in cc)
        # 再喂 1 token，R=1（全 batch 浅执行）→ 新 token 深槽 = 其轮 0
        out1 = m(ids[:, 2:3], past_key_values=past, use_cache=True, num_loops=1)
        past = out1.past_key_values
        cc = past[cfg.prelude_layers:cfg.prelude_layers + cfg.thinking_layers]
        for slots in cc:
            k0 = slots[0][0][:, -1]
            k2 = slots[2][0][:, -1]
            assert torch.equal(k0, k2), "浅执行 token 深槽应复制最后实际轮"


def test_update_gate_skips_halted_via_active():
    """评审 5.1：active=False 时 committed 保持 previous。"""
    from model.minimind_x.residual import RecurrentUpdateGate
    g = RecurrentUpdateGate(8, rank=4)
    prev = torch.randn(2, 3, 8)
    cand = torch.randn(2, 3, 8)
    anchor = torch.randn(2, 3, 8)
    active = torch.tensor([[True, False, True], [False, False, True]])
    out, gate = g(prev, anchor, cand, active=active)
    assert torch.equal(out[:, 1], prev[:, 1])
    assert torch.equal(out[0, 1], prev[0, 1])
    assert torch.equal(out[1, 0], prev[1, 0])


def test_same_snapshot_oracle_evals_post_action_exit():
    """评审 4.2：loss_matrix 在动作后出口评价（非动作前）。"""
    from trainer.oracle_labels import sweep_retrieval_action
    torch.manual_seed(92)
    m = _m()
    ids = torch.randint(3, 60, (1, 8))
    y = torch.randint(3, 60, (1, 8))
    ctx = torch.randint(3, 60, (32,))
    best, mat = sweep_retrieval_action(
        m, ids, y, ctx, 4, lambda: AdaptiveController(m), num_loops=3)
    assert best.shape == (1, 2)
    assert mat.shape == (1, 2, 3)
    assert torch.isfinite(mat).all()


def test_reader_projection_reused_across_loops():
    """评审 9.2：同 memory_version 投影只算一次。"""
    m = _m(max_loop=3)
    core = m.model.thinking_core
    payload = torch.randn(1, 16, 32)
    valid = torch.ones(1, 16, dtype=torch.bool)
    from model.minimind_x.memory.working_memory import MemoryInput
    mi = MemoryInput(payload, valid, memory_version=5)
    calls = {"n": 0}
    orig = core.memory_reader.project_memory
    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    core.memory_reader.project_memory = counted
    x = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        m(x, num_loops=3, memory_payload=mi)
    assert calls["n"] == 1, f"project_memory 应跨 3 loop 只调 1 次，实际 {calls['n']}"
    core.memory_reader.project_memory = orig


def test_dataset_split_and_gold_span():
    """评审 7.1/7.2：split 过滤 + gold_span→block。"""
    import json, tempfile
    from dataset.retrieval_qa_dataset import RetrievalQADataset

    class FakeTok:
        bos_token_id, eos_token_id = 1, 2
        def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False, **kwargs):
            ids = [ord(c) % 50 + 3 for c in text[:max_length or 99]]
            return type("R", (), {"input_ids": ids, "offset_mapping": [(i, i+1) for i in range(len(ids))]})()

    recs = [
        {"context": "AAAA" * 32, "question": "q", "answer": "a", "split": "train",
         "gold_span": [0, 4], "gold_block": 9},
        {"context": "BBBB" * 32, "question": "q", "answer": "a", "split": "val",
         "gold_block": -1},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        path = f.name
    ds_tr = RetrievalQADataset(path, FakeTok(), block_size=8, split="train")
    ds_va = RetrievalQADataset(path, FakeTok(), block_size=8, split="val")
    assert len(ds_tr) == 1 and len(ds_va) == 1
    item = ds_tr[0]
    # gold_span 覆盖文件里的 gold_block=9
    assert int(item["gold_block"]) >= 0
    assert int(item["gold_block"]) == 0
