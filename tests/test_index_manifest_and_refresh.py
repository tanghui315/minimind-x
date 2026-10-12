"""P0 Index Manifest / P1 key refresh / P2 query-encoder 路径口径。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.controller import AdaptiveController
from model.minimind_x.retrieval import LongContextStore


def _model(**over):
    torch.manual_seed(70)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2,
              freeze_key=True, key_refresh_interval=0)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def test_index_manifest_written_on_build():
    m = _model()
    ctx = torch.randint(3, 60, (32,))
    store = LongContextStore(ctx, 8)
    store.build_keys(m.model.retriever, m.model.embed_tokens)
    store.set_manifest(tokenizer="tok-v1", model_version="mmx-v2")
    md = store.metadata
    assert md["index_built"] is True
    assert md["key_encoder_hash"] and len(md["key_encoder_hash"]) == 16
    assert md["block_size"] == 8 and md["n_blocks"] == 4
    assert md["tokenizer"] == "tok-v1" and md["model_version"] == "mmx-v2"


def test_manifest_mismatch_raises():
    m = _model()
    ctx = torch.randint(3, 60, (32,))
    store = LongContextStore(ctx, 8)
    store.build_keys(m.model.retriever, m.model.embed_tokens)
    store.set_manifest(tokenizer="tok-v1")
    with pytest.raises(RuntimeError):
        store.assert_compatible(expected_tokenizer="tok-v2")
    with pytest.raises(RuntimeError):
        store.assert_compatible(expected_block_size=256)
    # 哈希一致则通过
    store.assert_compatible(expected_key_hash=store.metadata["key_encoder_hash"],
                           expected_block_size=8)


def test_key_refresh_rebuilds_index():
    m = _model(key_refresh_interval=1)
    ctx = torch.randint(3, 60, (32,))
    ctrl = AdaptiveController(m)
    ctrl.begin(ctx)
    h0 = ctrl.store.metadata["key_encoder_hash"]
    keys0 = ctrl.store.block_keys.clone()
    # 改 key_proj 后 refresh → 指纹与 keys 变化
    with torch.no_grad():
        m.model.retriever.key_proj.weight.mul_(1.5)
    assert ctrl.maybe_refresh_index(1) is True
    h1 = ctrl.store.metadata["key_encoder_hash"]
    assert h1 != h0
    assert not torch.equal(keys0, ctrl.store.block_keys)
    assert ctrl.stats.get("index_refreshes") == 1
    # interval=0 不 refresh
    cfg = m.config
    cfg.key_refresh_interval = 0
    assert ctrl.maybe_refresh_index(2) is False


def test_query_encoder_path_has_grad_not_only_linear():
    """P2：retrieval CE 梯度进 Prelude 表示 + query_proj（不只线性层）。"""
    torch.manual_seed(71)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", block_size=8, freeze_key=True)
    m = MiniMindXForCausalLM(cfg).train()
    ids = torch.randint(3, 60, (1, 6))
    ctx = torch.randint(3, 60, (32,))
    store = LongContextStore(ctx, 8)
    store.build_keys(m.model.retriever, m.model.embed_tokens)
    q_h = m.model.prelude_hidden(ids, None, no_grad=False)
    qv = m.model.retriever.project_query(q_h)
    scores = m.model.retriever.coarse_scores(qv, store)
    loss = torch.nn.functional.cross_entropy(scores, torch.tensor([1]))
    loss.backward()
    assert m.model.retriever.query_proj.weight.grad is not None
    assert m.model.retriever.key_proj.weight.grad is None
    # Prelude 参数应有梯度（query encoder 路径）
    prelude_grads = [p.grad for p in m.model.prelude.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in prelude_grads),         "Prelude（query 表示编码器）应收到 retrieval CE 梯度"
