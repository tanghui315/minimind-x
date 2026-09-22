"""检索链路测试（任务清单 T4.6 验收）：建库、两级检索 recall、gold CE、终局 controller 冒烟。"""
import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.retrieval import LongContextStore


def _tiny_model(**over):
    torch.manual_seed(7)
    kw = dict(hidden_size=32, num_hidden_layers=6, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="minimind_x", block_size=8, topk_coarse=4, topk_fine=2)
    kw.update(over)
    return MiniMindXForCausalLM(MiniMindXConfig(**kw)).eval()


def _orthogonal_retriever(model):
    """把检索投影钉成近似恒等，保证 cosine 可分。"""
    with torch.no_grad():
        d = model.config.hidden_size
        eye = torch.eye(d)
        model.model.retriever.key_proj.weight.copy_(eye)
        model.model.retriever.query_proj.weight.copy_(eye)
    return model


def test_store_build_and_recall():
    model = _orthogonal_retriever(_tiny_model())
    cfg = model.config
    # 构造 8 块上下文；gold 是第 5 块：该块 token 全部用同一独特 id 模式
    gold_block, gold_id = 5, 42
    torch.manual_seed(123)  # 上下文随机块与模型 init 解耦（防 RNG 消耗变化连带 flaky）
    ctx = []
    for b in range(8):
        if b == gold_block:
            ctx += [gold_id] * cfg.block_size
        else:
            ctx += torch.randint(1, 20, (cfg.block_size,)).tolist()
    store = LongContextStore(torch.tensor(ctx), cfg.block_size)
    store.build_keys(model.model.retriever, model.model.embed_tokens)
    assert store.n_blocks == 8 and store.block_keys.shape == (8, cfg.hidden_size)
    # query = gold 块 embedding 的均值方向 → coarse top1 应命中 gold
    q_hidden = model.model.embed_tokens(torch.full((1, 4), gold_id))
    result = model.model.retriever.retrieve(q_hidden, store, model)
    assert result.block_idx.shape == (1, cfg.topk_fine)
    # 正交+embedding 保证的是 coarse 召回：gold 必进候选、且 coarse rank 0
    # （fine 重排用 Prelude 编码 hidden，与 raw-embedding query 不同表示，
    #  不保证 gold 仍占 final top-k —— 旧断言靠 RNG 运气通过，已收紧到 coarse 契约）
    assert gold_block in result.cand_idx[0].tolist(), "粗检候选应含 gold"
    loss = model.model.retriever.retrieval_ce_loss(result.coarse_scores,
                                                   torch.tensor([gold_block]))
    assert torch.isfinite(loss)
    # MRR（外部评审 P2-1）：正交设定下 gold 应排第 1 → MRR==1
    ranking = result.coarse_scores[0].argsort(descending=True)
    rank = int((ranking == gold_block).nonzero().flatten().item())
    assert rank == 0, f"gold 排名 {rank}（应为 0）"
    mrr = 1.0 / (rank + 1)
    assert mrr == 1.0


def test_get_block_hidden_lazy():
    model = _tiny_model()
    cfg = model.config
    ctx = torch.randint(0, 64, (cfg.block_size * 5,))
    store = LongContextStore(ctx, cfg.block_size)
    h = store.get_block_hidden(model, torch.tensor([3, 1]))
    assert h.shape == (2, cfg.block_size, cfg.hidden_size)
    assert 1 in store._hidden_cache and 3 in store._hidden_cache
    h2 = store.get_block_hidden(model, torch.tensor([[3, 1], [0, 0]]))
    assert h2.shape == (2, 2, cfg.block_size, cfg.hidden_size)


def test_controller_end_to_end_smoke():
    """终局编排（T6.2/T6.3）：建库→初检→router 驱动生成，全链路无异常。"""
    from model.minimind_x.controller import AdaptiveController
    model = _tiny_model(max_loop=2)
    ctrl = AdaptiveController(model)
    ids = torch.randint(0, 64, (1, 6))
    ctx = torch.randint(0, 64, (model.config.block_size * 4,))
    out = ctrl.generate_with_context(ids, context_ids=ctx, max_new_tokens=4,
                                     do_sample=False, eos_token_id=None)
    assert out["generated_ids"].shape[1] == 10
    assert "avg_loops" in out["stats"] and out["stats"]["retrievals"] >= 1
    assert out["stats"]["memory_tokens_final"] > 0


def test_controller_training_forward():
    """teacher-forced 训练前向（SFT-C）：oracle 驱动，返回 exit logits 与检索 CE。"""
    from model.minimind_x.controller import AdaptiveController
    model = _orthogonal_retriever(_tiny_model(max_loop=2))
    ctrl = AdaptiveController(model)
    cfg = model.config
    ids = torch.randint(0, 64, (2, 6))
    labels = torch.randint(0, 64, (2, 6))
    gold_block = 2
    ctx = []
    for b in range(4):
        ctx += ([42] if b == gold_block else [5]) * cfg.block_size
    ctx = torch.tensor(ctx)
    out, extras = ctrl.forward_with_oracles(
        ids, None, labels, ctx, gold_block_idx=torch.tensor([gold_block, gold_block]),
        oracle_retrieve=[0, 1], oracle_budget=2)
    assert len(out.exit_logits) == 2
    assert "retrieval_ce" in extras and torch.isfinite(extras["retrieval_ce"])
    assert extras["memory_tokens"] > 0
