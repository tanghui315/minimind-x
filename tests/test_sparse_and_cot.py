"""sparse attention（Baseline C）与 CoT 对齐（T2.6）测试。"""
import torch

from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
from model.minimind_x.losses import cot_alignment_loss


def _cfg(**over):
    kw = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=64, flash_attn=False,
              variant="dense", block_size=8, sparse_topk_blocks=2, attn_window=16)
    kw.update(over)
    return MiniMindXConfig(**kw)


def test_sparse_baseline_c_forward():
    """Model C：全 sparse attention 前向 + 生成可用（T4.3/T4.5）。"""
    torch.manual_seed(0)
    cfg = _cfg(prelude_attn="sparse", core_attn="sparse", coda_attn="sparse",
               use_moe=False)
    # dense variant 的 use_loop=False；sparse 三区 + 固定深度即 Model C
    m = MiniMindXForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        out = m(ids, labels=ids.clone())
        assert torch.isfinite(out.loss)
        gen = m.generate(inputs=ids[:, :4], max_new_tokens=3, do_sample=False,
                         eos_token_id=None)
    assert gen.shape == (2, 7)


def test_local_window_attention_forward():
    torch.manual_seed(1)
    cfg = _cfg(prelude_attn="local", coda_attn="local", variant="middle_loop")
    m = MiniMindXForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (2, 20))
    with torch.no_grad():
        out = m(ids, num_loops=2, use_cache=True)
    assert torch.isfinite(out.logits).all()
    # 局部窗口下 KV cache 长度照常增长
    assert out.past_key_values[0][0].shape[1] == 20


def test_cot_alignment_loss():
    """T2.6：loop r 与 step r 对齐为正例 —— 完全对齐时应接近 0。"""
    torch.manual_seed(2)
    steps = [torch.randn(2, 16) for _ in range(3)]
    # exit 与 step 一一相同 → InfoNCE 近似 0、余弦项 0
    loss_same = cot_alignment_loss(list(steps), list(steps))
    assert loss_same.item() < 0.2
    # 打乱配对应显著更大
    exits = [torch.randn(2, 16) for _ in range(3)]
    loss_rand = cot_alignment_loss(exits, list(steps))
    assert loss_rand.item() > loss_same.item()
    # 轮数不匹配时自动截断
    loss_short = cot_alignment_loss(exits[:1], list(steps))
    assert torch.isfinite(loss_short)


def test_encode_cot_steps():
    torch.manual_seed(3)
    cfg = _cfg(variant="minimind_x")
    m = MiniMindXForCausalLM(cfg).eval()
    steps = [torch.randint(0, 64, (6,)), torch.randint(0, 64, (8,))]
    reps = m.model.encode_cot_steps(steps)
    assert len(reps) == 2 and reps[0].shape == (1, cfg.hidden_size)


def test_save_load_pretrained_x():
    import os
    import tempfile
    from trainer.x_utils import save_pretrained_x, load_pretrained_x
    torch.manual_seed(4)
    cfg = _cfg(variant="middle_loop")
    m = MiniMindXForCausalLM(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        out = save_pretrained_x(m, d)
        assert os.path.exists(os.path.join(out, "config.json"))
        m2 = load_pretrained_x(d)
        ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            a = m(ids, num_loops=1).logits
            b = m2(ids, num_loops=1).logits
        assert torch.allclose(a, b, atol=5e-2)  # 权重以 half 落盘（与仓库惯例一致）
