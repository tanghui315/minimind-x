"""MiniMindXConfig 测试（任务清单 T0.5）：preset、variant 解析、切分校验、预算换算。"""
import pytest
import torch

from model.minimind_x import MiniMindXConfig, PRESETS, resolve_variant


def test_default_split_is_3_3_2():
    cfg = MiniMindXConfig(hidden_size=64, num_hidden_layers=8, variant="minimind_x",
                          num_attention_heads=4, num_key_value_heads=2)
    assert (cfg.prelude_layers, cfg.thinking_layers, cfg.coda_layers) == (3, 3, 2)
    assert cfg.use_loop and cfg.use_memory and cfg.use_retrieval


def test_variant_matrix():
    assert resolve_variant("full_loop", 8)["thinking_layers"] == 8
    assert resolve_variant("full_loop", 8)["prelude_layers"] == 0
    # late_middle: L1-L4 + (L5-L6)xR + L7-L8
    r = resolve_variant("late_middle", 8)
    assert (r["prelude_layers"], r["thinking_layers"], r["coda_layers"]) == (4, 2, 2)
    assert resolve_variant("dense", 8)["use_loop"] is False


def test_split_validation():
    with pytest.raises(ValueError):
        MiniMindXConfig(hidden_size=64, num_hidden_layers=8, variant="minimind_x",
                        prelude_layers=5, thinking_layers=5, coda_layers=5)


def test_presets():
    for name, p in PRESETS.items():
        assert p["prelude_layers"] + p["thinking_layers"] + p["coda_layers"] == p["num_hidden_layers"]
    cfg = MiniMindXConfig(preset="S", variant="minimind_x")
    assert cfg.hidden_size == 512 and cfg.num_hidden_layers == 12
    assert (cfg.thinking_layers, cfg.prelude_layers, cfg.coda_layers) == (4, 4, 4)


def test_budget_blocks():
    cfg = MiniMindXConfig(hidden_size=64, num_hidden_layers=8, block_size=256)
    assert cfg.budget_blocks(0) == 4    # 1024/256
    assert cfg.budget_blocks(4) == 64   # 16384/256


def test_mode_caps():
    cfg = MiniMindXConfig(hidden_size=64, num_hidden_layers=8)
    assert cfg.mode_caps_for("fast")["max_loop"] == 1
    assert cfg.mode_caps_for("normal")["max_budget"] == 8192
    assert cfg.mode_caps_for(None)["max_loop"] == cfg.max_loop
    with pytest.raises(ValueError):
        cfg.mode_caps_for("turbo")
