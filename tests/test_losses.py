"""losses / oracle 标签测试（任务清单 T2.2/T2.3；v1.1 修正的 C_compute 期望公式）。"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.minimind_x import (MiniMindXConfig, joint_loss, expected_loops,
                              halt_labels_from_oracle)
from model.minimind_x.losses import (lm_loss_per_exit, halt_bce_loss, consistency_kl_loss,
                                     exit_loss_weights, expected_memory_cost)
from trainer.oracle_labels import (oracle_depth_from_losses, oracle_depth_from_correct,
                                   oracle_budget_from_matrix, oracle_retrieval_label)


def test_expected_loops_hand_example():
    probs = torch.tensor([[0.2], [0.5], [0.9]])  # [R,B]
    e = expected_loops(probs)
    # P(1)=0.2, P(2)=0.4, P(3)=0.36+never-stopped 0.04=0.40（force_stop_last）→ E=2.2
    assert abs(e.item() - 2.2) < 1e-5


def test_expected_loops_certain_stop():
    probs = torch.tensor([[0.0], [1.0], [0.0]])
    assert abs(expected_loops(probs).item() - 2.0) < 1e-6


def test_halt_labels():
    labels, mask = halt_labels_from_oracle(torch.tensor([3, 1, 4]), 4)
    assert labels.tolist() == [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    assert mask.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]]


def test_oracle_depth_from_losses():
    losses = torch.tensor([[3.0, 2.0, 0.7, 0.68], [0.5, 0.49, 0.48, 0.47]])
    r = oracle_depth_from_losses(losses, epsilon=0.05)
    # 样本1：min=0.68，首个 ≤0.73 的是第3轮 → 3；样本2：min=0.47，首个 ≤0.52 的是第1轮 → 1
    assert r.tolist() == [3, 1]


def test_oracle_depth_from_correct():
    corr = torch.tensor([[False, True, True], [False, False, False]])
    assert oracle_depth_from_correct(corr).tolist() == [2, 3]


def test_oracle_budget_and_retrieval():
    m = torch.tensor([[1.0, 0.5, 0.49]])
    assert oracle_budget_from_matrix(m, 0.05).tolist() == [1]  # 2 档即够
    lab = oracle_retrieval_label(torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.98]))
    assert lab.tolist() == [1, 0]


def test_exit_loss_weights():
    assert exit_loss_weights(4, "last").tolist() == [0, 0, 0, 1]
    assert abs(exit_loss_weights(4, "uniform").sum().item() - 1.0) < 1e-6


def test_joint_loss_parts():
    torch.manual_seed(0)
    cfg = MiniMindXConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                          num_key_value_heads=1, vocab_size=64, flash_attn=False,
                          variant="minimind_x", lambda_halt=0.1, lambda_compute=0.01,
                          alpha_consistency=0.1)
    B, S, R, V = 2, 8, cfg.max_loop, cfg.vocab_size
    exit_logits = [torch.randn(B, S, V) for _ in range(R)]
    labels = torch.randint(0, V, (B, S))
    halt_probs = [torch.rand(B) * 0.5 for _ in range(R)]
    parts = joint_loss(exit_logits, labels, halt_probs=halt_probs,
                       oracle_r=torch.tensor([2, 3]), config=cfg)
    for key in ("loss", "lm", "halt", "consistency", "compute"):
        assert key in parts and torch.isfinite(parts[key])
    # compute ∈ [1/R, 1]
    assert 0 < parts["compute"].item() <= 1.0


def test_consistency_kl_zero_when_identical():
    logits = torch.randn(1, 4, 16)
    labels = torch.randint(0, 16, (1, 4))
    assert consistency_kl_loss([logits.clone(), logits.clone()], labels).item() < 1e-6


def test_expected_memory_cost():
    probs = torch.tensor([[0.0, 1.0, 0.0]])
    c = expected_memory_cost(probs, [1024, 2048, 4096])
    assert abs(c.item() - 0.5) < 1e-6  # 2048/4096
