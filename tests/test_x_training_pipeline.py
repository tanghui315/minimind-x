"""Exercise the shipped trainers, checkpoint format, evaluators and document CLI on CPU."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from trainer.x_utils import load_x_checkpoint_meta

ROOT = Path(__file__).resolve().parents[1]


def test_x_training_to_inference(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    chat = tmp_path / "chat.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    doc = tmp_path / "context.txt"
    doc.write_text("设备的最新阈值是 37。")
    messages = [{"role": "user", "content": "1+1等于多少？"},
                {"role": "assistant", "content": "2"}]
    corpus.write_text((json.dumps({"text": "机器学习模型在文档中查找证据。"}) + "\n") * 4)
    chat.write_text((json.dumps({"conversations": messages}) + "\n") * 2)
    rl = tmp_path / "rl.jsonl"
    rl.write_text((json.dumps({"conversations": messages, "gt": "2", "context": doc.read_text()}) + "\n") * 2)
    sample = {"question": "阈值是多少？", "answer": "37", "context": doc.read_text(),
              "gold_span": [0, len(doc.read_text())], "split": "train"}
    # A malformed validation row makes accidental training on held-out data observable.
    retrieval.write_text("\n".join(json.dumps(r) for r in [sample, sample, {"split": "val"}]))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    common = ["--device", "cpu", "--tokenizer_path", ROOT / "model"]
    train = [*common, "--save_dir", tmp_path, "--hidden_size", 32, "--num_hidden_layers", 6,
             "--variant", "minimind_x", "--batch_size", 1]
    def run(script, args):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=120)
        assert result.returncode == 0, result.stdout
        return result.stdout

    run("trainer/train_pretrain_x.py", train + [
        "--residual_v2", 1, "--max_loop", 2, "--max_seq_len", 32, "--epochs", 1,
        "--data_path", corpus, "--checkpoint_dir", tmp_path / "resume",
        "--accumulation_steps", 2, "--num_workers", 0])
    parent = "pretrain_x"
    log = run("trainer/train_long_context.py", train + [
        "--from_weight", parent, "--stage_lens", "256", "--steps_per_stage", "2",
        "--data_path", corpus, "--needle_ratio", 0, "--accumulation_steps", 1, "--num_workers", 0])
    assert f"/{parent}_32_x.pth" in log
    parent = "longctx_x_stage0"
    _, meta = load_x_checkpoint_meta(tmp_path / f"{parent}_32_x.pth")
    assert meta["packing"]["valid_tokens"] > 0
    _, pretrain = load_x_checkpoint_meta(tmp_path / "pretrain_x_32_x.pth")
    assert meta["parent_checkpoint"]["id"] == pretrain["checkpoint_id"]
    parent_id = meta["checkpoint_id"]
    for phase in "abc":
        log = run("trainer/train_sft_x.py", train + [
            "--phase", phase, *(["--from_weight", parent] if phase == "a" else []), "--max_steps", 2,
            "--accumulation_steps", 3, "--num_workers", 0, "--data_path", chat,
            "--retrieval_data_path", retrieval, "--oracle_rate", 1, "--train_budget_router", 1])
        assert f"/{parent}_32_x.pth" in log
        parent = f"sft_x_{phase}"
        _, meta = load_x_checkpoint_meta(tmp_path / f"{parent}_32_x.pth")
        assert meta["parent_checkpoint"]["id"] == parent_id
        parent_id = meta["checkpoint_id"]
        assert meta["optimizer"]["state"]
        assert meta["config"]["residual_v2"] and meta["config"]["max_loop"] == 2
    for phase in (1, 2, 3):
        name = f"grpo_x_phase{phase}"
        log = run("trainer/train_grpo_x.py", train + [
            "--rl_phase", phase, "--data_path", rl, "--num_generations", 8, "--max_gen_len", 16, "--learning_rate", 0.0005])
        assert f"/{parent}_32_x.pth" in log
        previous, _ = load_x_checkpoint_meta(tmp_path / f"{parent}_32_x.pth")
        updated, meta = load_x_checkpoint_meta(tmp_path / f"{name}_32_x.pth")
        assert meta["parent_checkpoint"]["id"] == parent_id
        parent_id = meta["checkpoint_id"]
        for key in ("model.retrieve_router.action_head.weight", "model.budget_router.proj.weight"):
            assert not torch.equal(previous[key], updated[key])
        assert torch.isfinite(updated["lm_head.weight"]).all()
        assert not torch.equal(previous["lm_head.weight"], updated["lm_head.weight"])
        parent = name
    evaluate = common + ["--weight", parent, "--weight_dir", tmp_path]
    report = tmp_path / "report.json"
    run("scripts/eval_suite.py", evaluate + ["--ppl_data", corpus, "--out", report, "--skip_needle"])
    data = json.loads(report.read_text())
    assert data["config"]["architecture_version"] == "minimind_x_residual_v2"
    assert data["config"]["max_loop"] == 2 and 1 <= data["ppl"] < math.inf
    assert data["efficiency"]["gen_tokens"] > 0 and data["efficiency"]["ttft_ms"] > 0
    run("scripts/eval_depth_dist.py", evaluate + [
        "--difficulties", 1, "--n_per_difficulty", 1, "--max_new_tokens", 2,
        "--out", tmp_path / "depth.json"])
    depth = json.loads((tmp_path / "depth.json").read_text())["per_difficulty"][0]
    assert len(depth["acc_fixed_by_R"]) == 2 and depth["n"] == 1
    assert depth["token_decisions"]["tokens"] >= 1
    run("scripts/eval_needle.py", evaluate + [
        "--ctx_tokens", 128, "--depths", 0.5, "--n_per_cell", 1, "--max_new_tokens", 2,
        "--out", tmp_path / "needle.json"])
    val = tmp_path / "validation.jsonl"
    val.write_text("\n".join(json.dumps(dict(sample, split="val", family=family)) for family in ["A","A","B"]))
    run("scripts/eval_needle.py", evaluate + ["--data", val, "--n_per_cell", 1, "--max_new_tokens", 1,
                                             "--out", tmp_path / "families.json"])
    cells = json.loads((tmp_path / "families.json").read_text())["cells"]
    assert {c["family"]: c["n"] for c in cells} == {"A": 2, "B": 1}
    log = run("scripts/eval_x.py", evaluate + ["--context_file", doc, "--max_new_tokens", 2])
    assert "[Speed]" in log
