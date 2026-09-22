"""实验矩阵 Runner（任务清单 T8.1/T8.5；Doc 6 §6）。

Model A Dense（原 train_pretrain） / B MoE（原 train_pretrain --use_moe）
Model C Sparse（train_pretrain_x --variant dense + sparse attn）
Model D MiniMind-X（train_pretrain_x --variant minimind_x）
× 规模 S/M/L × token 预算。顺序执行（或 --dry_run 打印命令），记录落 experiments/。
"""
import os
import sys
import json
import argparse
import subprocess
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.minimind_x import PRESETS, count_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def build_commands(args):
    cmds = []
    for size in args.sizes.split(","):
        preset = PRESETS[size]
        for model_type in args.models.split(","):
            name = f"{model_type}_{size}"
            if model_type == "A":  # Dense baseline（原模型，不改）
                cmds.append((name, ["python", "trainer/train_pretrain.py",
                                    "--hidden_size", str(preset["hidden_size"]),
                                    "--num_hidden_layers", str(preset["num_hidden_layers"]),
                                    "--epochs", "1"]))
            elif model_type == "B":  # MoE baseline
                cmds.append((name, ["python", "trainer/train_pretrain.py",
                                    "--hidden_size", str(preset["hidden_size"]),
                                    "--num_hidden_layers", str(preset["num_hidden_layers"]),
                                    "--use_moe", "1", "--epochs", "1"]))
            elif model_type == "C":  # Sparse attention baseline（fixed depth）
                cmds.append((name, ["python", "trainer/train_pretrain_x.py",
                                    "--preset", size, "--variant", "dense",
                                    "--loop_mode", "fixed", "--fixed_loops", "1",
                                    "--prelude_attn", "sparse", "--core_attn", "sparse",
                                    "--coda_attn", "sparse"]))
            elif model_type == "D":  # MiniMind-X
                cmds.append((name, ["python", "trainer/train_pretrain_x.py",
                                    "--preset", size, "--variant", "minimind_x",
                                    "--loop_mode", "full_unroll"]))
            else:
                raise ValueError(f"未知 model_type {model_type}（A/B/C/D）")
    return cmds


def main():
    p = argparse.ArgumentParser(description="Scaling 实验矩阵（Doc 6 §6）")
    p.add_argument("--models", type=str, default="D", help="A/B/C/D 组合（逗号分隔）")
    p.add_argument("--sizes", type=str, default="S", help="S/M/L 组合（逗号分隔）")
    p.add_argument("--dry_run", action="store_true", help="只打印命令不执行")
    p.add_argument("--exp_dir", type=str, default="experiments")
    args = p.parse_args()

    cmds = build_commands(args)
    os.makedirs(args.exp_dir, exist_ok=True)
    for name, cmd in cmds:
        print(f"[matrix] {name}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        rec = {"name": name, "cmd": cmd, "returncode": proc.returncode,
               "sec": round(time.time() - t0, 1),
               "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
        with open(os.path.join(args.exp_dir, f"{name}_{int(time.time())}.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        if proc.returncode != 0:
            print(f"[matrix] {name} FAILED，见 {args.exp_dir}")
    print("[matrix] done. records ->", args.exp_dir)


if __name__ == "__main__":
    main()
