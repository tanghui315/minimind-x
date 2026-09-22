"""Scaling Law 拟合（任务清单 T8.3；Doc 6 §14）。

拟合 L = A · N^−α · D^−β · R^−γ · M^−δ（log 线性最小二乘 + 标准误与 R²）。
输入 experiments/ 下的记录 jsonl：每行 {"N":, "D":, "R":, "M":, "loss":}
（R/M 可选；缺失的维度自动跳过 —— 检验 γ,δ>0 是否成立是核心研究问题）。
"""
import os
import sys
import json
import argparse
import glob

import numpy as np


def load_records(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r, dict) and "loss" in r and "N" in r and "D" in r:
                rows.append(r)
    return rows


def fit(rows, use=("N", "D", "R", "M")):
    vars_ = [v for v in use if all(v in r and r[v] for r in rows)]
    X = np.stack([np.log(np.array([float(r[v]) for r in rows])) for v in vars_], axis=1)
    y = np.log(np.array([float(r["loss"]) for r in rows]))
    X1 = np.concatenate([np.ones((len(rows), 1)), X], axis=1)
    coef, residuals, rank, _ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ coef
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    r2 = 1 - ss_res / ss_tot
    dof = max(len(rows) - X1.shape[1], 1)
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    names = ["logA"] + [f"-exp({v})" for v in vars_]
    return dict(zip(names, zip(coef.tolist(), se.tolist()))), r2, vars_


def main():
    p = argparse.ArgumentParser(description="Scaling Law 拟合")
    p.add_argument("--records", type=str, default="experiments/scaling_records.jsonl")
    p.add_argument("--use", type=str, default="N,D,R,M", help="参与拟合的变量")
    args = p.parse_args()
    paths = sorted(glob.glob(args.records)) or (
        [args.records] if os.path.exists(args.records) else [])
    rows = load_records(paths)
    if len(rows) < len(args.use.split(",")) + 2:
        print(f"记录不足（{len(rows)} 条），请先通过实验矩阵累积记录。格式：")
        print('  {"N": 参数量, "D": tokens, "R": 平均深度, "M": memory tokens, "loss": val loss}')
        return
    coefs, r2, used = fit(rows, tuple(args.use.split(",")))
    print(f"拟合变量: {used}   R² = {r2:.4f}")
    for name, (c, s) in coefs.items():
        print(f"  {name:>8}: {c:+.4f} ± {s:.4f}")
    print("\n研究问题（Doc 6 §14）：γ(R) 与 δ(M) 的指数是否显著 > 0（即 ±1.96·SE 下界 > 0）")


if __name__ == "__main__":
    main()
