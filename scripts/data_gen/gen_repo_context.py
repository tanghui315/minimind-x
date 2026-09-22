"""代码仓库上下文数据（任务清单 T3.1；Doc 4 §7.4、Doc 5 §8）。

合成迷你仓库（目录树 + 多文件函数），生成 Issue→Patch 任务。
产出检索式 jsonl：{context(仓库全文), question(Issue), answer(Patch), gold_block}。
"""
import os
import json
import random
import argparse

FUNC_VERBS = ["计算", "校验", "解析", "转换", "缓存", "汇总", "排序", "过滤"]
TYPES = ["int", "float", "str", "list", "dict", "bool"]


def gen_repo(rng, n_files, funcs_per_file):
    files = {}
    for fi in range(n_files):
        name = f"module_{fi}.py"
        lines = [f"# {name}: auto-generated module", "from typing import Any", ""]
        for gi in range(funcs_per_file):
            fn = f"{rng.choice(FUNC_VERBS)}_{rng.randint(10, 99)}"
            args = ", ".join(f"a{i}: {rng.choice(TYPES)}" for i in range(rng.randint(1, 3)))
            body = [f"def {fn}({args}) -> Any:"]
            indent = "    "
            if rng.random() < 0.5:
                body.append(f"{indent}return sum(a for a in [{args.split(':')[0]}] if isinstance(a, (int, float)))")
            else:
                body.append(f"{indent}return str({args.split(':')[0]})")
            lines.extend(body + [""])
        files[name] = "\n".join(lines)
    tree = "\n".join(f"- {n} ({len(c)} bytes)" for n, c in files.items())
    return f"src/\n{tree}\n\n" + "\n\n".join(f"# file: {n}\n{c}" for n, c in files.items()), files


def gen_sample(rng, n_files, funcs_per_file, ctx_tokens, block_size):
    ctx, files = gen_repo(rng, n_files, funcs_per_file)
    target_file = rng.choice(list(files))
    # 截断/填充到目标长度
    if len(ctx) > ctx_tokens * 4:
        ctx = ctx[:ctx_tokens * 4]
    bug_fn = [l.split("(")[0][4:] for l in files[target_file].splitlines()
              if l.startswith("def ")]
    fn = rng.choice(bug_fn) if bug_fn else "main"
    issue = (f"Issue: {target_file} 中的 {fn}() 在空输入时抛出 TypeError。\n"
             f"请给出最小修复 patch。")
    answer = (f"Patch ({target_file}):\n"
              f"@@ def {fn} @@\n"
              f"-    return sum(a for a in [a0] ...)\n"
              f"+    if not args: return 0\n"
              f"+    return sum(a for a in args if isinstance(a, (int, float)))")
    file_pos = ctx.find(f"# file: {target_file}")
    gold_ratio = max(file_pos, 0) / max(len(ctx), 1)
    n_blocks = max(1, round(ctx_tokens / block_size))
    return {"context": ctx, "question": issue, "answer": answer,
            "gold_block": min(int(gold_ratio * n_blocks), n_blocks - 1)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="生成 repo-context 检索式数据")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n_files", type=int, default=12)
    p.add_argument("--funcs_per_file", type=int, default=6)
    p.add_argument("--ctx_tokens", type=int, default=8192)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(json.dumps(gen_sample(rng, args.n_files, args.funcs_per_file,
                                          args.ctx_tokens, args.block_size), ensure_ascii=False) + "\n")
    print(f"written {args.n} samples -> {args.out}")
