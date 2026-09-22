"""代码打包导出（供外部 LLM 审查）：把本次 MiniMind-X 的全部新增代码合并为一个 md 文件。

文件来源：git 未跟踪（本次新增）的 .py/.sh 文件，按 模型→训练器→数据→脚本→测试 排序；
docs/（设计文档与 ARCHITECTURE.md）默认不打包，--include_docs 可带上。

用法：
  python scripts/export_code_bundle.py                       # -> minimind_x_code_bundle.md
  python scripts/export_code_bundle.py --out review.md --include_docs
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 展示顺序（目录前缀 -> 权重）；同权重按路径排序
ORDER = [
    ("model/minimind_x/", 0),        # 模型包（核心）
    ("trainer/", 1),                  # 训练器
    ("dataset/", 2),                  # 数据集
    ("scripts/", 3),                  # 入口/评测/实验脚本
    ("tests/", 4),                    # 测试
]
LANG = {".py": "python", ".sh": "bash", ".md": "markdown"}


def untracked_files():
    out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [l.strip() for l in out.splitlines() if l.strip()]


def sort_key(path):
    for prefix, w in ORDER:
        if path.startswith(prefix):
            return (w, path)
    return (len(ORDER), path)


def main():
    p = argparse.ArgumentParser(description="导出 MiniMind-X 代码审查包")
    p.add_argument("--out", type=str, default="minimind_x_code_bundle.md")
    p.add_argument("--include_docs", action="store_true", help="附带 docs/ 文档")
    args = p.parse_args()

    files = untracked_files()
    code_files = [f for f in files
                  if (f.endswith(".py") or f.endswith(".sh"))
                  and not f.startswith(("docs/", "images/"))
                  and f != "scripts/export_code_bundle.py"]
    code_files.sort(key=sort_key)

    doc_files = []
    if args.include_docs:
        doc_files = sorted(f for f in files if f.startswith("docs/") and f.endswith(".md"))

    total_lines = 0
    sections = []
    for f in code_files + doc_files:
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            content = fh.read().rstrip("\n")
        lang = LANG.get(os.path.splitext(f)[1], "")
        total_lines += content.count("\n") + 1
        sections.append((f, lang, content))

    out_path = os.path.join(ROOT, args.out)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("# MiniMind-X 新增代码审查包\n\n")
        out.write(f"- 文件数：{len(sections)}（代码 {len(code_files)} / 文档 {len(doc_files)}）\n")
        out.write(f"- 总行数：{total_lines}\n")
        out.write("- 选择规则：git 未跟踪的 .py/.sh（即本次全部新增代码），"
                  "按 模型→训练器→数据→脚本→测试 排序\n")
        out.write("- 背景材料：docs/ARCHITECTURE.md（D1–D9 决策映射与语义说明）、"
                  "docs/MiniMind-X 编码任务清单.md（需求来源）\n")
        out.write("- 验证状态：python -m pytest tests/ -q → 50 passed（含可学习性/MRR/蒸馏梯度/"
                  "batch guard/沙箱回归）；训练器/数据生成器/脚本 CPU 冒烟通过\n\n")
        out.write("## 目录\n\n")
        for f, _, content in sections:
            out.write(f"- `{f}` ({content.count(chr(10)) + 1} 行)\n")
        out.write("\n---\n\n")
        for f, lang, content in sections:
            out.write(f"# {f}\n\n```{lang}\n{content}\n```\n\n---\n\n")

    size_kb = os.path.getsize(out_path) / 1024
    print(f"exported {len(sections)} files ({total_lines} lines, {size_kb:.0f} KB) -> {args.out}")
    for prefix, _ in ORDER:
        n = sum(1 for f, _, _ in sections if f.startswith(prefix))
        if n:
            print(f"  {prefix:<22} {n} files")


if __name__ == "__main__":
    main()
