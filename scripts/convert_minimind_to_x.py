"""原 MiniMind 权重 -> MiniMind-X 布局转换（任务清单 T0.3）。

用法：
  python scripts/convert_minimind_to_x.py --in ../out/pretrain_768.pth \
      --out ../out/pretrain_x_768_x.pth --variant middle_loop
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.minimind_x import MiniMindXConfig, remap_minimind_state_dict


def main():
    p = argparse.ArgumentParser(description="MiniMind -> MiniMind-X 权重转换")
    p.add_argument("--in", dest="src", type=str, required=True, help="原 MiniMind .pth")
    p.add_argument("--out", dest="dst", type=str, required=True, help="输出 X .pth")
    p.add_argument("--variant", type=str, default="minimind_x",
                   choices=["dense", "full_loop", "middle_loop", "late_middle", "minimind_x"])
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--preset", type=str, default=None, choices=["S", "M", "L"])
    p.add_argument("--prelude_layers", type=int, default=None)
    p.add_argument("--thinking_layers", type=int, default=None)
    p.add_argument("--coda_layers", type=int, default=None)
    args = p.parse_args()

    cfg = MiniMindXConfig(preset=args.preset, variant=args.variant,
                          **{k: v for k, v in dict(
                              hidden_size=args.hidden_size,
                              num_hidden_layers=args.num_hidden_layers,
                              prelude_layers=args.prelude_layers,
                              thinking_layers=args.thinking_layers,
                              coda_layers=args.coda_layers).items() if v is not None})
    src = torch.load(args.src, map_location="cpu")
    if "model" not in src and not any(k.startswith("model.") for k in src):
        raise ValueError(f"{args.src} 不像 MiniMindForCausalLM 的 state_dict")
    mapped = remap_minimind_state_dict(src, cfg)
    n_mapped = len([k for k in mapped if k.startswith("model.")])
    total_backbone = cfg.num_hidden_layers * (4 + 1) + 1  # 粗略：层数×(attn4+mlp1)+norm
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(mapped, args.dst)
    print(f"converted {len(mapped)} tensors -> {args.dst}")
    print(f"zone layout: prelude={cfg.prelude_layers} core={cfg.thinking_layers} "
          f"coda={cfg.coda_layers} (variant={cfg.variant})")
    print("note: routers/retriever 新模块未初始化，加载时用 strict=False（训练中学习）")


if __name__ == "__main__":
    main()
