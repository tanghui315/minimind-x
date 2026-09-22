"""SFT 数据配比混合器（任务清单 T7.2；Doc 4 §18 / Doc 5 §12）。

默认配比：General 30% / Coding Agent 25% / Long Context QA 20% / Math 15% / Planning 10%。
输入多个 jsonl 路径 + 比例，按比例有放回采样拼接为单一数据集视图。
"""
import os
import random

import torch
from torch.utils.data import Dataset

DEFAULT_MIX = {
    "general": ("sft_general.jsonl", 0.30),
    "coding": ("sft_coding.jsonl", 0.25),
    "longctx": ("sft_longctx.jsonl", 0.20),
    "math": ("sft_math.jsonl", 0.15),
    "planning": ("sft_planning.jsonl", 0.10),
}


class MixedSFTDataset(Dataset):
    """按比例混合多个 SFT jsonl（均须为 conversations 格式，与 SFTDataset 兼容）。

    用法：
        ds = MixedSFTDataset(tokenizer, "../dataset",
                             [("sft_general.jsonl", 0.3), ("sft_coding.jsonl", 0.7)],
                             total_size=100000)
    """

    def __init__(self, tokenizer, data_dir, path_ratios=None, total_size=100000,
                 max_length=1024, seed=42):
        from dataset.lm_dataset import SFTDataset
        if path_ratios is None:
            path_ratios = [(f, r) for f, r in DEFAULT_MIX.values()]
        norm = sum(r for _, r in path_ratios)
        self.sources = []
        self.weights = []
        for fname, ratio in path_ratios:
            path = fname if os.path.isabs(fname) else os.path.join(data_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"混合数据缺失: {path}")
            self.sources.append(SFTDataset(path, tokenizer, max_length=max_length))
            self.weights.append(ratio / norm)
        self.total_size = total_size
        self.rng = random.Random(seed)
        self.index_plan = self._make_plan()

    def _make_plan(self):
        plan = []
        for src_idx, w in enumerate(self.weights):
            plan += [(src_idx, self.rng.randrange(len(self.sources[src_idx])))
                     for _ in range(int(self.total_size * w))]
        self.rng.shuffle(plan)
        return plan

    def __len__(self):
        return len(self.index_plan)

    def __getitem__(self, index):
        src_idx, sample_idx = self.index_plan[index]
        return self.sources[src_idx][sample_idx]
