"""长上下文数据集（任务清单 T3.2；Doc 4 §7、Doc 5 §8）。

- LongContextPretrainDataset：document packing（多文档拼接至 max_length，无 padding 浪费），
  支持 needle 混合（合成长依赖样本 + gold 位置加权 → L_retrieval-aware）。
- 返回 (input_ids, labels, loss_weight)：loss_weight 供 retrieval-aware 辅助损失加权。

与 scripts/data_gen/* 配合：生成器产出更复杂的 jsonl（needle/multihop/repo），
本类负责 token 化、打包与 gold 权重标注。
"""
import json
import os
import random

import torch
from torch.utils.data import Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FILLER_WORDS = ("系统 数据 模型 训练 语言 上下文 检索 记忆 注意 表示 计算 结构 过程 "
                "信息 网络 参数 优化 目标 函数 结果 分析 方法 实验 性能 任务 序列 知识").split()

NEEDLE_TEMPLATES = [
    ("重要提示：本段的魔法数字是 {num}。", "本段的魔法数字是多少？", "{num}"),
    ("关键记录：会议编号为 {num}。", "会议编号是多少？", "{num}"),
    ("核心配置：缓存大小设置为 {num}。", "缓存大小是多少？", "{num}"),
]


def synth_needle_text(target_tokens, depth_ratio=None, rng=None):
    """构造一条 needle 样本文本：填充文本 + 深处插入 needle + 末尾提问。

    返回 (text, needle_sentence)。depth_ratio: 插入深度比例（0~1，None 随机）。
    """
    rng = rng or random
    num = rng.randint(1000, 9999)
    tpl_idx = rng.randrange(len(NEEDLE_TEMPLATES))
    template, question, _ = NEEDLE_TEMPLATES[tpl_idx]
    needle = template.format(num=num)
    depth = depth_ratio if depth_ratio is not None else rng.random()
    # 用填充句堆到目标长度（按字符近似，1 token ≈ 1.5 中文字符）
    target_chars = int(target_tokens * 1.5)
    parts, total = [], 0
    while total < target_chars:
        sent = "".join(rng.choice(FILLER_WORDS) for _ in range(rng.randint(8, 20))) + "。"
        parts.append(sent)
        total += len(sent)
    pos = max(1, int(len(parts) * depth))
    parts.insert(pos, needle)
    text = "".join(parts) + question
    return text, needle


class LongContextPretrainDataset(Dataset):
    """打包式长上下文预训练数据（stage A–C：8K→128K 全量 attention 主干）。"""

    def __init__(self, data_path, tokenizer, max_length=8192, needle_ratio=0.1,
                 needle_weight=5.0, seed=42, min_chunks=1):
        from array import array
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.needle_ratio = needle_ratio
        self.needle_weight = needle_weight
        self.seed = seed
        self.epoch = 0
        # A compact contiguous stream preserves long document tails and document boundaries.
        self.tokens = array("I")
        with open(data_path, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    ids = tokenizer(json.loads(line)["text"], add_special_tokens=False).input_ids
                    self.tokens.extend([tokenizer.bos_token_id, *ids, tokenizer.eos_token_id])
        self.chunk_size = min(max_length, len(self.tokens) // min_chunks)
        if self.chunk_size < 2:
            raise ValueError("Too few corpus tokens for one training example per rank")

    def __len__(self):
        return (len(self.tokens) + self.chunk_size - 1) // self.chunk_size

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = random.Random(self.seed + index + self.epoch * len(self))
        start = index * self.chunk_size
        ids = list(self.tokens[start:start + self.chunk_size])
        weight = torch.ones(self.max_length)
        if rng.random() < self.needle_ratio:
            from scripts.data_gen.gen_needle import gen_sample
            from dataset.retrieval_qa_dataset import token_span
            # Leave room for the question and supervised answer at the end.
            sample = gen_sample(rng, self.max_length - 32, [rng.random()], 1, 256, self.tokenizer)
            enc = self.tokenizer(sample["context"], add_special_tokens=False, return_offsets_mapping=True)
            first, last = token_span(enc.offset_mapping, sample["gold_span"])
            suffix = self.tokenizer(sample["question"] + sample["answer"], add_special_tokens=False).input_ids
            ids = [self.tokenizer.bos_token_id, *enc.input_ids, *suffix, self.tokenizer.eos_token_id]
            if len(ids) > self.max_length:
                raise ValueError("Needle question exceeds the reserved space")
            weight[first + 1:last + 1] = self.needle_weight
            weight[len(enc.input_ids) + 1:len(ids)] = self.needle_weight
        x = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        x[:len(ids)] = torch.tensor(ids)
        labels = x.clone()
        labels[len(ids):] = -100
        return x, labels, weight


def collate_long_context(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    ws = torch.stack([b[2] for b in batch])
    return xs, ys, ws
