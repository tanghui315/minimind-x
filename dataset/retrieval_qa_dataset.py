"""检索式 SFT 数据集（任务清单 T6.5；Doc 4 §8）。

jsonl schema：{"context": str(长上下文), "question": str, "answer": str,
               "gold_block": int(可选，块号), "oracle_r": int(可选), "oracle_retrieve": [0/1](可选)}

主干序列只含 问题+答案（长上下文走 store，D9）；context 分块 tokenize 供建库。
batch 内变长，默认 collate 成 B=1 的 dict（研究级口径；训练器可用任意 B 自行 pad）。
"""
import json
import os

import torch
from torch.utils.data import Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class RetrievalQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_question_len=512, max_answer_len=512,
                 max_context_tokens=None, block_size=256):
        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.max_context_tokens = max_context_tokens
        self.block_size = block_size
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        q_ids = self.tokenizer(s["question"], add_special_tokens=False,
                               max_length=self.max_question_len, truncation=True).input_ids
        a_ids = self.tokenizer(s["answer"], add_special_tokens=False,
                               max_length=self.max_answer_len, truncation=True).input_ids
        ans_prefix = self.tokenizer("答案：", add_special_tokens=False).input_ids
        input_ids = ([self.tokenizer.bos_token_id] + q_ids + ans_prefix +
                     a_ids + [self.tokenizer.eos_token_id])
        # 可见前缀边界（V2 T0.1 因果契约）：初始检索只允许用 [bos]+question，
        # "答案："前缀与 gold answer 对检索 query 不可见
        question_len = 1 + len(q_ids)
        labels = [-100] * question_len + [-100] * len(ans_prefix) + a_ids + [self.tokenizer.eos_token_id]
        ctx_ids = self.tokenizer(s["context"], add_special_tokens=False,
                                 max_length=self.max_context_tokens,
                                 truncation=True).input_ids if s.get("context") else []
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "question_len": torch.tensor(question_len, dtype=torch.long),
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "gold_block": torch.tensor(s.get("gold_block", -1), dtype=torch.long),
            "oracle_r": torch.tensor(s.get("oracle_r", 0), dtype=torch.long),
            "oracle_retrieve": torch.tensor(s.get("oracle_retrieve", []), dtype=torch.long),
            "answer_text": s["answer"],
        }


def collate_retrieval_qa(batch, pad_id=0):
    """变长 pad collate（input/labels pad；context 取 batch 内最长）。"""
    if len(batch) == 1:
        b = batch[0]
        return {**{k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in b.items()}}
    out = {}
    for key in ("input_ids", "labels"):
        seqs = [b[key] for b in batch]
        max_len = max(len(s) for s in seqs)
        pad_value = -100 if key == "labels" else pad_id
        out[key] = torch.stack([torch.cat([s, s.new_full((max_len - len(s),), pad_value)]) for s in seqs])
    out["question_len"] = torch.stack([b["question_len"] for b in batch])
    ctxs = [b["context_ids"] for b in batch]
    max_ctx = max(len(c) for c in ctxs)
    out["context_ids"] = torch.stack([torch.cat([c, c.new_zeros(max_ctx - len(c))]) for c in ctxs])
    for key in ("gold_block", "oracle_r"):
        out[key] = torch.stack([b[key] for b in batch])
    out["oracle_retrieve"] = [b["oracle_retrieve"] for b in batch]
    out["answer_text"] = [b["answer_text"] for b in batch]
    return out
