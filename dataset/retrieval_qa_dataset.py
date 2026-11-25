"""检索式 SFT 数据集（任务清单 T6.5；Doc 4 §8）。

jsonl schema：{"context": str, "question": str, "answer": str,
               "gold_block": int, "gold_span": [start,end) 字符偏移,
               "split": "train"|"val", "oracle_r": int, "oracle_retrieve": [...],
               "action_schema_version": 1|2}

主干序列只含 问题+答案（长上下文走 store，D9）；context 分块 tokenize 供建库。
batch 内变长，默认 collate 成 B=1 的 dict（研究级口径；训练器可用任意 B 自行 pad）。
"""
from bisect import bisect_left, bisect_right
import json
import os

import torch
from torch.utils.data import Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def token_span(offsets, span):
    """Map a complete character span to a half-open range of tokenizer positions."""
    if not span or not offsets:
        return None
    start, end = span
    if start < 0 or end <= start or end > offsets[-1][1]:
        return None
    first = bisect_right(offsets, start, key=lambda pair: pair[1])
    last = bisect_left(offsets, end, key=lambda pair: pair[0])
    return (first, last) if first < last else None


def encode_evidence(record, tokenizer, block_size, max_tokens=None):
    context = record.get("context", "")
    span = record.get("gold_span")
    # Earlier Needle files retained the literal needles and one numerical answer.
    if span and span[0] < 0:
        span = None
    if span is None and record.get("needles"):
        matches = [n for n in record["needles"] if n.endswith(f' {record["answer"]}。') and n in context]
        if len(matches) == 1:
            start = context.index(matches[0])
            span = (start, start + len(matches[0]))
    encoded = tokenizer(context, add_special_tokens=False, truncation=max_tokens is not None,
                        max_length=max_tokens, return_offsets_mapping=bool(span))
    ids = torch.tensor(encoded.input_ids, dtype=torch.long)
    n_blocks = (len(ids) + block_size - 1) // block_size
    blocks = torch.zeros(n_blocks, dtype=torch.long)
    gold = record.get("gold_block", -1)
    visible_from = record.get("evidence_visible_from_round", 0)
    if span:
        offsets = encoded.offset_mapping
        end = min(span[1], offsets[-1][1]) if offsets else 0
        positions = token_span(offsets, (span[0], end))
        gold = -1
        if positions:
            first, last = positions
            # A truncated fragment is still hidden until its release round.
            blocks[first // block_size:(last - 1) // block_size + 1] = visible_from
            if end == span[1]:
                gold = (first + last - 1) // 2 // block_size
    elif 0 <= gold < n_blocks:
        blocks[gold] = visible_from
    return ids, gold if 0 <= gold < n_blocks else -1, blocks


class RetrievalQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_question_len=512, max_answer_len=512,
                 max_context_tokens=None, block_size=256, split=None):
        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.max_context_tokens = max_context_tokens
        self.block_size = block_size
        self.split = split
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # 评审 7.2：按 split 过滤，val 不得进训练
                if split is not None and rec.get("split", "train") != split:
                    continue
                self.samples.append(rec)

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
        ctx_ids, gold, rounds = encode_evidence(s, self.tokenizer, self.block_size, self.max_context_tokens)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "question_len": torch.tensor(question_len, dtype=torch.long),
            "context_ids": ctx_ids,
            "gold_truncated": torch.tensor(self.max_context_tokens is not None and gold < 0 and
                                           (s.get("gold_block", -1) >= 0 or bool(s.get("gold_span"))) and
                                           encode_evidence(s, self.tokenizer, self.block_size)[1] >= 0),
            "gold_block": torch.tensor(gold, dtype=torch.long),
            "block_visible_from": rounds,
            "oracle_r": torch.tensor(s.get("oracle_r", 0), dtype=torch.long),
            "oracle_retrieve": torch.tensor(s.get("oracle_retrieve", []), dtype=torch.long),
            "action_schema_version": s.get("action_schema_version", 1),
            "answer_text": s["answer"],
        }


def collate_retrieval_qa(batch, pad_id=0):
    """变长 pad collate（input/labels pad；context 取 batch 内最长）。"""
    if len(batch) == 1:
        b = batch[0]
        out = {**{k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in b.items()}}
        # sft_c_step 用 batch["action_schema_version"] 取标量/列表
        if isinstance(out.get("action_schema_version"), list):
            out["action_schema_version"] = out["action_schema_version"][0]
        return out
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
    out["action_schema_version"] = [b.get("action_schema_version", 1) for b in batch]
    out["block_visible_from"] = [b["block_visible_from"] for b in batch]
    out["answer_text"] = [b["answer_text"] for b in batch]
    return out
