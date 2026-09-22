"""Working Memory（任务清单 T5.1；Doc 1 §9、Doc 2 §8、§15）。

Block-slot 容器：每个 slot 是一个块 ——
  hidden: [B, K_blocks, block_size, d]（块内容，Prelude 编码）
  keys:   [B, K_blocks, d]（块键，检索/打分用）
  usage / persistence: [B, K_blocks] 统计量
容量 B ≤ B_max 以 block 数计。纯张量容器、无可学习参数（可学习部分在 MemoryManager）。

跨 loop K/V 缓存（Doc 2 §15，即 Doc 2 §23 的 memory_cache.py）：
块 hidden 展平为 token 序列后经 ThinkingCore.project_memory 投影为各层 K/V 并缓存，
memory 内容不变时 loop 间直接复用，内容变更（新对象）后自动失效重投影。
"""
import torch


class WorkingMemory:
    def __init__(self, hidden, keys, usage=None, persistence=None, block_size=256,
                 source_ids=None, memory_version=0, valid=None, metadata=None):
        assert hidden.dim() == 4 and keys.dim() == 3
        assert hidden.shape[:2] == keys.shape[:2]
        self.hidden = hidden
        self.keys = keys
        self.block_size = hidden.shape[2] if hidden.shape[2] else block_size
        bsz, k = hidden.shape[0], hidden.shape[1]
        self.usage = usage if usage is not None else hidden.new_zeros(bsz, k)
        self.persistence = persistence if persistence is not None else hidden.new_zeros(bsz, k)
        # T3.3（V2 §9.3）：去重身份与版本——source_ids 标记块来源；每次内容变更
        # 产生新对象 + 递增版本（get_kv 的投影缓存随之失效）
        self.source_ids = source_ids if source_ids is not None else torch.arange(
            k, device=hidden.device).unsqueeze(0).expand(bsz, -1).contiguous()
        self.memory_version = memory_version
        # T3.3 WorkingMemoryV2：valid mask + metadata
        self.valid = valid if valid is not None else (
            torch.ones(bsz, k, dtype=torch.bool, device=hidden.device) if k
            else torch.zeros(bsz, 0, dtype=torch.bool, device=hidden.device))
        self.metadata = dict(metadata or {})
        self._kv_cache = None  # [(k,v), ...] 与 core 层对齐；内容变更即失效

    # ---- 基本属性 ----
    @property
    def size(self):  # block 数（标量；batch 内统一容量）
        return self.hidden.shape[1]

    @property
    def token_size(self):  # 以 token 计的记忆量（供 C_m 统计）
        return self.hidden.shape[1] * self.block_size

    def is_empty(self):
        return self.hidden.shape[1] == 0

    def token_hidden(self):
        """[B, K, block, d] -> [B, K*block, d]（供 project_memory）。"""
        b, k, bs, d = self.hidden.shape
        return self.hidden.reshape(b, k * bs, d)

    def token_valid(self):
        """[B, K] valid -> [B, K*block] token 级 valid（供 GatedMemoryReader）。"""
        b, k = self.valid.shape
        bs = self.block_size if k else 0
        return self.valid.unsqueeze(-1).expand(b, k, bs).reshape(b, k * bs)

    # ---- 修改操作（返回新对象，函数式；Doc 1 §10 ADD/KEEP/EVICT/REPLACE 由 manager 组合）----
    def with_added(self, hidden, keys, source_ids=None):
        """并入新块（版本递增；source_ids 去重——同 source 只更新 usage，不复制 slot）。"""
        if self.is_empty():
            return WorkingMemory(hidden, keys, block_size=self.block_size,
                                 source_ids=source_ids, memory_version=self.memory_version + 1,
                                 valid=torch.ones(hidden.shape[0], hidden.shape[1],
                                                  dtype=torch.bool, device=hidden.device),
                                 metadata=self.metadata)
        new_ids = (source_ids if source_ids is not None else
                   torch.arange(self.size, self.size + hidden.shape[1],
                                device=self.hidden.device).unsqueeze(0).expand(
                       hidden.shape[0], -1).contiguous())
        # T3.3/V2 §9.3：按 source_id 去重——重复检索同一块不复制 slot
        keep_rows = []
        bsz, k_new = hidden.shape[0], hidden.shape[1]
        # 研究级 B=1/共享 source 布局；逐样本 mask 新块中已存在的 source
        dup = torch.zeros(bsz, k_new, dtype=torch.bool, device=hidden.device)
        for b in range(bsz):
            existing = set(int(x) for x in self.source_ids[b].tolist())
            for j in range(k_new):
                sid = int(new_ids[b, j].item())
                if sid in existing:
                    dup[b, j] = True
        if bool(dup.any()):
            keep = ~dup.all(dim=0)  # 任一样本非重复则保留该槽（batch 统一容量，研究级）
            hidden = hidden[:, keep]
            keys = keys[:, keep]
            new_ids = new_ids[:, keep]
            if hidden.shape[1] == 0:
                return WorkingMemory(self.hidden, self.keys, self.usage, self.persistence,
                                     block_size=self.block_size, source_ids=self.source_ids,
                                     memory_version=self.memory_version,  # 无新内容
                                     valid=self.valid, metadata=self.metadata)
        new_valid = torch.ones(hidden.shape[0], hidden.shape[1],
                               dtype=torch.bool, device=hidden.device)
        return WorkingMemory(
            torch.cat([self.hidden, hidden], dim=1),
            torch.cat([self.keys, keys], dim=1),
            torch.cat([self.usage, self.usage.new_zeros(hidden.shape[0], hidden.shape[1])], dim=1),
            torch.cat([self.persistence, self.persistence.new_zeros(hidden.shape[0], hidden.shape[1])], dim=1),
            block_size=self.block_size,
            source_ids=torch.cat([self.source_ids, new_ids], dim=1),
            memory_version=self.memory_version + 1,
            valid=torch.cat([self.valid, new_valid], dim=1),
            metadata=self.metadata)

    def select(self, keep_idx):
        """keep_idx: [B, m'] —— 按样本保留 slot（驱逐 argmin(U) 之后的结果）。"""
        bsz = self.hidden.shape[0]
        arange = torch.arange(bsz, device=self.hidden.device).unsqueeze(1)
        return WorkingMemory(
            self.hidden[arange, keep_idx], self.keys[arange, keep_idx],
            self.usage[arange, keep_idx], self.persistence[arange, keep_idx],
            block_size=self.block_size, source_ids=self.source_ids[arange, keep_idx],
            memory_version=self.memory_version + 1,
            valid=self.valid[arange, keep_idx], metadata=self.metadata)

    def with_usage_bumped(self, hit_mask, decay=0.9):
        """attention/retrieval 使用统计的 EMA 更新：被再次选中的块 usage 提升。"""
        usage = decay * self.usage + (1 - decay) * hit_mask.float()
        return WorkingMemory(self.hidden, self.keys, usage, self.persistence,
                             block_size=self.block_size, source_ids=self.source_ids,
                             memory_version=self.memory_version,
                             valid=self.valid, metadata=self.metadata)

    def with_persistence_ema(self, scores, decay=0.9):
        """persistence = 历史 U 的 EMA（Doc 1 §11 Persistence 因子）。"""
        persistence = decay * self.persistence + (1 - decay) * scores.detach()
        return WorkingMemory(self.hidden, self.keys, self.usage, persistence,
                             block_size=self.block_size, source_ids=self.source_ids,
                             memory_version=self.memory_version,
                             valid=self.valid, metadata=self.metadata)

    # ---- 跨 loop K/V 缓存 ----
    def get_kv(self, thinking_core):
        """返回与 thinking_core.layers 对齐的 [(k, v), ...]；缓存失效自动重投影。"""
        if self._kv_cache is None:
            self._kv_cache = thinking_core.project_memory(self.token_hidden())
        return self._kv_cache

    def invalidate_kv(self):
        self._kv_cache = None
