"""LongContextStore（任务清单 T4.1；Doc 1 §6、Doc 2 §5、D9）。

长期记忆库：1M 上下文按 block_size（默认 256）切块存储。
- 索引 key：块内 token embedding 的 mean-pool + 线性投影（廉价，建库即算）
- 块内容 hidden：复用 Prelude(windowed) 分块独立编码（懒编码 + LRU 缓存，
  避免一次性物化全部块 hidden 的显存开销）

建库与编码不回传梯度（D5：检索走非可微路径，监督只作用于 router 的分类损失）。
"""
import torch


class LongContextStore:
    def __init__(self, context_ids, block_size=256):
        """context_ids: [S] 或 [1,S] 的 LongTensor（host 或 device 均可）。"""
        if context_ids.dim() == 2:
            context_ids = context_ids[0]
        self.context_ids = context_ids
        self.block_size = block_size
        n = context_ids.numel()
        self.n_blocks = (n + block_size - 1) // block_size
        pad = self.n_blocks * block_size - n
        if pad:
            context_ids = torch.cat([context_ids, context_ids.new_zeros(pad)])
        self._chunks = context_ids.view(self.n_blocks, block_size)
        self.block_keys = None          # [n_blocks, d]（build_keys 后可用）
        self._hidden_cache = {}         # idx -> [block_size, d]
        self._cache_order = []

    # ---- 建索引 ----
    @torch.no_grad()
    def build_keys(self, retriever, embed_tokens, max_blocks=None, batch_blocks=256):
        """用 retriever.key_proj 对块内 embedding mean-pool 投影得到索引 key。"""
        device = next(retriever.parameters()).device
        n = min(self.n_blocks, max_blocks) if max_blocks else self.n_blocks
        keys = []
        for i in range(0, n, batch_blocks):
            emb = embed_tokens(self._chunks[i:i + batch_blocks].to(device))  # [b, block, d]
            keys.append(retriever.key_proj(emb.mean(dim=1)))
        self.block_keys = torch.cat(keys, dim=0)
        if max_blocks:  # 只索引前 max_blocks 块时同步裁剪 chunk 视图
            self.n_blocks = self.block_keys.shape[0]
        return self.block_keys

    # ---- 懒编码块内容 ----
    @torch.no_grad()
    def get_block_hidden(self, model, block_idx, cache_limit=4096):
        """block_idx: [k] 或 [B, k] 的 LongTensor -> 对应块 hidden（[k, block, d] 或 [B, k, block, d]）。

        model: MiniMindXForCausalLM；用其 Prelude 分块编码并做 LRU 缓存。
        """
        squeeze = block_idx.dim() == 1
        flat_idx = block_idx.reshape(-1)
        missing = sorted({int(i) for i in flat_idx.tolist() if i not in self._hidden_cache})
        if missing:
            newly = model.model.encode_context_blocks(
                self.context_ids, block_idxs=missing)
            for j, i in enumerate(missing):
                self._hidden_cache[i] = newly[j]
                self._cache_order.append(i)
            while len(self._cache_order) > cache_limit:  # LRU 驱逐
                self._hidden_cache.pop(self._cache_order.pop(0), None)
        hidden = torch.stack([self._hidden_cache[int(i)] for i in flat_idx.tolist()])
        if squeeze:
            return hidden
        return hidden.view(*block_idx.shape, self.block_size, -1)
