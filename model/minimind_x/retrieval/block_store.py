"""LongContextStore（任务清单 T4.1；Doc 1 §6、Doc 2 §5、D9）。

长期记忆库：1M 上下文按 block_size（默认 256）切块存储。
- 索引 key：块内 token embedding 的 mean-pool + 线性投影（廉价，建库即算）
- 块内容 hidden：复用 Prelude(windowed) 分块独立编码（懒编码 + LRU 缓存）

Frozen Index：建库在 no_grad 下用 key_proj；训练只优化 query 路径。
Index Manifest（P0）：metadata 记录 key_encoder 哈希/tokenizer/block_size，
避免「模型升级 + 旧索引」静默错配。
"""
import hashlib
import torch


def encoder_hash(retriever, embeddings):
    weights = (retriever.key_proj.weight, embeddings.weight)
    versions = tuple((id(w), w._version) for w in weights)
    cached = getattr(retriever, "_index_fingerprint", None)
    if cached is None or cached[0] != versions:
        digest = hashlib.sha256()
        for weight in weights:
            digest.update(str((weight.shape, weight.dtype)).encode())
            digest.update(weight.detach().contiguous().view(torch.uint8).cpu().numpy())
        cached = retriever._index_fingerprint = versions, digest.hexdigest()[:16]
    return cached[1]


class LongContextStore:
    def __init__(self, context_ids, block_size=256, block_visible_from=None):
        """context_ids: [S] 或 [1,S] 的 LongTensor（host 或 device 均可）。"""
        if context_ids.dim() == 2:
            context_ids = context_ids[0]
        self.context_ids = context_ids
        self.block_size = block_size
        n = context_ids.numel()
        self.n_blocks = (n + block_size - 1) // block_size
        self.block_visible_from = None
        if block_visible_from is not None:
            self.block_visible_from = torch.as_tensor(block_visible_from, device=context_ids.device).flatten()
            if self.block_visible_from.numel() != self.n_blocks or (self.block_visible_from < 0).any():
                raise ValueError("block_visible_from requires one nonnegative round per context block")
        self.round = 0
        pad = self.n_blocks * block_size - n
        if pad:
            context_ids = torch.cat([context_ids, context_ids.new_zeros(pad)])
        self._chunks = context_ids.view(self.n_blocks, block_size)
        self.block_keys = None          # [n_blocks, d]（build_keys 后可用）
        self.encoded_blocks = 0
        self._hidden_cache = {}         # idx -> [block_size, d]
        self._cache_order = []
        # P0 Index Manifest：加载侧校验，防模型/索引版本错配
        self.metadata = {
            "key_encoder_hash": None,
            "block_size": int(block_size),
            "n_blocks": int(self.n_blocks),
            "tokenizer": None,
            "model_version": None,
            "index_built": False,
        }

    def token_mask(self, indices):
        offsets = torch.arange(self.block_size, device=indices.device)
        return indices[..., None] * self.block_size + offsets < self.context_ids.numel()

    def advance_round(self, round_index):
        self.round = max(self.round, round_index)

    @property
    def visible_count(self):
        if self.block_visible_from is None:
            return self.n_blocks
        return int((self.block_visible_from[:self.n_blocks] <= self.round).sum().item())

    # ---- 建索引 ----
    @torch.no_grad()
    def build_keys(self, retriever, embed_tokens, max_blocks=None, batch_blocks=256):
        """用 retriever.key_proj 对块内 embedding mean-pool 投影得到索引 key。"""
        device = next(retriever.parameters()).device
        n = min(len(self._chunks), max_blocks) if max_blocks is not None else len(self._chunks)
        keys = []
        for i in range(0, n, batch_blocks):
            emb = embed_tokens(self._chunks[i:min(i + batch_blocks, n)].to(device))  # [b, block, d]
            indices = torch.arange(i, i + len(emb), device=device)
            mask = self.token_mask(indices).unsqueeze(-1)
            pooled = (emb * mask).sum(1) / mask.sum(1)
            keys.append(retriever.key_proj(pooled))
        self._hidden_cache.clear()
        self._cache_order.clear()
        self.block_keys = torch.cat(keys, dim=0) if keys else None
        self.n_blocks = n
        # P0：写入 index manifest（key encoder 指纹 + 规模）
        self.metadata.update({
            "key_encoder_hash": encoder_hash(retriever, embed_tokens),
            "block_size": int(self.block_size),
            "n_blocks": int(self.n_blocks),
            "index_built": True,
        })
        return self.block_keys

    def set_manifest(self, tokenizer=None, model_version=None):
        """补齐 tokenizer / 模型版本（部署加载时必须可核对）。"""
        if tokenizer is not None:
            self.metadata["tokenizer"] = str(tokenizer)
        if model_version is not None:
            self.metadata["model_version"] = str(model_version)
        return self.metadata

    def assert_compatible(self, expected_key_hash=None, expected_block_size=None,
                          expected_tokenizer=None, expected_model_version=None):
        """加载索引前校验 manifest（P0）：不一致直接报错，禁止静默降精度。"""
        if not self.metadata.get("index_built"):
            raise RuntimeError("Index has not been built")
        if expected_key_hash and self.metadata.get("key_encoder_hash") != expected_key_hash:
            raise RuntimeError(
                f"Index manifest key_encoder_hash 不匹配: "
                f"index={self.metadata.get('key_encoder_hash')} model={expected_key_hash}")
        if expected_block_size and int(self.metadata.get("block_size", -1)) != int(expected_block_size):
            raise RuntimeError(
                f"Index block_size 不匹配: {self.metadata.get('block_size')} vs {expected_block_size}")
        if expected_tokenizer and self.metadata.get("tokenizer") != expected_tokenizer:
            raise RuntimeError(
                f"Index tokenizer 不匹配: {self.metadata.get('tokenizer')} vs {expected_tokenizer}")
        if expected_model_version and self.metadata.get("model_version") != expected_model_version:
            raise RuntimeError(f"Index model_version mismatch: {self.metadata.get('model_version')} vs {expected_model_version}")
        return True

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
            self.encoded_blocks += len(missing)
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
