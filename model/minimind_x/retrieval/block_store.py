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


def _tensor_hash(t: torch.Tensor) -> str:
    """key_proj 权重指纹（用于 index manifest，防静默错配）。"""
    if t is None:
        return "none"
    h = hashlib.sha1(t.detach().float().cpu().numpy().tobytes()).hexdigest()
    return h[:16]


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
        # P0 Index Manifest：加载侧校验，防模型/索引版本错配
        self.metadata = {
            "key_encoder_hash": None,
            "block_size": int(block_size),
            "n_blocks": int(self.n_blocks),
            "tokenizer": None,
            "model_version": None,
            "index_built": False,
        }

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
        # P0：写入 index manifest（key encoder 指纹 + 规模）
        self.metadata.update({
            "key_encoder_hash": _tensor_hash(getattr(retriever, "key_proj", None)
                                            and retriever.key_proj.weight),
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
                          expected_tokenizer=None):
        """加载索引前校验 manifest（P0）：不一致直接报错，禁止静默降精度。"""
        if not self.metadata.get("index_built") and self.block_keys is not None:
            pass
        if expected_key_hash and self.metadata.get("key_encoder_hash") not in (None, expected_key_hash):
            raise RuntimeError(
                f"Index manifest key_encoder_hash 不匹配: "
                f"index={self.metadata.get('key_encoder_hash')} model={expected_key_hash}")
        if expected_block_size and int(self.metadata.get("block_size", -1)) != int(expected_block_size):
            raise RuntimeError(
                f"Index block_size 不匹配: {self.metadata.get('block_size')} vs {expected_block_size}")
        if expected_tokenizer and self.metadata.get("tokenizer") not in (None, expected_tokenizer):
            raise RuntimeError(
                f"Index tokenizer 不匹配: {self.metadata.get('tokenizer')} vs {expected_tokenizer}")
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
