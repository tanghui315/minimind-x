"""MiniMind-X 训练公共工具（各 X trainer 共享；不改动原 trainer_utils.py）。"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.model_minimind import precompute_freqs_cis
from model.minimind_x import MiniMindXForCausalLM
from trainer.trainer_utils import Logger, get_model_params


def x_weight_path(save_dir, name, hidden_size):
    return f'{save_dir}/{name}_{hidden_size}_x.pth'


def init_model_x(lm_config, from_weight='pretrain', tokenizer_path='../model',
                 save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindXForCausalLM(lm_config)
    if from_weight != 'none':
        path = x_weight_path(save_dir, from_weight, lm_config.hidden_size)
        weights = torch.load(path, map_location=device)
        meta = weights.pop("__meta__", None)  # T5.2 manifest 不进 state_dict
        if meta:
            Logger(f'ckpt meta: arch={meta.get("architecture_version")} '
                   f'params={meta.get("param_manifest", {}).get("total")}')
        missing, unexpected = model.load_state_dict(weights, strict=False)
        Logger(f'loaded X weights from {path} (missing={len(missing)}, unexpected={len(unexpected)})')
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


def param_manifest(model) -> dict:
    """T5.2 / V2 §18.2：sum(p.numel()) 参数量 manifest（分模块）。"""
    raw = model.module if hasattr(model, 'module') else model
    raw = getattr(raw, '_orig_mod', raw)
    by_module = {}
    total = 0
    for n, p in raw.named_parameters():
        total += p.numel()
        top = n.split('.')[0] if '.' in n else n
        # 细分到二级（model.thinking_core.update_gate 等）
        parts = n.split('.')
        key = '.'.join(parts[:3]) if len(parts) >= 3 else n
        by_module[key] = by_module.get(key, 0) + p.numel()
    # T4.3：参数组清单（new-module LR 倍率 / no-decay 标记）
    RESIDUAL_KEYS = ("loop_mhar", "update_gate", "memory_reader",
                     "halt_router", "retrieve_router", "budget_router", "retriever")
    NO_DECAY_HINTS = ("scale_logit", "loop_mhar.query", "type_bias")
    groups = {"backbone_decay": 0, "backbone_no_decay": 0,
              "residual_decay": 0, "residual_no_decay": 0}
    for n, pr in raw.named_parameters():
        if not pr.requires_grad:
            continue
        is_residual = any(k in n for k in RESIDUAL_KEYS)
        no_decay = pr.ndim <= 1 or any(h in n for h in NO_DECAY_HINTS)
        key = ("residual_" if is_residual else "backbone_") + ("no_decay" if no_decay else "decay")
        groups[key] += pr.numel()
    return {"total": total, "by_module": by_module, "param_groups": groups,
            "architecture_version": raw.config.architecture_version()
            if hasattr(raw.config, "architecture_version") else "minimind_x"}


def validate_config(lm_config):
    """启动即校验（T5.2）。"""
    if hasattr(lm_config, "validate"):
        lm_config.validate()
    return lm_config


def save_x_checkpoint(model, args_save_dir, save_weight, hidden_size,
                      optimizer=None, scaler=None, extra=None):
    """T5.2 / V2 §19：完整 checkpoint——权重 + config/架构/缓存版本 + optimizer/scaler
    + 各 rank RNG + 数据与 warm-up 进度 + source/tokenizer manifest + 参数量 manifest。
    """
    import random
    os.makedirs(args_save_dir, exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    raw = getattr(raw, '_orig_mod', raw)
    payload = {k: v.half().cpu() for k, v in raw.state_dict().items()}
    meta = {
        "param_manifest": param_manifest(model),
        "architecture_version": raw.config.architecture_version()
        if hasattr(raw.config, "architecture_version") else "minimind_x",
        "cache_version": 1,
        "config": raw.config.to_dict() if hasattr(raw.config, "to_dict") else {},
    }
    # 各 rank RNG（可复现）
    try:
        import numpy as np
        meta["rng_numpy"] = np.random.get_state()
    except Exception:
        meta["rng_numpy"] = None
    meta["rng_python"] = random.getstate()
    meta["rng_torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        meta["rng_cuda"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        meta["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        meta["scaler"] = scaler.state_dict()
    if extra:
        # 允许传入 data_progress / warmup_progress / tokenizer / source manifest
        meta.update(extra)
    payload["__meta__"] = meta
    path = x_weight_path(args_save_dir, save_weight, hidden_size)
    torch.save(payload, path)
    return path


def load_x_checkpoint_meta(path):
    """读回 __meta__（optimizer/RNG/进度等），权重键由 load_state_dict 负责。"""
    weights = torch.load(path, map_location="cpu")
    meta = weights.pop("__meta__", None)
    return weights, meta


def restore_training_state(meta, optimizer=None, scaler=None, model=None):
    """从 __meta__ 恢复 optimizer/scaler/RNG（T5.2 严格模式加载无静默冲突）。"""
    if meta is None:
        return {"restored": []}
    restored = []
    if optimizer is not None and meta.get("optimizer"):
        optimizer.load_state_dict(meta["optimizer"])
        restored.append("optimizer")
    if scaler is not None and meta.get("scaler"):
        scaler.load_state_dict(meta["scaler"])
        restored.append("scaler")
    import random
    if meta.get("rng_python"):
        random.setstate(meta["rng_python"])
        restored.append("rng_python")
    if meta.get("rng_torch") is not None:
        torch.set_rng_state(meta["rng_torch"].cpu() if torch.is_tensor(meta["rng_torch"])
                            else meta["rng_torch"])
        restored.append("rng_torch")
    if meta.get("rng_cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(meta["rng_cuda"])
            restored.append("rng_cuda")
        except Exception:
            pass
    if meta.get("rng_numpy") is not None:
        try:
            import numpy as np
            np.random.set_state(meta["rng_numpy"])
            restored.append("rng_numpy")
        except Exception:
            pass
    return {"restored": restored,
            "data_progress": meta.get("data_progress"),
            "warmup_progress": meta.get("warmup_progress"),
            "tokenizer_manifest": meta.get("tokenizer_manifest"),
            "source_manifest": meta.get("source_manifest"),
            "param_manifest": meta.get("param_manifest"),
            "architecture_version": meta.get("architecture_version")}


def weighted_ce(logits, labels, weight=None):
    """逐 token 加权 CE（retrieval-aware 辅助损失：gold/needle 位置加权，T3.3）。"""
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    ce = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1),
                         ignore_index=-100, reduction="none").view(y.shape)
    mask = (y != -100).float()
    if weight is None:
        return (ce * mask).sum() / mask.sum().clamp(min=1)
    w = weight[..., 1:].contiguous()
    return (ce * w * mask).sum() / (w * mask).sum().clamp(min=1)


def apply_rope_stage(model, max_position_embeddings=None, rope_theta=None):
    """curriculum 阶段切换时重算 RoPE buffer（任务清单 D8）。兼容 DDP/compile 包装。"""
    raw = model
    for attr in ("module", "_orig_mod"):
        raw = getattr(raw, attr, raw)
    cfg = raw.config
    if max_position_embeddings is not None:
        cfg.max_position_embeddings = max_position_embeddings
    if rope_theta is not None:
        cfg.rope_theta = rope_theta
    cos, sin = precompute_freqs_cis(dim=cfg.head_dim, end=cfg.max_position_embeddings,
                                    rope_base=cfg.rope_theta, rope_scaling=cfg.rope_scaling)
    device = raw.model.freqs_cos.device
    raw.model.freqs_cos, raw.model.freqs_sin = cos.to(device), sin.to(device)


def set_gradient_checkpointing(model, enabled=True):
    """per-loop / per-layer 梯度检查点（任务清单 T1.4）：R 大时省激活显存。兼容 DDP/compile。"""
    raw = model
    for attr in ("module", "_orig_mod"):
        raw = getattr(raw, attr, raw)
    for stack in (raw.model.prelude.stack, raw.model.coda.stack):
        stack.gradient_checkpointing = enabled
    raw.model.thinking_core.gradient_checkpointing = enabled


def build_optimizer(model, lr, weight_decay=0.01, residual_lr_mult=2.0):
    """V2 参数组优化器（任务清单 T4.3；V2 §14.4）。

    新模块（loop_mhar/update_gate/memory_reader/routers/retriever）用 1-2× LR；
    norm/bias/gate scale/pseudo-query 等一维或特殊参数不施加 weight decay。
    """
    raw = model
    for attr in ("module", "_orig_mod"):
        raw = getattr(raw, attr, raw)
    RESIDUAL_KEYS = ("loop_mhar", "update_gate", "memory_reader",
                     "halt_router", "retrieve_router", "budget_router", "retriever")
    NO_DECAY_HINTS = ("scale_logit", "loop_mhar.query", "type_bias")
    groups = {"bd": [], "bnd": [], "rd": [], "rnd": []}
    for n, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        is_residual = any(k in n for k in RESIDUAL_KEYS)
        no_decay = p.ndim <= 1 or any(h in n for h in NO_DECAY_HINTS)
        key = ("r" if is_residual else "b") + ("nd" if no_decay else "d")
        groups[key].append(p)
    param_groups = [{"params": groups["bd"], "lr": lr, "weight_decay": weight_decay, "mult": 1.0},
                    {"params": groups["bnd"], "lr": lr, "weight_decay": 0.0, "mult": 1.0}]
    if groups["rd"]:
        param_groups.append({"params": groups["rd"], "lr": lr * residual_lr_mult,
                             "weight_decay": weight_decay, "mult": residual_lr_mult})
    if groups["rnd"]:
        param_groups.append({"params": groups["rnd"], "lr": lr * residual_lr_mult,
                             "weight_decay": 0.0, "mult": residual_lr_mult})
    import torch as _torch
    return _torch.optim.AdamW(param_groups, lr=lr)


def tokenize_tensors(tokenizer, text, device="cpu", **kw):
    """tokenizer 输出 -> dict of tensors（绕开 transformers 在旧 torch 上的
    return_tensors='pt' 版本门禁；新环境同样适用）。"""
    enc = tokenizer(text, return_tensors=None, **kw)
    return {k: torch.tensor([v], device=device) for k, v in enc.items() if v is not None}


def save_pretrained_x(model, out_dir):
    """研究格式的『导出』：config.json + model.pth（T7.5）。

    注：完整 HF transformers 格式导出需要 remote-code 建模文件，超出研究平台范围；
    重新实例化用 MiniMindXConfig.from_json_file + load_state_dict(strict=False)。
    """
    import json
    os.makedirs(out_dir, exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    raw = getattr(raw, '_orig_mod', raw)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(raw.config.to_dict(), f, ensure_ascii=False, indent=2)
    torch.save({k: v.half().cpu() for k, v in raw.state_dict().items()},
               os.path.join(out_dir, "model_x.pth"))
    return out_dir


def load_pretrained_x(out_dir, device="cpu"):
    """save_pretrained_x 的对偶：读回 config + 权重。"""
    import json
    from model.minimind_x import MiniMindXConfig, MiniMindXForCausalLM
    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
        cfg = MiniMindXConfig(**{k: v for k, v in json.load(f).items()
                                 if k not in ("model_type", "transformers_version",
                                              "architectures")})
    model = MiniMindXForCausalLM(cfg)
    weights = torch.load(os.path.join(out_dir, "model_x.pth"), map_location="cpu")
    weights.pop("__meta__", None)
    model.load_state_dict(weights, strict=False)
    return model.to(device)
