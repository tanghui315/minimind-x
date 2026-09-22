"""V2 迁移 warm-up 课程（任务清单 T4.2；V2 文档 §14.2）。

同一训练 run 内把 history_strength(η) 与 update_progress(λ) 从 0 平滑升至目标值：
  初始化检查点：η=0, λ=0 —— 与旧路径数值一致（严格兼容锚点，T2.3 已锁定）
  前 warmup_steps：线性升至 1×base —— 新模块渐进接入
  之后：保持 base —— 主训练

实现为「写共享 config 对象」：ThinkingCore 每轮读 config，改 config 即生效，
无需重建模型。日志口径：训练器应同时记录 raw 与 effective 值（V2 §14.2 提醒）。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def residual_progress(step, warmup_steps):
    """0→1 线性 ramp；warmup_steps<=0 视为从头训练（恒 1）。"""
    if warmup_steps is None or warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step) / float(warmup_steps)))


def apply_residual_progress(model, frac, base_strength=1.0, base_progress=1.0):
    """按 frac 设置 η/λ（base 为目标终值）。兼容 DDP/compile 包装。返回 (η, λ)。"""
    raw = model
    for attr in ("module", "_orig_mod"):
        raw = getattr(raw, attr, raw)
    cfg = raw.config
    eta = base_strength * frac
    lam = base_progress * frac
    cfg.mhar_strength = eta
    cfg.update_progress = lam
    return eta, lam
