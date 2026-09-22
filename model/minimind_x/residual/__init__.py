# V2 残差改造三模块（任务清单 P1；V2 文档 §6-8 + 附录 A 参考实现移植）
from .loop_mhar import LoopMHAR
from .recurrent_update import RecurrentUpdateGate

__all__ = ["LoopMHAR", "RecurrentUpdateGate"]
