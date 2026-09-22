#!/usr/bin/env bash
# H20 8 卡启动脚本（任务清单 T7.5；Doc 5 §27）
# 用法：bash scripts/launch_h20_8card.sh <stage> [额外参数...]
#   stage ∈ {pretrain, longctx, sft_a, sft_b, sft_c, grpo1, grpo2, grpo3}
set -e
cd "$(dirname "$0")/.."

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
STAGE=${1:?用法: launch_h20_8card.sh <pretrain|longctx|sft_a|sft_b|sft_c|grpo1|grpo2|grpo3> [args...]}
shift || true

case "$STAGE" in
  pretrain) CMD="trainer/train_pretrain_x.py --variant minimind_x --preset M --loop_mode full_unroll --grad_ckpt 1 $*" ;;
  longctx)  CMD="trainer/train_long_context.py --variant dense --preset M --stage_lens 8192,32768,131072 --steps_per_stage 2000,2000,2000 --grad_ckpt 1 $*" ;;
  sft_a)    CMD="trainer/train_sft_x.py --phase a --variant minimind_x --preset M $*" ;;
  sft_b)    CMD="trainer/train_sft_x.py --phase b --variant minimind_x --preset M --full_unroll_ratio 0.25 --alpha_consistency 0.1 $*" ;;
  sft_c)    CMD="trainer/train_sft_x.py --phase c --variant minimind_x --preset M --oracle_rate 0.1 --train_halt_router 1 $*" ;;
  grpo1)    CMD="trainer/train_grpo_x.py --rl_phase 1 --variant minimind_x --preset M $*" ;;
  grpo2)    CMD="trainer/train_grpo_x.py --rl_phase 2 --variant minimind_x --preset M $*" ;;
  grpo3)    CMD="trainer/train_grpo_x.py --rl_phase 3 --variant minimind_x --preset M $*" ;;
  *) echo "未知 stage: $STAGE"; exit 1 ;;
esac

echo "[launch] torchrun --nproc_per_node=$NPROC $CMD"
exec torchrun --nproc_per_node="$NPROC" --nnodes="$NNODES" $CMD
