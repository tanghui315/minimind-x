#!/usr/bin/env bash
# Stage-0 预训练的唯一命令来源（single source of truth）：
# night_runner / 看门狗 / 手动冒烟一律调用本脚本，杜绝「命令不同源」事故（2026-09-18 教训）。
# 用法: run_stage0.sh [额外 train_pretrain_x 参数...]
# 环境要求: 先 export PYTHONPATH=/data/llm-infra/venvs/vllm/lib/python3.12/site-packages（或自行适配）
set -e
cd "$(dirname "$0")/.."
PY=${MMX_PYTHON:-./venv/bin/python}
NPROC=${MMX_NPROC:-8}
PORT=${MMX_PORT:-29517}

exec $PY -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$PORT \
  trainer/train_pretrain_x.py \
  --variant middle_loop --hidden_size 768 --num_hidden_layers 8 \
  --max_loop 4 --loop_mode full_unroll --alpha_consistency 0.1 \
  --batch_size 16 --accumulation_steps 2 --max_seq_len 768 \
  --learning_rate 5e-4 --grad_ckpt 1 --num_workers 4 \
  --log_interval 50 --save_interval 500 \
  --tokenizer_path model \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain_x --save_dir out "$@"
