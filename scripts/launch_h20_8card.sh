#!/usr/bin/env bash
# Follow the checkpoint produced by the preceding stage. Paths in overrides are
# relative to the repository root, including paths with spaces.
set -eu
cd "$(dirname "$0")/.."
stage=${1:?expected pretrain, longctx, sft_a/b/c or grpo1/2/3}
shift
common=(--variant minimind_x --tokenizer_path model --save_dir out)
preset=${MMX_PRESET-M}
if [[ -n "$preset" ]]; then common+=(--preset "$preset"); fi
case "$stage" in
  pretrain)
    cmd=(trainer/train_pretrain_x.py --data_path dataset/pretrain_t2t_mini.jsonl --loop_mode full_unroll --grad_ckpt 1) ;;
  longctx)
    cmd=(trainer/train_long_context.py --data_path dataset/pretrain_t2t_mini.jsonl --from_weight pretrain_x --steps_per_stage 2000,2000,2000) ;;
  sft_a|sft_b|sft_c)
    phase=${stage#sft_}
    cmd=(trainer/train_sft_x.py --phase "$phase" --data_path dataset/sft_t2t_mini.jsonl --retrieval_data_path dataset/retrieval_qa.jsonl)
    if [[ "$phase" == a ]]; then cmd+=(--from_weight longctx_x_stage2); fi ;;
  grpo1|grpo2|grpo3)
    phase=${stage#grpo}
    cmd=(trainer/train_grpo_x.py --rl_phase "$phase" --data_path dataset/agent_rl_math.jsonl) ;;
  *) echo "Unknown stage: $stage" >&2; exit 1 ;;
esac
launcher=(torchrun)
if [[ -n "${MMX_PYTHON:-}" ]]; then launcher=("$MMX_PYTHON" -m torch.distributed.run); fi
exec "${launcher[@]}" --master_port="${MASTER_PORT:-29500}" --nproc_per_node="${NPROC:-8}" --nnodes="${NNODES:-1}" "${cmd[@]}" "${common[@]}" "$@"
