#!/usr/bin/env bash
# Both public entry points use the same pretraining defaults and checkpoint identity.
set -eu
export NPROC=${MMX_NPROC:-${NPROC:-8}}
export MASTER_PORT=${MMX_PORT:-${MASTER_PORT:-29517}}
if [[ -z "${MMX_PYTHON:-}" && -x "$(dirname "$0")/../venv/bin/python" ]]; then
  export MMX_PYTHON=./venv/bin/python
fi
exec bash "$(dirname "$0")/launch_h20_8card.sh" pretrain "$@"
