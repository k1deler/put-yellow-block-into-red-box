#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -t 0 ]] || { echo 'Use an interactive terminal (SSH: allocate a TTY with ssh -t)'; exit 1; }
[[ $# == 0 ]] || { echo 'Use the recorded project configs; CLI overrides are disabled here'; exit 2; }
python3 "$ROOT/tools/prepare_assets.py" --check
cd "$ROOT/project"
PY="$ROOT/project/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/project:$ROOT/project/third_party/i2rt"
unset CKPT YAM_LEFT_CAM_SERIAL YAM_RIGHT_CAM_SERIAL
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo 'This copy is already running'; exit 1; }
"$PY" "$ROOT/tools/verify_real.py" --check
"$PY" -m yam_infer.tools.free_hw --check --hw-config configs/hardware.yaml
echo 'The next step INITIALIZES DRIVERS and MOVES to the model start pose BEFORE waiting for Enter.'
echo 'q+Enter or Ctrl+C performs MOVING home before torque-off; it is not an immediate emergency stop.'
read -r -p 'On-site operator ready? Type START: ' answer
[[ "$answer" == START ]] || exit 1
exec "$PY" -u -m yam_infer.run --models-config configs/models.yaml --hw-config configs/hardware.yaml \
  --run-config configs/run.yaml --device cuda --goto-start task_mean --park home --record-dir runs_infer
