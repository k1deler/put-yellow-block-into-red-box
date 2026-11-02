#!/usr/bin/env bash
# Yellow block -> red container only. Supply a trusted checkpoint path.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -le 1 ]] || { echo 'Usage: bash setup_yellow_red.sh /absolute/path/to/model.pt'; exit 2; }
CHECKPOINT="${1:-${TASK_CHECKPOINT:-}}"
[[ -n "$CHECKPOINT" ]] || { echo 'Specify a trusted model path or TASK_CHECKPOINT; nothing started.'; exit 2; }
exec bash "$ROOT/setup_robot_copy.sh" "$CHECKPOINT" put_block_in_bowl
