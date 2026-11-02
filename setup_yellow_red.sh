#!/usr/bin/env bash
# Yellow block -> red container only. Supply a trusted checkpoint path.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -le 1 ]] || { echo 'Usage: bash setup_yellow_red.sh /absolute/path/to/model.pt'; exit 2; }
CHECKPOINT="${1:-${TASK_CHECKPOINT:-$ROOT/input_models/yellow_red_seed42_selected.pt}}"
[[ -f "$CHECKPOINT" ]] || { echo "New trained checkpoint missing: $CHECKPOINT. Download yellow_red_seed42_selected.pt from the repository release or specify its absolute path; nothing started."; exit 2; }
exec bash "$ROOT/setup_robot_copy.sh" "$CHECKPOINT" put_block_in_bowl
