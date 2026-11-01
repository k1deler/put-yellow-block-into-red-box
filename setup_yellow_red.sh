#!/usr/bin/env bash
# Yellow block -> red container only. Copies the already existing trusted model.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/setup_robot_copy.sh" \
  /home/team1/fufu/yam_infer/models/act_ref/block_bowl_delta.pt put_block_in_bowl
