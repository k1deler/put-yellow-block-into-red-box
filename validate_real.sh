#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$ROOT/tools/prepare_assets.py" --check
cd "$ROOT/project"
PY="$ROOT/project/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/project:$ROOT/project/third_party/i2rt"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
[[ -f "$ROOT/evidence/registered_model.json" ]] || { echo 'First register a trusted real trained model using setup_robot_copy.sh'; exit 1; }
# Defaults and tests both use the selected task directly.
"$PY" -m pytest -q -p no:cacheprovider \
  2>&1 | tee "$ROOT/evidence/selected_model_tests.log"
"$PY" scripts/check_env.py 2>&1 | tee "$ROOT/evidence/environment_check.log"
"$PY" "$ROOT/tools/verify_real.py" 2>&1 | tee "$ROOT/evidence/real_model_smoke.log"
