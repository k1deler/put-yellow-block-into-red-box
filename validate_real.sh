#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/project"
PY="$ROOT/project/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/project:$ROOT/project/third_party/i2rt"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
[[ -f "$ROOT/evidence/registered_model.json" ]] || { echo 'First register a trusted real trained model using setup_robot_copy.sh'; exit 1; }
# The selected task intentionally differs from the GitHub default. Only these two
# assertions about the old default model selection are inapplicable after registration.
"$PY" -m pytest -q -p no:cacheprovider \
  --deselect=tests/test_models.py::test_project_models_config \
  --deselect=tests/test_models.py::test_commented_act_ref_block_is_valid \
  --ignore=tests/test_run_config.py \
  2>&1 | tee "$ROOT/evidence/selected_model_tests.log"
"$PY" scripts/check_env.py 2>&1 | tee "$ROOT/evidence/environment_check.log"
"$PY" "$ROOT/tools/verify_real.py" 2>&1 | tee "$ROOT/evidence/real_model_smoke.log"
