#!/usr/bin/env bash
# Independent environment and task-model registration.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$ROOT/tools/prepare_assets.py"
[[ $# == 2 ]] || { echo 'Usage: bash setup_robot_copy.sh /absolute/path/to/TRUSTED_REAL.pt task_name'; exit 2; }
CKPT_INPUT="$1"
TASK="$2"
[[ -f "$CKPT_INPUT" && "$CKPT_INPUT" == /* ]] || { echo 'Need an absolute path to a trusted trained checkpoint'; exit 1; }
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo 'Runner is active; model replacement refused. Stop it normally first.'; exit 1; }
cd "$ROOT/project"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x .venv/bin/python ]]; then
    if command -v uv >/dev/null; then uv venv .venv --python /usr/bin/python3.10
    else /usr/bin/python3.10 -m venv .venv; fi
fi
PY="$ROOT/project/.venv/bin/python"
pip_install() {
    if command -v uv >/dev/null; then uv pip install --python "$PY" "$@"
    else "$PY" -m pip install "$@"; fi
}
# Validated torch/CUDA combination for RTX 50xx.
pip_install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url https://download.pytorch.org/whl/cu128
pip_install -r requirements.txt pytest ruff --index-url "${PIP_MIRROR:-https://pypi.org/simple}"
export PYTHONPATH="$ROOT/project:$ROOT/project/third_party/i2rt"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p "$ROOT/evidence"
# Validate the single-task source before model registration.
if [[ ! -e "$ROOT/evidence/registered_model.json" ]]; then
    "$PY" -m pytest -q -p no:cacheprovider 2>&1 | tee "$ROOT/evidence/upstream_full_tests.log"
fi
# The task profile pins this new checkpoint. Replacement saves the old model and
# validation record and does not rewrite existing execution/control settings.
"$PY" "$ROOT/tools/register_model.py" --checkpoint "$CKPT_INPUT" --task "$TASK" --replace
bash "$ROOT/validate_real.sh"
echo 'TASK_SETUP_PASSED; no real devices have been opened.'
