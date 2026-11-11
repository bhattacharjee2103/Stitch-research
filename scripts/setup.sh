#!/usr/bin/env bash
# Isolated, repeatable setup. No global package or system-tool installation.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p .cache/pycache .cache/matplotlib .cache/pip results/logs vendor models
export PYTHONPYCACHEPREFIX="$project_root/.cache/pycache"
export MPLCONFIGDIR="$project_root/.cache/matplotlib"
export XDG_CACHE_HOME="$project_root/.cache"
export PIP_CACHE_DIR="$project_root/.cache/pip"
export TORCH_HOME="$project_root/models/torch"
if [ ! -x .venv/bin/python ]; then
  chosen_python="${PYTHON:-}"
  if [ -z "$chosen_python" ]; then
    for candidate in python3.12 python3.11 python3.10 python3.9 /usr/bin/python3; do
      if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys;assert (3,9)<=sys.version_info[:2]<=(3,12)' 2>/dev/null; then
        chosen_python="$candidate"
        break
      fi
    done
  fi
  if [ -z "$chosen_python" ]; then
    echo 'Install CPython 3.9--3.12 or set PYTHON=/path/to/supported/python.' >&2
    exit 1
  fi
  "$chosen_python" -m venv .venv
fi
py=.venv/bin/python
"$py" -c 'import sys;assert (3,9)<=sys.version_info[:2]<=(3,12),"Use CPython 3.9--3.12"'
"$py" scripts/run_logged.py "$py" -m pip install --upgrade pip
"$py" scripts/run_logged.py "$py" -m pip install -r requirements.txt
lightglue_revision=5a650c444718caffc2669dc0094cde908f42dd73
cosplace_revision=52b56e95ea62245281281f3bafd7b9390d19a0fd
if [ ! -d vendor/LightGlue/.git ]; then
  "$py" scripts/run_logged.py git clone https://github.com/cvg/LightGlue.git vendor/LightGlue
  "$py" scripts/run_logged.py git -C vendor/LightGlue checkout "$lightglue_revision"
fi
if [ ! -d vendor/CosPlace/.git ]; then
  "$py" scripts/run_logged.py git clone https://github.com/gmberton/CosPlace.git vendor/CosPlace
  "$py" scripts/run_logged.py git -C vendor/CosPlace checkout "$cosplace_revision"
fi
test "$(git -C vendor/LightGlue rev-parse HEAD)" = "$lightglue_revision"
test "$(git -C vendor/CosPlace rev-parse HEAD)" = "$cosplace_revision"
"$py" scripts/run_logged.py "$py" -m pip install -e . --no-deps
"$py" scripts/run_logged.py "$py" -m pip install -e vendor/LightGlue --no-deps
"$py" scripts/run_logged.py "$py" scripts/download_models.py
"$py" scripts/run_logged.py "$py" scripts/snapshot_environment.py
echo 'Setup complete. Next: make data'
