#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x "venv/bin/python" ]; then
  python3 -m venv venv
fi

if venv/bin/python - <<'PY' >/dev/null 2>&1
import mujoco
import numpy
PY
then
  echo "Dependencies are already installed."
else
  venv/bin/python -m pip install --upgrade pip setuptools wheel
  venv/bin/python -m pip install -r requirements.txt
fi

echo "Virtual environment is ready: $PROJECT_ROOT/venv"
echo "Open viewer: python3 main.py"
echo "Generate data: python3 main.py --headless"
