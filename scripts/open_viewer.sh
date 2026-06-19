#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x "venv/bin/mjpython" ]; then
  echo "MuJoCo viewer launcher is missing. Setting up the virtual environment first."
  scripts/setup_venv.sh
fi

exec venv/bin/mjpython main.py --viewer "$@"
