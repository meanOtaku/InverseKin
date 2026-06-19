#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x "venv/bin/python" ]; then
  echo "Virtual environment is missing. Setting it up first."
  scripts/setup_venv.sh
fi

exec venv/bin/python main.py "$@"
