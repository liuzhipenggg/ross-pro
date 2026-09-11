#!/usr/bin/env bash
# Usage: source /path/to/ross-pro/activate_ross.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/.venv-ross/bin/activate"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
echo "ROSS_ROOT=$ROSS_ROOT"
echo "python=$(which python) $(python -V 2>&1)"
