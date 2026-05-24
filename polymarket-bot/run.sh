#!/usr/bin/env bash
# One-command launcher for polymarket-bot.
#   ./run.sh           — set up venv + deps if needed, run preflight + bot
#   ./run.sh --check   — only run preflight checks, exit
#
# Designed for macOS but works on Linux. Requires Python 3.11+.

set -euo pipefail

cd "$(dirname "$0")"

PYBIN="${PYTHON:-}"
if [[ -z "$PYBIN" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYBIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYBIN" ]]; then
  echo "ERROR: Python 3.11+ not found. Install with: brew install python@3.11"
  exit 1
fi

# Refuse Python < 3.11 (we use modern type syntax).
PYVER=$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PYMAJOR=${PYVER%%.*}
PYMINOR=${PYVER##*.}
if [[ "$PYMAJOR" -lt 3 || ( "$PYMAJOR" -eq 3 && "$PYMINOR" -lt 11 ) ]]; then
  echo "ERROR: Need Python 3.11+, found $PYVER ($PYBIN)"
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo ">> Creating virtualenv (.venv) with $PYBIN ($PYVER)..."
  "$PYBIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Install / refresh deps. pip is fast on a no-op so this is cheap to run every time.
echo ">> Ensuring dependencies..."
pip install -q --disable-pip-version-check -r requirements.txt

if [[ ! -f config.yaml ]]; then
  echo "ERROR: config.yaml not found. (It should ship with the project.)"
  exit 1
fi

# Load .env if present (live mode credentials).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${1:-}" == "--check" ]]; then
  echo ">> Running preflight only..."
  exec python -m src.main --preflight-only
fi

echo ">> Starting bot. Press Ctrl+C to stop."
exec python -m src.main "$@"
