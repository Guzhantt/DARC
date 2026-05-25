#!/usr/bin/env bash
# Launch the web dashboard. Sets up venv + deps if needed, then opens browser.
#
#   ./run-web.sh           - start dashboard at http://localhost:8000
#   ./run-web.sh --port 9000 - pick a different port
#
# Once the dashboard is up, the bot itself is started/stopped from the UI.

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

echo ">> Ensuring dependencies..."
pip install -q --disable-pip-version-check -r requirements.txt

PORT=8000
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    *) shift;;
  esac
done

URL="http://localhost:${PORT}"
echo ">> Starting dashboard at ${URL}"
echo ">> Browser will open automatically. Stop with Ctrl+C."

# Open browser after server has time to start.
(
  sleep 2
  if command -v open >/dev/null 2>&1; then
    open "${URL}"             # macOS
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${URL}"         # Linux
  fi
) &

WEB_PORT="${PORT}" exec python -m web.app
