#!/usr/bin/env bash
# Start the app. Run setup.sh first if you haven't.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  printf '\n\033[31mNot set up yet.\033[0m Run ./setup.sh first.\n\n' >&2
  exit 1
fi

exec ./.venv/bin/python app.py
