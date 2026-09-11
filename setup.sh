#!/usr/bin/env bash
# One-time setup for macOS and Linux. Run:  ./setup.sh
# Then start the app with:  ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m   %s\n' "$*"; }
die()  { printf '\n\033[31mSetup stopped:\033[0m %s\n\n' "$*" >&2; exit 1; }

say "1/3  Checking Python"
PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    # 3.9 is the floor: the code uses dict-ordering and typing features below that.
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      PY="$candidate"; break
    fi
  fi
done
[ -n "$PY" ] || die "Python 3.9 or newer not found. Install it from https://www.python.org/downloads/ and run this again."
ok "$("$PY" --version)"

say "2/3  Installing Python packages into .venv"
[ -d .venv ] || "$PY" -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet -r requirements.txt
ok "packages installed"

say "3/3  Checking Tesseract (needed only to read Form 6 scans)"
if command -v tesseract >/dev/null 2>&1; then
  ok "$(tesseract --version 2>&1 | head -1)"
else
  warn "Tesseract is not installed."
  if [ "$(uname -s)" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      printf '      Install it now with Homebrew? [y/N] '
      read -r reply </dev/tty || reply=""
      case "$reply" in
        [yY]*) brew install tesseract && ok "Tesseract installed" ;;
            *) warn "Skipped — run 'brew install tesseract' before uploading a Form 6." ;;
      esac
    else
      warn "Install Homebrew from https://brew.sh, then run: brew install tesseract"
    fi
  elif command -v apt-get >/dev/null 2>&1; then
    warn "Run: sudo apt-get install -y tesseract-ocr"
  else
    warn "Install the 'tesseract' package using your system's package manager."
  fi
  warn "Everything except the Form 6 upload works without it."
fi

say "Done. Start the app with:  ./run.sh"
