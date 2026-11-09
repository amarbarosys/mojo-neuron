#!/usr/bin/env bash
# Build the window app: .work/app. Needs mojo-imkit sources equal to tag v0.8.0 (local path), mojo-gguf main, mojo-json.
set -euo pipefail
cd "$(dirname "$0")/.."
IMKIT=${IMKIT:-$HOME/Projects/mojo/mojo-imkit/src}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}
M=${MOJO:-$HOME/Projects/mojo/mojo-imkit/.venv/bin/mojo}
git -C "$IMKIT/.." diff --quiet v0.8.0 -- src || { echo "FAIL build: $IMKIT differs from mojo-imkit tag v0.8.0 in src/"; exit 1; }
mkdir -p .work; rm -f .work/app  # a failed build must not leave the old binary to pass
timeout 400 "$M" build -I src -I "$IMKIT" -I "$GG" -I "$J" -I "${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}" src/app.mojo -o .work/app 2>&1 | { grep -E "error" -A4 || true; }
[ -x .work/app ] || { echo "FAIL build: no .work/app"; exit 1; }
echo "built .work/app"
