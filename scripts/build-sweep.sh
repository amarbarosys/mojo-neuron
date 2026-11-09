#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}
for t in sweep reverse; do timeout 300 "$M" build src/$t.mojo -I src -I "${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}" -I "$BARO" -I "$BARO/serve" \
  -I "$HOME/Projects/mojo/mojo-subprocess/src" -o .work/$t 2> .work/$t-build.log || { grep -A5 -m3 error .work/$t-build.log | cut -c1-240; echo "FAIL build: $t"; exit 1; }; done
