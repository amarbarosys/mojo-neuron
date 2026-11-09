#!/usr/bin/env bash
# Build lane-agent tools into .work/ (dream).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; P=$HOME/Projects/mojo; B=$P/mojo-baro
for t in dream; do
  for mut in 0 1; do
    out=.work/$t; d=""; [ "$mut" = 1 ] && { out=.work/${t}_m; d="-D DREAM_MUT=1"; }
    timeout 300 "$M" build src/agent/$t.mojo $d -I src -I $P/mojo-gguf/src -I $P/mojo-json/src -I $P/mojo-log/src -I $B/serve -I $B \
      -I $P/mojo-subprocess/src -o $out 2> $out-build.log || { grep -A5 -m3 error $out-build.log | cut -c1-240; echo "FAIL build: $t$d"; exit 1; }
  done
done
