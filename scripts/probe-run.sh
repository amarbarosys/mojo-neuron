#!/usr/bin/env bash
# F24 end to end after the pairs exist (probe-gen.sh, probe-verify.sh): capture -> fit (bars 1, 2) -> steering run (bars 3, 4). The probes land in
# the shared store (default ~/Projects/mojo/mojo-neuron/.work/probe/lie, where `nx probe` looks). usage: probe-run.sh [OUT=.work/f24] [MAXPAIRS]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-.work/f24}; MAXP=${2:-100000}; MAIN=${MAIN:-$HOME/Projects/mojo/mojo-neuron}
CAP=${CAP:-$MAIN/.work/cap/qwythos-lie}; PD=${PD:-$MAIN/.work/probe/lie}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
for t in lie_fit lie_steer; do timeout 280 "$M" build src/probe/$t.mojo -I src -I "$GG" -I "$JS" -I "$LG" -o .work/$t 2> .work/$t.log || { grep -A5 -m3 error .work/$t.log | cut -c1-240; echo "FAIL build: $t"; exit 1; }; done
[ -s "$CAP/req.i32" ] && [ -s "$OUT/labels.tsv" ] || CAP=$CAP scripts/probe-capture.sh "$OUT/pairs.tsv" "$OUT" "$MAXP"
mkdir -p "$PD"
.work/lie_fit "$CAP" "$OUT/labels.tsv" "$PD" | tail -3 || echo "NOTE fit: a prereg bar 1 or 2 missed (reported, steering still runs)"
scripts/probe-steer.sh "$CAP" "$OUT/labels.tsv" "$PD" "$OUT/steer"
