#!/usr/bin/env bash
# F24 step 3: capture every statement of pairs.tsv (true, false, ...). Writes $OUT/labels.tsv (label, pair, text: pair p = true then false) and the
# capture into $CAP (default: the shared main-checkout .work/cap/qwythos-lie), request 2p+1 = true, 2p+2 = false. usage: probe-capture.sh PAIRS_TSV OUT [MAXPAIRS]
set -euo pipefail
cd "$(dirname "$0")/.."
PAIRS=${1:?usage: probe-capture.sh PAIRS_TSV OUT [MAXPAIRS]}; OUT=${2:?}; MAXP=${3:-100000}
CAP=${CAP:-$HOME/Projects/mojo/mojo-neuron/.work/cap/qwythos-lie}
[ -s "$PAIRS" ] || { echo "FAIL setup: $PAIRS missing or empty"; exit 1; }
mkdir -p "$OUT"
head -n "$MAXP" "$PAIRS" | awk -F'\t' '{ printf "1\t%d\t%s\n0\t%d\t%s\n", NR-1, $1, NR-1, $2 }' > "$OUT/labels.tsv"
cut -f3 "$OUT/labels.tsv" > "$OUT/texts.txt"
LABEL=probe-capture exec scripts/cap-texts.sh "$OUT/texts.txt" "$CAP" "$OUT"
