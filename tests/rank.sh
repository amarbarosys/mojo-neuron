#!/usr/bin/env bash
# Gate: rank vs the hand-built rollout golden, plus two mutants (no base row, no controls) that must change the answer.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}
"$M" build src/rank.mojo -I src -I "$J" -o .work/rank || { echo "FAIL build: rank"; exit 1; }
G=tests/rank_golden.jsonl
out=$(.work/rank "$G")
want() { # neuron mass pct
  echo "$out" | awk -F'\t' -v n="$1" -v m="$2" -v p="$3" '$1==n{d=$2-m; e=$3-p; if(d<0)d=-d; if(e<0)e=-e; if(d<1e-9&&e<1e-9)ok=1} END{exit !ok}' \
    || { echo "FAIL golden: neuron $1 want mass $2 pct $3, got: $(echo "$out" | awk -F'\t' -v n="$1" '$1==n')"; exit 1; }; }
want 7 0.8333333333333334 1
want 9 0.4166666666666667 0.3333333333333333
want ctrl0 0.5 0.8333333333333334
[ "$(echo "$out" | head -1 | cut -f1)" = 7 ] || { echo "FAIL order: neuron 7 is not first"; exit 1; }
echo "$out" | awk -F'\t' '$1==7{print $6}' | grep -q '^snake:6 ' || { echo "FAIL top words of neuron 7"; exit 1; }
echo "ok   golden (3 units, order, top words)"
# mutant 1: drop the base row -> "nice" is no longer removed, neuron 7 mass must change
grep -v '"neuron":"base"' $G > .work/rank_m1.jsonl
m1=$(.work/rank .work/rank_m1.jsonl | awk -F'\t' '$1==7{print $2}')
[ "$m1" != 0.8333333333333334 ] || { echo "FAIL mutant: base removal has no effect"; exit 1; }
# mutant 2: drop controls -> pct must become -1
grep -v ctrl $G > .work/rank_m2.jsonl
.work/rank .work/rank_m2.jsonl | awk -F'\t' '$1==7{print $3}' | grep -q '^-1' || { echo "FAIL mutant: pct without controls"; exit 1; }
echo "ok   mutants (no base, no controls) differ"
echo PASS rank
