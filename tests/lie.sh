#!/usr/bin/env bash
# Gate F24 (fit half): on a synthetic capture with a planted truth direction at L3 and pure noise at L7, lie_fit finds L3 (val, test >= 0.95),
# leaves L7 near chance, and a mutant (labels shuffled across requests) cannot reach 0.8, so the check can fail. CPU only, seconds.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
timeout 280 "$M" build src/probe/lie_fit.mojo -I src -I "$GG" -I "$JS" -I "$LG" -o .work/lie_fit 2> .work/lie_fit.log || { grep -A5 -m3 error .work/lie_fit.log | cut -c1-240; echo "FAIL build: lie_fit"; exit 1; }
S=.work/lie-synth; rm -rf $S; python3 -I tools/lie-synth.py $S 300
run() { mkdir -p "$S/$1"; NX_RLOG=off .work/lie_fit $S $2 $S/$1 > $S/$1/log.txt 2>&1 || true; [ -s $S/$1/fit.json ] || { echo "FAIL fit: $1 wrote no fit.json: $(tail -1 $S/$1/log.txt)"; exit 1; }; }
run ok $S/labels.tsv
python3 -I - "$S/labels.tsv" > $S/shuf.tsv <<'PY'
import sys, random
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
ys = [r[0] for r in rows]; random.Random(3).shuffle(ys)
for r, y in zip(rows, ys): print("\t".join([y] + r[1:]))
PY
run mut $S/shuf.tsv
q() { jq -r "$2" "$S/$1/fit.json"; }
[ "$(q ok '.best_layer')" = 3 ] || { echo "FAIL fit: best layer $(q ok '.best_layer'), want the planted L3"; exit 1; }
[ "$(q ok '.layers[0] | (.val >= 0.95 and .test >= 0.95 and .pairwise_test >= 0.95)')" = true ] || { echo "FAIL fit: planted layer below 0.95"; exit 1; }
[ "$(q ok '.layers[1] | (.test > 0.35 and .test < 0.65)')" = true ] || { echo "FAIL fit: noise layer not near chance"; exit 1; }
[ "$(q mut '.real_test < 0.8')" = true ] || { echo "FAIL mutant: shuffled labels still reach 0.8, the gate cannot fail"; exit 1; }
# nx probe: the live score per token on the same capture: request 1 (true) ends above 0, request 2 (false) below, 3 tokens each, L7 probe cannot separate
timeout 280 "$M" build src/nx.mojo -I src -I "$GG" -I "$JS" -I "$LG" -o .work/nx 2> .work/nx-build.log || { grep -A5 -m3 error .work/nx-build.log | cut -c1-240; echo "FAIL build: nx"; exit 1; }
np() { NX_RLOG=off .work/nx probe "{\"cap\":\"$S\",\"dir\":\"$S/ok\",\"req\":$1,\"layer\":$2}"; }
[ "$(np 1 3 | jq '(.scores | length == 3) and (.last > 1)')" = true ] || { echo "FAIL nx probe: true request does not end above 1 at L3"; exit 1; }
[ "$(np 2 3 | jq '.last < -1')" = true ] || { echo "FAIL nx probe: false request does not end below -1 at L3"; exit 1; }
ok=0; for r in 1 2 3 4 5 6 7 8; do v=$(np $r 7 | jq '.last > 0'); [ "$v" = "$([ $((r % 2)) = 1 ] && echo true || echo false)" ] && ok=$((ok + 1)); done
[ "$ok" -le 7 ] || { echo "FAIL nx probe mutant: the noise-layer probe separates 8 of 8 requests, the check cannot fail"; exit 1; }
echo "PASS lie: planted L3 recovered (val/test/pairwise >= 0.95), noise L7 near chance, shuffled-label mutant stays under 0.8, nx probe scores true above and false below"
