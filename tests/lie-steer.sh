#!/usr/bin/env bash
# Gate F24 (steering half) on synthetic captures: a steered capture built by adding the prep vector at L3 and L7 passes every bar; three mutants
# (no effect, negated vector, effect at L3 only) each fail the bar that names them. CPU only, seconds.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
for t in lie_fit lie_steer; do timeout 280 "$M" build src/probe/$t.mojo -I src -I "$GG" -I "$JS" -I "$LG" -o .work/$t 2> .work/$t.log || { grep -A5 -m3 error .work/$t.log | cut -c1-240; echo "FAIL build: $t"; exit 1; }; done
S=.work/lie-steer-synth; rm -rf $S; python3 -I tools/lie-steer-synth.py cap $S
mkdir -p $S/probes $S/prep; export NX_RLOG=off
.work/lie_fit $S $S/labels.tsv $S/probes > $S/fit.log 2>&1 || true; [ -s $S/probes/fit.json ] || { echo "FAIL fit: $(tail -1 $S/fit.log)"; exit 1; }
.work/lie_steer prep $S $S/labels.tsv $S/probes $S/prep > $S/prep.log || { echo "FAIL prep: $(tail -1 $S/prep.log)"; exit 1; }
[ "$(jq -r '.Ls, .Lm' $S/prep/prep.json | tr '\n' ' ')" = "3 7 " ] || { echo "FAIL prep: Ls/Lm are not 3/7"; exit 1; }
[ "$(wc -l < $S/prep/req.jsonl)" = 900 ] && grep -q '"steer_file"' $S/prep/req.jsonl && head -1 $S/prep/req.jsonl | grep -qv steer_file || { echo "FAIL prep: req.jsonl is not 900 lines with a steer_file on every steered one"; exit 1; }
run() { python3 -I tools/lie-steer-synth.py steered $S $S/prep $S/st-$1 $1; .work/lie_steer eval $S $S/labels.tsv $S/probes $S/st-$1 $S/prep > $S/ev-$1.log 2>&1 && echo 0 || echo $?; }
[ "$(run ok)" = 0 ] || { echo "FAIL eval: the good steered capture missed a bar: $(tail -2 $S/ev-ok.log | tr '\n' ' ')"; exit 1; }
for m in none wrongsign applied; do
  [ "$(run $m)" = 2 ] || { echo "FAIL mutant: steered capture '$m' passed every bar, the gate cannot fail"; exit 1; }
done
b() { grep -q "bars: shift $2 sign .* applied $3" $S/ev-$1.log || { echo "FAIL mutant: '$1' did not fail the bars that name it (want shift $2, applied $3)"; tail -1 $S/ev-$1.log; exit 1; }; }
b none False False; b wrongsign False False; b applied False True   # L3-only: the steer is applied at Ls but never reaches the probe layer
echo "PASS lie-steer: prep picks L3/L7 and writes 900 requests, the planted steer passes all bars, no-effect and negated mutants fail shift and applied, the L3-only one fails shift alone"
