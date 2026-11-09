#!/usr/bin/env bash
# Gate F24b (CPU, synthetic): lie_eval passes every bar when the sets carry the truth signal and the inverted set reads right; a probe that follows the surface
# form fails the inverted bar (S4) only; a set with no signal fails the family bars. The engine halves are probe-sets.sh.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
for t in lie_fit lie_eval; do timeout 280 "$M" build src/probe/$t.mojo -I src -I "$GG" -I "$JS" -I "$LG" -o .work/$t 2> .work/$t.log || { grep -A5 -m3 error .work/$t.log | cut -c1-240; echo "FAIL build: $t"; exit 1; }; done
export NX_RLOG=off; E=.work/lie-eval-synth; rm -rf $E; mkdir -p $E
run() { python3 -I tools/lie-eval-synth.py $E/$1 $1; mkdir -p $E/$1/probes
  .work/lie_fit $E/$1/main $E/$1/main/labels.tsv $E/$1/probes > $E/$1/fit.log 2>&1 || true; [ -s $E/$1/probes/fit.json ] || { echo "FAIL fit: $1: $(tail -1 $E/$1/fit.log)"; exit 1; }
  .work/lie_eval $E/$1/main $E/$1/main/labels.tsv $E/$1/probes $E/$1/sets $E/$1/sets/all.tsv $E/$1/sets/judge.out $E/$1/sets/tokids.txt $E/$1/unemb15.f32 $E/$1/unemb19.f32 $E/$1/eval.json > $E/$1/eval.log 2>&1 && echo 0 || echo $?; }
[ "$(run ok)" = 0 ] || { echo "FAIL eval: the good synthetic sets missed a bar: $(tail -1 $E/ok/eval.log)"; exit 1; }
[ "$(jq '.S4.auroc_ensemble > 0.9 and .S0.auroc_output > 0.6 and .S0.auroc_unembedding_direction < .S0.auroc_ensemble' $E/ok/eval.json)" = true ] || { echo "FAIL eval: AUROCs of the good run are off: $(jq -c '.S4, .S0' $E/ok/eval.json)"; exit 1; }
[ "$(run surface)" = 2 ] && grep -q "inverted False" $E/surface/eval.log && grep -q "S0 True families True" $E/surface/eval.log || { echo "FAIL mutant: the surface reader did not fail the inverted bar alone: $(tail -1 $E/surface/eval.log)"; exit 1; }
[ "$(run noprobe)" = 2 ] && grep -q "families False" $E/noprobe/eval.log || { echo "FAIL mutant: a set without signal still passed the family bars: $(tail -1 $E/noprobe/eval.log)"; exit 1; }
echo "PASS lie-eval: truth-bearing sets pass all bars, a surface-form reader fails the inverted bar alone, signal-free sets fail the family bars"
