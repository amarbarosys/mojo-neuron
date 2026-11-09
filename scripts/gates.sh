#!/usr/bin/env bash
# Every gate of the engine lane, one line each, then the receipt .work/gates.txt (HEAD sha + outputs).
set -uo pipefail
cd "$(dirname "$0")/.."
out=.work/gates.txt; { echo "head $(git rev-parse --short HEAD) baro $(git -C ${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate} rev-parse --short HEAD)"; } > $out
run() { local n=$1; shift; local r; r=$("$@" 2>&1 | grep -E "^(PASS|FAIL|ok )" | tail -${TAILN:-1}); echo "$n: $r" | tee -a $out; }
run lens                scripts/check.sh
TAILN=6 run steer-matrix scripts/steer-matrix.sh ${MATRIX_TAGS:-}
run cvec-perf          scripts/cvec-perf.sh
run slens-mutants       scripts/slens-mutants.sh
run sweep-complete      scripts/sweep-complete-check.sh
TAILN=1 run engine-mega      scripts/engine-mega-check.sh
run engine-sweep     scripts/engine-sweep-check.sh
run engine-slens      scripts/engine-slens-check.sh
run engine-slens-sweep scripts/engine-slens-sweep-check.sh
run sweep               scripts/sweep-check.sh
run gain                scripts/gain-check.sh
run reverse             scripts/reverse-check.sh
TAILN=1 run engine-steer scripts/engine-steer-check.sh
A=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}
[ "${GATES_27B:-0}" = 1 ] && [ -x $A/.work/steer/engine-27b ] && TAILN=1 TAG=orca27 ENGINE=$A/.work/steer/engine-27b PACK=$A/.work/steer/orca/pack GGUF=$HOME/Models/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored.gguf LAYERS=64 H=5120 EXTRA_ENV="BARO_EMBD_HOST=1 BARO_TMAX=1088" RECS='16:100:60,40:50:200 8:100:100,56:50:300' run engine-steer-27b scripts/engine-steer-check.sh
