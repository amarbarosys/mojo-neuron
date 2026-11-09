#!/usr/bin/env bash
# Gate B1: transcoder step parity vs f64 CPU reference; three planted defects (-D MUT=1..3) must each FAIL it.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-baro/.venv/bin/mojo}; OPS=${OPS_SRC:-$HOME/Projects/mojo/mojo-baro/miniagi/train}
build() { timeout 250 "$M" build src/features/tc_check.mojo -I src/features -I "$OPS" -I "${KERN_SRC:-$HOME/Projects/mojo/mojo-baro/kernels}" "${@:2}" -o ".work/tc_check$1" 2> ".work/tc_check$1.log" || { grep -A5 -m3 error ".work/tc_check$1.log" | cut -c1-240; echo "FAIL build: tc_check$1"; exit 1; }; }
build "" & build _m1 -D MUT=1 & build _m2 -D MUT=2 & build _m3 -D MUT=3 & wait
run() { gpu-wait run --timeout 90 --shared --vram 1 -- env PATH="$PATH" HOME="$HOME" ".work/tc_check$1" 2>&1; }
run "" | tail -1 | grep -q '^PASS' || { echo "FAIL tc parity: clean build"; exit 1; }
for m in 1 2 3; do
  if run _m$m | grep -q '^PASS'; then echo "FAIL mutant MUT=$m passed the parity gate"; exit 1; fi
done
echo "PASS tc-check: parity + mutants K+1, no renorm, db_dec sign all fail"
