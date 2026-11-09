#!/usr/bin/env bash
# Gate (boss review item 3): steering on engine.mojo must not cost the megakernel.
#   1. a steered run still runs the megakernel ("mega barrier gen" > 0 in its log) and its decode speed is >= 0.95 x the
#      unsteered default (median of 4 interleaved pairs, 128 tokens);
#   2. Gate A (residual tap, docs/prereg-engine-steer.md) passes with the megakernel on, mutants included.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; TAG=${TAG:-qwythos9}
ENGINE=${ENGINE:-$BARO/.work/steer/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
S=$PWD/.work/steer/$TAG; PY=${PY:-.work/venv-gne/bin/python}
[ -f "$S/v.f32" ] || RECS="24:100:150" MEGA=1 scripts/engine-steer-check.sh >/dev/null 2>&1 || true
[ -f "$S/v.f32" ] || { echo "FAIL setup: no steering vector (run engine-steer-check.sh first)"; exit 1; }
run() { # run NAME [env...] -> tok/s_gen; log in $S/NAME.log
  local n=$1; shift
  gpu-wait run --timeout 300 -- env BARO_PROMPT="$S/prompt.ids" BARO_PACK="$PACK" BARO_GEN=128 BARO_SPEC=0 BARO_MEGA=1 ${EXTRA_ENV:-} "$@" "$ENGINE" > "$S/$n.log" 2>&1 || true
  grep -m1 "tok/s_gen" "$S/$n.log" | awk '{print $2}'
}
a=(); b=()
for i in 1 2 3 4; do a+=($(run u)); b+=($(run s BARO_STEER="$S/v.f32")); done
grep -q "mega barrier gen: [1-9]" "$S/s.log" || { echo "FAIL mega: the steered run did not run the megakernel ($(grep -m1 'mega barrier gen' "$S/s.log"))"; exit 1; }
$PY - "${a[*]}" "${b[*]}" <<'PYEOF' || { echo "FAIL speed: steered megakernel decode below 0.95 x unsteered"; exit 1; }
import sys, statistics as st
u = [float(x) for x in sys.argv[1].split()]; s = [float(x) for x in sys.argv[2].split()]
mu, ms = st.median(u), st.median(s)
print("unsteered mega %.1f tok/s, steered mega %.1f tok/s, ratio %.3f" % (mu, ms, ms / mu))
sys.exit(0 if ms / mu >= 0.95 else 1)
PYEOF
out=$(RECS="${RECS:-8:100:30,20:50:120 24:100:150}" MEGA=1 scripts/engine-steer-check.sh 2>&1) || { echo "$out" | grep -E "gate|FAIL" | tail -4; echo "FAIL tap: Gate A or B failed with the megakernel on"; exit 1; }
echo "$out" | grep -E "gate A|gate B v2"
echo "PASS engine-mega-check"
