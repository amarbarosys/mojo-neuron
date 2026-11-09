#!/usr/bin/env bash
# Resume after the 2026-11-09 21:41 box reset (GPU queue emptied): re-queue only what has no receipt, one stage at a time, no builds. Wall seconds per stage go to the log.
# Order (boss): guard tests first (they gate the neuron2b merge and the fix-engine lane), then T1c eval-real / eval-ctl / verdict, then the G4 chain.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/Projects/mojo/mojo-baro/.venv/bin/python3; B=$HOME/Projects/mojo/mojo-baro-lanes/neuron2b; W=.work/impl-m3; H="$PY -I tools/implants-harness.py $W"
stage() { local n=$1; shift; local t0=$SECONDS; echo "== $n start"; "$@"; local rc=$?; echo "== $n rc=$rc wall $((SECONDS - t0)) s (queue wait included)"; return 0; }
stage guards-old bash -c "[ -s .work/guards-old.out ] && grep -qE '^(MISSING|ok)' .work/guards-old.out || scripts/guards-check.sh $B/.work/engine-9b-pfcap old > .work/guards-old.out 2>&1; cat .work/guards-old.out"
stage guards-fix bash -c "scripts/guards-check.sh $B/.work/engine-9b-fix fix > .work/guards-fix.out 2>&1; cat .work/guards-fix.out"
if grep -q "^PASS guards-check" .work/guards-fix.out; then
  herd tell a pane "[a pane] LANDED P1-P4 guards, neuron2b $(git -C $B rev-parse --short HEAD) (P1 561b25a splice rows vs prompt+n, P2 a5490af implants_* without BARO_IMPLANTS / lo > k, P3 48aad01 slens_foil needs slens_tgt, P4 0d194ed capture chunk cap + comptime assert). guards-check: old binary a1a608a FAILED these cases (see .work/guards-old.out), fixed binary engine-9b-fix PASS (every bad request refused with a named error, good ones served). P4 has no runtime test (pf_chunk = min(env, CP), CP = PF_CAP_ROWS: unreachable); the build is its check. Built before the reset, one build." || true
fi
stage eval-real $H eval-real
stage eval-ctl  $H eval-ctl
stage verdict   $H verdict
stage g4 env W=$W scripts/g4-run.sh
