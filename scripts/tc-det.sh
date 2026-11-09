#!/usr/bin/env bash
# Gate B1 determinism: two same-seed tc_train runs on a capture write byte-identical tc.bin. usage: tc-det.sh CAPDIR LAYER (needs .work/tc_train, see docs/reports)
set -euo pipefail
cd "$(dirname "$0")/.."
CAP=${1:?capture dir}; L=${2:?layer}
for r in 1 2; do rm -rf .work/tc-det$r
  gpu-wait run --timeout 300 --shared --vram 4 -- env PATH="$PATH" HOME="$HOME" NX_RLOG=off .work/tc_train "$CAP" "$L" .work/tc-det$r 1024 32 512 100 5e-4 --seed 5 >/dev/null 2>&1 || { echo "FAIL tc-det: run $r"; exit 1; }; done
cmp -s .work/tc-det1/tc.bin .work/tc-det2/tc.bin || { echo "FAIL tc-det: weights differ between same-seed runs"; exit 1; }
echo "PASS tc-det: same seed, byte-identical tc.bin ($(sha256sum .work/tc-det1/tc.bin | cut -c1-16))"
