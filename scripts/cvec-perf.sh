#!/usr/bin/env bash
# E4 gate: a recipe of 24 units in one layer must cost about one layer load, not 24 (wall time ratio <= 2 against one unit).
set -euo pipefail
cd "$(dirname "$0")/.."
GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}; O=$PWD/.work/cvec-perf; mkdir -p "$O"
recipe=$(for n in $(seq 0 23); do printf '16:%d:1.0,' $n; done); recipe=${recipe%,}
python3 - "$GG" "$O/x" "$recipe" <<'PYEOF' || { echo "FAIL cvec-perf: 24 units in one layer cost more than 2x one unit"; exit 1; }
import subprocess, sys, time
gg, out, recipe = sys.argv[1:4]
def t(r):
    s = time.perf_counter(); subprocess.run([".work/cvec", gg, r, out], check=True, capture_output=True); return time.perf_counter() - s
one, many = min(t("16:0:1.0") for _ in range(2)), min(t(recipe) for _ in range(2))
print("1 unit %.2f s, 24 units %.2f s, ratio %.1f" % (one, many, many / one))
sys.exit(0 if many / one <= 2 else 1)
PYEOF
echo "PASS cvec-perf"
