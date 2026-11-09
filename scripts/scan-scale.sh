#!/usr/bin/env bash
# Falsifier for the scan's flat KL floor (A2 run 1: 24 neurons all scored 0.8e-5..1.2e-5): if gain is applied where it should be, damage grows
# with the size of the change: KL ~ (k - 1)^2 for small effects, so k=10 must score far above k=0 and k=-1 about 4x k=0 on the same neurons.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; G=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf}
S=${SCAN:-$PWD/.work/scan-check/scan}; O=$PWD/.work/scan-scale; rm -rf "$O"; mkdir -p "$O"
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
export PATH='$PATH' HOME='$HOME' NX_RLOG='$O/rl'
cd '$PWD'
for k in 0 -1 10 100; do mkdir -p '$O'/k\$k; '$S' '$G' '$O'/k\$k layer=16 neurons=3,3087 k=\$k prompts=tests/data/eval64.txt baro='$BARO' spark='$ENGINE' pack='$PACK' runner=none > '$O'/k\$k/out.txt 2>&1; done
EOS
chmod +x "$O/run.sh"; gpu-wait run --timeout 1200 -- "$O/run.sh" || true
python3 -I - "$O" <<'PYEOF'
import sys
O = sys.argv[1]
def rd(k): return {int(a): float(b) for a, b, c in (l.split("\t") for l in open(f"{O}/k{k}/L16.scan.tsv").read().split("\n")[1:] if l)}
r = {k: rd(k) for k in ("0", "-1", "10", "100")}
for n in (3, 3087): print(n, {k: f"{r[k][n]:.3e}" for k in r}, "ratios to k=0:", {k: round(r[k][n] / r["0"][n], 1) for k in r})
ok = all(r["10"][n] > 20 * r["0"][n] and r["100"][n] > r["10"][n] for n in (3, 3087))
print("PASS scan-scale: damage grows with the gain" if ok else "FAIL scan-scale: damage does not grow with the gain (k=10 not > 20x k=0, or k=100 <= k=10)")
sys.exit(0 if ok else 1)
PYEOF
