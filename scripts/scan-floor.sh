#!/usr/bin/env bash
# Diagnostic for the scan's flat KL floor (A2 run 1, scan-scale): KL of neuron 3 at gains 0.999 ... -1. A pure linear effect gives KL ~ (k-1)^2 and goes to 0 as
# k -> 1; a floor that stays near 8e-6 at k = 0.999 means any non-identity gain moves something that is not the neuron.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; G=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf}
S=${SCAN:-$PWD/.work/scan-check/scan}; O=$PWD/.work/scan-floor; rm -rf "$O"; mkdir -p "$O"
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
export PATH='$PATH' HOME='$HOME' NX_RLOG='$O/rl'
cd '$PWD'
for k in 0.999 0.99 0.9 0.5 0 -1; do mkdir -p '$O'/k\$k; '$S' '$G' '$O'/k\$k layer=16 neurons=3,5 k=\$k prompts=tests/data/eval64.txt baro='$BARO' spark='$ENGINE' pack='$PACK' runner=none > '$O'/k\$k/out.txt 2>&1; done
EOS
chmod +x "$O/run.sh"; gpu-wait run --timeout 1200 -- "$O/run.sh" || true
for k in 0.999 0.99 0.9 0.5 0 -1; do echo "k=$k: $(tail -n +2 "$O/k$k/L16.scan.tsv" | tr '\t\n' '  ' | cut -c1-120)"; done
