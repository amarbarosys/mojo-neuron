#!/usr/bin/env bash
# Gate A2 (docs/prereg-a2.md): ablation scan of 24 neurons of Qwythos-9B layer 16 in one resident engine on the 64-prompt eval, then
# (A2b: ctl=0.99 floor pass, 16 neurons, 4 singles, see docs/prereg-a2b.md) (1) singles re-run one process each match the batch, (2) identity gain scores 0, (3) a wrong-neuron comparison differs (mutant),
# (4) the scan wrote a normal rlog line. One gpu-wait job runs everything GPU side.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; G=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}; SP=${SUBPROC_SRC:-$HOME/Projects/mojo/mojo-subprocess/src}
L=${LAYER:-16}; NEU=${NEURONS:-3,1031,2059,3087,4115,5143,6171,7199,8227,9255,10283,11311,1545,4629,7713,10797}
SINGLE=${SINGLE:-3,1545,3087,4629}
O=$PWD/.work/scan-check; rm -rf "$O"; mkdir -p "$O"
[ -x "$ENGINE" ] || { echo "FAIL setup: no engine at $ENGINE"; exit 1; }
timeout 300 "$M" build src/causal/scan.mojo -I src -I "$GG" -I "$BARO" -I "$BARO/serve" -I "$SP" -I "$LG" -o "$O/scan" 2> "$O/build.log" || { grep -A5 -m3 error "$O/build.log" | cut -c1-240; echo "FAIL build: scan"; exit 1; }
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
export PATH='$PATH' HOME='$HOME' NX_RLOG='$O/rl'
cd '$PWD'
r() { # r DIR NEURONS K
  mkdir -p "\$1"; '$O/scan' '$G' "\$1" layer=$L neurons=\$2 k=\${3:-0} \${4:-} prompts=tests/data/eval64.txt baro='$BARO' spark='$ENGINE' pack='$PACK' runner=none > "\$1/out.txt" 2>&1
}
r '$O/batch' $NEU 0 ctl=0.99
r '$O/ident' 5,6 1
r '$O/g999' 5 0.999
for n in \$(echo $SINGLE | tr , ' '); do r "$O/single-\$n" \$n 0 ctl=0.99; done
EOS
chmod +x "$O/run.sh"
gpu-wait run --timeout 2400 -- "$O/run.sh" || true
[ -s "$O/batch/L$L.scan.tsv" ] || { echo "FAIL batch: $(tail -2 "$O/batch/out.txt" | cut -c1-200)"; exit 1; }
python3 -I tools/scan-verify.py "$O" "$SINGLE" "$L"
n=$(grep -c '"kind":"normal"' "$O"/rl/2*.jsonl); [ "$n" -ge 5 ] || { echo "FAIL rlog: $n normal lines (want >= 5)"; exit 1; }
echo "PASS scan-check: 16-neuron batch with floor pass == 4 single-process re-runs, identity gain scores 0, wrong-neuron mutant rejected, $n rlog lines"
