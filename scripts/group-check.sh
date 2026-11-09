#!/usr/bin/env bash
# A2c-group (docs/prereg-a2c-group.md): capture -> prerank -> floor + group scoring in one resident engine process -> verdict. Two GPU jobs (capture, scoring).
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; CAPENG=${CAPENG:-$BARO/.work/engine-9b-imp}; ENGINE=${ENGINE:-$BARO/.work/engine-9b-foil}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; G=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}; SP=${SUBPROC_SRC:-$HOME/Projects/mojo/mojo-subprocess/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}
L=${LAYER:-16}; O=$PWD/.work/group-check; mkdir -p "$O/cap"; rm -f "$O"/L$L.group.tsv
[ -x "$ENGINE" ] && [ -x "$CAPENG" ] || { echo "FAIL setup: engine $ENGINE or $CAPENG missing"; exit 1; }
for t in group prerank; do timeout 300 "$M" build src/causal/$t.mojo -I src -I "$GG" -I "$JS" -I "$LG" -I "$SP" -I "$BARO" -I "$BARO/serve" -o "$O/$t" 2> "$O/$t.log" || { grep -A5 -m3 error "$O/$t.log" | cut -c1-240; echo "FAIL build: $t"; exit 1; }; done
if [ ! -s "$O/cap/L$L.f16" ]; then
  cat > "$O/cap.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' NX_RLOG='$O/rl'
cd '$PWD'; rm -rf '$O/cap'; mkdir -p '$O/cap'
'$O/group' '$G' '$O' layer=$L facts=tests/data/capitals.tsv capture='$O/cap' baro='$BARO' spark='$CAPENG' pack='$PACK' runner=none > '$O/cap.log' 2>&1
EOS
  chmod +x "$O/cap.sh"; gpu-wait run --timeout 600 -- "$O/cap.sh" || true
fi
[ -s "$O/cap/L$L.f16" ] || { echo "FAIL capture: $(tail -n 2 "$O/cap.log" | cut -c1-200)"; exit 1; }
NX_RLOG="$O/rl" "$O/prerank" "$G" "$O/cap" $L tests/data/capitals.tsv "$O/pre"
cp "$O/pre.de.f32" "$O/pre.de.f32.keep" 2>/dev/null || true
python3 -I tools/group-verify.py make "$O/pre.rank.tsv" "$O/groups.tsv"
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' NX_RLOG='$O/rl'
cd '$PWD'
'$O/group' '$G' '$O' layer=$L facts=tests/data/capitals.tsv groups='$O/groups.tsv' baro='$BARO' spark='$ENGINE' pack='$PACK' runner=none > '$O/run.log' 2>&1
EOS
chmod +x "$O/run.sh"; gpu-wait run --timeout 1500 -- "$O/run.sh" || true
[ -s "$O/L$L.group.tsv" ] || { echo "FAIL group run: $(tail -n 2 "$O/run.log" | cut -c1-200)"; exit 1; }
python3 -I tools/group-verify.py verify "$O"
