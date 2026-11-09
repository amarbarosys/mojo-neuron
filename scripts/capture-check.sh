#!/usr/bin/env bash
# Gate A1 (docs/prereg-a1.md): engine capture vs the engine's BARO_DUMP at the last prompt position, plus a layer-shift mutant.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; LAYERS=${LAYERS:-0,16,31}
O=$PWD/.work/capture-check; rm -rf "$O"; mkdir -p "$O/cap"
[ -x "$ENGINE" ] || { echo "FAIL setup: no engine at $ENGINE"; exit 1; }
NP=$(wc -w < "$PACK/prompt-tokens.txt")
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SPEC=0 BARO_DUMP='$O/dump.bin' BARO_CAPTURE='$O/cap' BARO_CAPTURE_LAYERS='$LAYERS'
exec '$ENGINE' > '$O/engine.log' 2>&1
EOS
chmod +x "$O/run.sh"
gpu-wait run --timeout 300 -- "$O/run.sh" || true
[ -s "$O/cap/tokens.i32" ] && [ -s "$O/dump.bin" ] || { echo "FAIL run: $(grep -iE 'error|abort' "$O/engine.log" | tail -1 | cut -c1-200)"; exit 1; }
python3 -I tools/capture-oracle.py "$O/cap" "$O/dump.bin" "$NP" ok
python3 -I tools/capture-oracle.py "$O/cap" "$O/dump.bin" "$NP" mutant
echo "PASS capture-check: layers $LAYERS bit-equal to BARO_DUMP at the last prompt position, y/x sane, layer-shift mutant rejected"
