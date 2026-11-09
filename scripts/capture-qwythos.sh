#!/usr/bin/env bash
# Capture L<layer> x|y of Qwythos-9B for the transcoder: request lines (tcreq) into one resident engine with BARO_CAPTURE (docs/format-capture.md).
# usage: capture-qwythos.sh REQ.jsonl OUT_DIR [LAYER=16]   (OUT_DIR must exist and be empty; run in the background, ~95 tok/s)
set -euo pipefail
REQ=$(realpath "$1"); OUT=$(realpath "$2"); L=${3:-16}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
[ -x "$ENGINE" ] || { echo "FAIL setup: no engine at $ENGINE"; exit 1; }
[ -z "$(ls -A "$OUT")" ] || { echo "FAIL setup: $OUT is not empty (capture appends)"; exit 1; }
S=$OUT.run.sh
cat > "$S" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SPEC=0 BARO_SERVE=1 BARO_TMAX=1088 BARO_CAPTURE='$OUT' BARO_CAPTURE_LAYERS='$L' BARO_CAPTURE_FIELDS=xy BARO_CAPTURE_MODEL=qwythos-9b
exec '$ENGINE' < '$REQ' > '$OUT.engine.log' 2>&1
EOS
chmod +x "$S"
gpu-wait run --timeout ${TIMEOUT:-3000} -- "$S"
[ -s "$OUT/tokens.i32" ] || { echo "FAIL capture: $(grep -iE 'error|abort' "$OUT.engine.log" | tail -1 | cut -c1-200)"; exit 1; }
echo "PASS capture: $(( $(stat -c %s "$OUT/tokens.i32") / 4 )) tokens in $OUT"
