#!/usr/bin/env bash
# F24 step 5: steering run. prep (lie_steer) -> engine serve with BARO_STEER_SERVE=1 over prep/req.jsonl (150 test statements x 6 conditions,
# capture at Ls and Lm) -> lie_steer eval (bars 3, 4). usage: probe-steer.sh CAP LABELS PROBES OUT
set -euo pipefail
cd "$(dirname "$0")/.."
CAPD=${1:?usage: probe-steer.sh CAP LABELS PROBES OUT}; LABELS=${2:?}; PROBES=${3:?}; OUT=${4:?}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
for f in "$ENGINE" .work/lie_steer "$CAPD/meta.json" "$LABELS" "$PROBES/fit.json"; do [ -e "$f" ] || { echo "FAIL setup: $f missing"; exit 1; }; done
mkdir -p "$OUT"; SCAP=$OUT/cap; rm -rf "$SCAP"; mkdir -p "$SCAP"
.work/lie_steer prep "$CAPD" "$LABELS" "$PROBES" "$OUT" | tail -1
LS=$(jq -r .Ls "$OUT/prep.json"); LM=$(jq -r .Lm "$OUT/prep.json")
cat > "$OUT/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SERVE=1 BARO_SPEC=0 BARO_STEER_SERVE=1 BARO_CAPTURE='$SCAP' BARO_CAPTURE_LAYERS='$LS,$LM' BARO_CAPTURE_FIELDS=r BARO_CAPTURE_MODEL=qwythos-9b
exec '$ENGINE' < '$OUT/req.jsonl' > '$OUT/engine.out' 2> '$OUT/engine.err'
EOS
chmod +x "$OUT/run.sh"
gpu-wait run --label probe-steer --timeout "${TIMEOUT:-900}" -- "$OUT/run.sh" || true
d=$(grep -c '"done":true' "$OUT/engine.out" || true); e=$(grep -c '"error"' "$OUT/engine.out" || true)
[ "$d" = 900 ] && [ "$e" = 0 ] || { echo "FAIL run: $d of 900 requests done, $e errors: $(grep -m1 '"error"' "$OUT/engine.out" | cut -c1-200) $(grep -iE 'error|abort' "$OUT/engine.err" | tail -1 | cut -c1-160)"; exit 1; }
.work/lie_steer eval "$CAPD" "$LABELS" "$PROBES" "$SCAP" "$OUT"
