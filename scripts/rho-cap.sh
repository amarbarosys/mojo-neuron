#!/usr/bin/env bash
# rho-cap [LAYER] -> .work/dreams/rho<LAYER>.txt: unsteered Qwythos capture (pack prompt), median residual norm at LAYER.
set -euo pipefail
cd "$(dirname "$0")/.."; L=${1:-16}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
O=$PWD/.work/dreams/rho; rm -rf "$O"; mkdir -p "$O/cap"
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SPEC=0 BARO_CAPTURE='$O/cap' BARO_CAPTURE_LAYERS='$L' BARO_CAPTURE_FIELDS=r
exec '$BARO/.work/engine-9b' > '$O/engine.log' 2>&1
EOS
chmod +x "$O/run.sh"
gpu-wait run --timeout 300 -- "$O/run.sh" || true
[ -s "$O/cap/L$L.f16" ] || { echo "FAIL rho-cap: no capture ($(grep -iE 'error|abort' "$O/engine.log" | tail -1 | cut -c1-160))"; exit 1; }
python3 -I tools/rho-cap.py "$O/cap" "$L" | tee .work/dreams/rho$L.txt
