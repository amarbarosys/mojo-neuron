#!/usr/bin/env bash
# dream.sh RUN: F26 model dreams on Qwythos-9B. Engine (gpu-wait) and .work/dream talk over two FIFOs; then score.
# env: MULTS (e.g. 1,2,3,4,6 = F26c sweep), MULT, DISTINCT, LAYER (16) STEPS (8) NFEAT (8) NCTRL (4). Needs .work/dreams/rho$LAYER.txt (scripts/rho-cap.sh) and a lens.bin of the layer.
set -euo pipefail
cd "$(dirname "$0")/.."; RUN=${1:?usage: dream.sh RUN}; L=${LAYER:-16}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
MODEL=${MODEL:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf}
STEM=$(basename "$MODEL" .gguf | tr 'A-Z' 'a-z'); export MNEURON_DIR=${MNEURON_DIR:-$PWD/.work/neurons}
O=$PWD/.work/dreams/$RUN; rm -rf "$O"; mkdir -p "$O"; mkfifo "$O/req" "$O/resp"
[ -s .work/dreams/rho$L.txt ] || { echo "FAIL setup: run scripts/rho-cap.sh $L first"; exit 1; }
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_TMAX=256
exec '$BARO/.work/engine-9b' > '$O/resp' 2> '$O/engine.err' < '$O/req'
EOS
chmod +x "$O/run.sh"
gpu-wait run --timeout 900 -- "$O/run.sh" & EP=$!
.work/dream run "$MODEL" "$O" rho="$(cat .work/dreams/rho$L.txt)" layer="$L" steps="${STEPS:-8}" nfeat="${NFEAT:-8}" nctrl="${NCTRL:-4}" mult="${MULT:-1}" mults="${MULTS:-}" lens="$STEM" < "$O/resp" > "$O/req" || { kill $EP 2>/dev/null; echo "FAIL dream run: see $O/engine.err"; exit 1; }
wait $EP || true
.work/dream ${MULTS:+sweep}score "$O" layer="$L" lens="$STEM" distinct="${DISTINCT:-0}" | tee "$O/score.log"
