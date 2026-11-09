#!/usr/bin/env bash
# Gate F30b section B (docs/prereg-f30b.md): steer-first labels on qwen2.5-0.5B (bf16) through the spark `slens` op, one gpu-wait job.
# Deviations from the prereg text, named: no rho file for this model, so dose is the absolute unit-vector length (DOSE, default 8); layer is LAYER
# (default 22); the null is the steer-up list of another unit (ctrl=N stores KL only, no token lists).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG_SRC=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; SPARK=${SPARK:-$BARO/.work/steer/spark-q05}; PACK=${PACK:-$BARO/.work/abl/q/pack}
GGUF=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}; MOD=qwen2.5-0.5b-instruct-bf16; LAYER=${LAYER:-22}
timeout 280 "$M" build src/nx.mojo -I src -I "$GG_SRC" -I "$J" -I "$LG" -o .work/nx_wiki 2> .work/nx_wiki.log || { grep -A5 -m3 error .work/nx_wiki.log | cut -c1-240; echo "FAIL build: nx_wiki"; exit 1; }
[ -x .work/lens ] || timeout 280 "$M" build src/lens.mojo -I src -I "$GG_SRC" -I "$J" -o .work/lens >/dev/null 2>&1 || { echo "FAIL build: lens"; exit 1; }
scripts/build-sweep.sh >/dev/null
D=$PWD/.work/wiki-steer; rm -rf "$D"; mkdir -p "$D"; export MNEURON_DIR=$D/neurons NX_RLOG=$D/rlog
.work/lens "$GGUF" "$LAYER" 8 --out "$MNEURON_DIR" >/dev/null
printf 'What is the capital of France?\nTell me about dogs.\nThe weather today is\nWrite a poem about the sea.\n' > "$D/p4.txt"
gpu-wait run --timeout 600 --vram 3 --label wiki-steer -- bash -c "export PATH='$PATH' HOME='$HOME'; .work/sweep '$GGUF' '$MNEURON_DIR/$MOD' mode=slens layer=$LAYER neurons=0-199 doses=${DOSE:-8} prompts='$D/p4.txt' ctrl=16 k=8 spark='$SPARK' pack='$PACK' baro='$BARO'" > "$D/sweep.log" 2>&1 || { tail -3 "$D/sweep.log"; echo "FAIL wiki-steer: slens job"; exit 1; }
[ -f "$MNEURON_DIR/$MOD/$LAYER.steer.bin" ] || { tail -3 "$D/sweep.log"; echo "FAIL wiki-steer: no steer.bin"; exit 1; }
.work/nx_wiki wiki "{\"model\":\"$MOD\",\"layer\":$LAYER,\"steer\":1}" > "$D/summary.json"
python3 -I tools/wiki-steer-check.py "$PWD/.work/nx_wiki" "$MNEURON_DIR/$MOD" "$MOD" "$LAYER" "$D/summary.json" | tee "$D/receipt.txt"
echo "PASS wiki-steer: planted false link refused, shuffled null within the bar; see $D/receipt.txt"
