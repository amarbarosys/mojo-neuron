#!/usr/bin/env bash
# Gate for the steer lens: steer.bin (mean over 4 contexts of the next-token log-prob change at +dose) against
# llama.cpp's full-vocabulary answer for three sampled neurons, plus a neighbour-neuron mutant.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; SPARK=${SPARK:-$BARO/.work/steer/spark-q05}; PACK=${PACK:-$BARO/.work/abl/q/pack}
GG=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}; PY=${PY:-.work/venv-gne/bin/python}
LAYER=${LAYER:-12}
S=$PWD/.work/steer/${TAG:-q05}; O=$PWD/.work/slens-check-${TAG:-q05}; rm -rf "$O"; mkdir -p "$O"
[ -f "$S/rho.f32" ] || { echo "FAIL setup: run scripts/steer-check.sh first (rho.f32)"; exit 1; }
printf 'What is the capital of France?\nTell me about dogs.\nThe weather today is\nWrite a poem about the sea.\n' > "$O/p4.txt"
scripts/build-sweep.sh
.work/sweep "$GG" "$O" mode=slens layer=$LAYER neurons=0-199 doses=${DOSE:-0.5} prompts="$O/p4.txt" ctrl=16 k=8 rho="$S/rho.f32" spark="$SPARK" pack="$PACK" baro="$BARO"
grep -q '"null_kl":{' "$O/$LAYER.steer.json" || { echo "FAIL slens: no null_kl in steer.json"; exit 1; }
$PY tools/slens-verify.py "$O/$LAYER.steer.bin" "$GG" "$S/rho.f32" "$PWD/.work/cvec" "$O/p4.txt" 5 77 150
echo PASS slens-check
