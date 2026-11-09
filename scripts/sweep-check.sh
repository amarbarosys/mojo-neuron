#!/usr/bin/env bash
# Gate for step 4: a resident sweep returns the same tokens as one-shot runs; reports startup amortisation.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; SPARK=${SPARK:-$BARO/.work/steer/spark-q05}; PACK=${PACK:-$BARO/.work/abl/q/pack}
GG=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}; PY=${PY:-.work/venv-gne/bin/python}
S=$PWD/.work/steer/${TAG:-q05}; O=$PWD/.work/sweep-check; rm -rf "$O"; mkdir -p "$O"
[ -f "$S/rho.f32" ] || { echo "FAIL setup: run scripts/steer-check.sh first (rho.f32)"; exit 1; }
printf 'What is the capital of France?\nTell me about dogs.\n' > "$O/prompts.txt"
scripts/build-sweep.sh
.work/sweep "$GG" "$O" layer=12 neurons=100,101 doses=0.2,0.5 prompts="$O/prompts.txt" ctrl=2 n=16 rho="$S/rho.f32" spark="$SPARK" pack="$PACK" baro="$BARO"
$PY tools/sweep-verify.py "$O" 12 "$GG" "$S/rho.f32" "$PWD/.work/cvec" "$BARO" "$SPARK" "$PACK"
echo PASS sweep-check
