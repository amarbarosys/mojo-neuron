#!/usr/bin/env bash
# Gate (boss review item 6): a resident sweep on engine.mojo (qwen35 family) with per-request steering returns the same tokens as
# independent one-shot BARO_STEER runs, and two neurons differ. Needs ENGINE built with BARO_STEER_SERVE support.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; TAG=${TAG:-qwythos9}
ENGINE=${ENGINE:-$BARO/.work/steer/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
LAYERS=${LAYERS:-32}; LAYER=${LAYER:-16}; PY=${PY:-.work/venv-gne/bin/python}
S=$PWD/.work/steer/$TAG; O=$PWD/.work/engine-sweep-$TAG; rm -rf "$O"; mkdir -p "$O"
$PY -c "import numpy as np;np.ones($LAYERS,dtype=np.float32).tofile('$O/rho1.f32')"   # raw unit multiples: the driver and cvec then agree on the vector
printf 'What is the capital of France?\n' > "$O/prompts.txt"
scripts/build-sweep.sh
.work/sweep "$GG" "$O" layer=$LAYER neurons=100,101 doses=30,60 prompts="$O/prompts.txt" n=8 rho="$O/rho1.f32" spark="$ENGINE" pack="$PACK" baro="$BARO" env="BARO_MEGA=1 ${EXTRA_ENV:-}" >/dev/null
ONESHOT_ENV="BARO_MEGA=1 ${EXTRA_ENV:-}" $PY tools/sweep-verify.py "$O" $LAYER "$GG" "$O/rho1.f32" "$PWD/.work/cvec" "$BARO" "$ENGINE" "$PACK"
echo "PASS engine-sweep-check $TAG"
