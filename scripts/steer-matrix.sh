#!/usr/bin/env bash
# Steer gate over every spark architecture we have a pack for: builds spark per profile if needed, then steer-check.
# usage: steer-matrix.sh [TAG...]   (default: all)  -> prints one PASS/FAIL line per model
set -uo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; B=$HOME/Projects/mojo/mojo-baro
declare -A PROF=( [q05]="$BARO/.work/abl/q/prof" [llama1b]="$B/.work/auto-llama/profile" [granite]="$BARO/.work/abl/prof" )
declare -A PACK=( [q05]="$BARO/.work/abl/q/pack" [llama1b]="$B/.work/auto-llama/pack" [granite]="$BARO/.work/abl/pack" )
declare -A GGUF=( [q05]="$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf"
  [llama1b]="$B/.work/auto-llama/llama-3.2-1b-instruct-q4_k_m-q4_K_M.gguf"
  [granite]="$HOME/Models/granite-4.2-3b-bf16/granite-4.2-3b-BF16-BARO-3c6c25c.gguf" )
tags=("$@"); [ ${#tags[@]} -eq 0 ] && tags=(q05 llama1b granite)
for t in "${tags[@]}"; do
  bin=$BARO/.work/steer/spark-$t
  if [ ! -x "$bin" ]; then
    (cd "$BARO" && timeout 400 ./.venv/bin/mojo build serve/spark.mojo -I "${PROF[$t]}" -I . -I kernels -I serve -I tools -o "$bin" > ".work/steer/build-$t.log" 2>&1) \
      || { echo "FAIL $t: spark build ($(grep -m1 'error:' "$BARO/.work/steer/build-$t.log" | cut -c1-150))"; continue; }
  fi
  if out=$(TAG=$t PROF=${PROF[$t]} PACK=${PACK[$t]} GGUF=${GGUF[$t]} scripts/steer-check.sh 2>&1); then echo "$out" | tail -1
  else echo "FAIL $t: $(echo "$out" | grep -E '^FAIL|rel err' | tail -2 | tr '\n' ' ')"; continue; fi
  nl=$(sed -n 's/^comptime N_LAYERS = \([0-9]*\).*/\1/p' "${PROF[$t]}/profile.mojo")
  if out=$(TAG=$t LAYER=$((nl / 2)) SPARK=$bin PACK=${PACK[$t]} GGUF=${GGUF[$t]} scripts/slens-check.sh 2>&1); then echo "$out" | tail -1 | sed "s/\$/ $t/"
  else echo "FAIL slens $t: $(echo "$out" | grep -E '^FAIL|overlap|KL' | tail -3 | tr '\n' ' ')"; fi
done
