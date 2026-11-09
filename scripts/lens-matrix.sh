#!/usr/bin/env bash
# Lens over the model matrix: golden check on small models, then lens.bin for the shared store. One row per model on stdout:
# model|arch|layer|check|bin|seconds   (PASS, FAIL(reason) or UNSUPPORTED(reason)). Usage: lens-matrix.sh [pattern]
set -uo pipefail
cd "$(dirname "$0")/.."
SHARED=${NEURONS:-$HOME/Projects/mojo/mojo-neuron/.work/neurons}
MD=$HOME/Models
# rel-path : arch : layer : check(0 = bin only)
LIST="qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf:qwen2:12:8
vibethinker-1.5b-gguf/vibethinker-1.5b-Q4_K_M.gguf:qwen2:14:4
llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf:llama:8:4
minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf:llama:20:4
granite-4.2-3b-bf16/granite-4.2-3b-BF16-BARO-3c6c25c.gguf:granite:20:4
spark-x2.5-4b/Spark-X2.5-4B-Q4_K_M.gguf:spark2_5:18:4
ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M-stock.gguf:qwen35:16:2
qwen3.8-27b-obliterated-q4_K_M/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf:qwen35:32:0
TriLM-1.5B-TQ2_0.gguf:llama:12:4
lily-cybersecurity-7b-v0.2-q6_k/lily-cybersecurity-7b-v0.2-IQ1_S.gguf:llama:16:0"
cap() { systemd-run --user --scope -q -p MemoryMax=${MEM:-24G} -p MemorySwapMax=0 -- timeout ${T:-900} "$@"; }
echo "$LIST" | while IFS=: read -r f arch l n; do
  case "$f" in *${1:-}*) ;; *) continue;; esac
  [ -f "$MD/$f" ] || { echo "$f|$arch|$l|FAIL(model missing)||0"; continue; }
  s=$(date +%s); chk=skip
  if [ "$n" -gt 0 ]; then
    r=$(cap .work/lens "$MD/$f" "$l" 8 "$n" 2>&1 | tail -1); case "$r" in PASS*) chk=PASS;; *) chk="FAIL(${r:0:80})";; esac
  fi
  r=$(cap .work/lens "$MD/$f" "$l" 8 --out "$SHARED" 2>&1 | tail -1); case "$r" in wrote*) bin=PASS;; *) bin="FAIL(${r:0:80})";; esac
  echo "$f|$arch|$l|$chk|$bin|$(( $(date +%s) - s ))"
done
