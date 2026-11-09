#!/usr/bin/env bash
# Runs a local llama-server (pair generator) in the foreground; start it under gpu-wait:
#   gpu-wait run --shared --vram 7 --timeout 2400 --label probe-gen -- scripts/probe-server.sh
set -euo pipefail
B=${LLAMA_BIN:-$HOME/Models/llama.cpp/build/bin}; M=${GEN_MODEL:-$HOME/Models/qwen2.5-7b-instruct-import/qwen2.5-7b-instruct-q4_K_M-BARO-3c6c25c.gguf}
[ -f "$M" ] || { echo "FAIL model: $M missing"; exit 1; }
export LD_LIBRARY_PATH=$B PATH="$PATH" HOME="$HOME"
exec "$B/llama-server" -m "$M" -ngl 99 -c $((8 * 2048)) -np 8 --port "${PORT:-8088}" --host 127.0.0.1 --log-disable
