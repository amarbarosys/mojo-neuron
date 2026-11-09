#!/usr/bin/env bash
# usage: llama-cvec-load.sh MODEL.gguf CVEC.gguf LOG -> exit 0 only when llama.cpp loads MODEL with the control vector and generates a token
# (run inside `gpu-wait run --room cpu`; llama-completion is the non-interactive binary, llama-cli waits for chat input)
set -uo pipefail
timeout 60 "${LLAMA_BIN:-$HOME/Models/llama.cpp/build/bin}/llama-completion" -m "$1" --control-vector "$2" -n 1 -p hi -ngl 0 --device none --no-warmup </dev/null > "$3" 2>&1 || exit 1
! grep -q "common_contr" "$3"
