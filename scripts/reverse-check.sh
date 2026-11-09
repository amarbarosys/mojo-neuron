#!/usr/bin/env bash
# Gate for reverse lookup: persona contrast, cosine rank of all neurons of all layers, against an f64 oracle
# (own bf16 decode via the python gguf reader); the swapped contrast (B - A) must NOT match.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; SPARK=${SPARK:-$BARO/.work/steer/spark-q05}; PACK=${PACK:-$BARO/.work/abl/q/pack}
GG=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}; PY=${PY:-.work/venv-gne/bin/python}
O=$PWD/.work/reverse-check; rm -rf "$O"; mkdir -p "$O/swap"
printf 'What is the capital of France?\nTell me about dogs.\nHow do I bake bread?\nWhat is two plus two?\nWrite a short greeting.\nExplain gravity.\n' > "$O/prompts.txt"
scripts/build-sweep.sh
run() { .work/reverse "$GG" "$1" prompts="$O/prompts.txt" "sys_a=You are a pirate. Speak like a pirate." "sys_b=You are a helpful assistant." top=10 spark="$SPARK" pack="$PACK" baro="$BARO" "${@:2}"; }
run "$O"
$PY tools/reverse-oracle.py "$O" "$GG"
run "$O/swap" swap=1 >/dev/null
if $PY tools/reverse-oracle.py "$O/swap" "$GG" >/dev/null 2>&1; then echo "FAIL mutant: the swapped contrast matches the oracle"; exit 1; fi
echo "ok   mutant (B - A) rejected"
echo PASS reverse-check
