#!/usr/bin/env bash
# E3 gate: the sweep driver must refuse a damaged spark response (dropped, duplicated, malformed or out-of-range lines)
# instead of writing records from empty requests. Runs a real sweep, then feeds mutated copies of its response via resp=.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; SPARK=${SPARK:-$BARO/.work/steer/spark-q05}; PACK=${PACK:-$BARO/.work/abl/q/pack}
GG=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}
S=$PWD/.work/steer/${TAG:-q05}; O=$PWD/.work/sweep-complete; rm -rf "$O"; mkdir -p "$O"
printf 'What is the capital of France?\nTell me about dogs.\n' > "$O/prompts.txt"
args=(layer=12 neurons=100,101 doses=0.2 prompts="$O/prompts.txt" ctrl=1 n=8 rho="$S/rho.f32" spark="$SPARK" pack="$PACK" baro="$BARO")
.work/sweep "$GG" "$O" "${args[@]}" >/dev/null
R=$O/.resp.jsonl; rm -f "$O/12.sweep.jsonl"
.work/sweep "$GG" "$O" "${args[@]}" resp="$R" >/dev/null || { echo "FAIL setup: the intact response was refused"; exit 1; }
[ -s "$O/12.sweep.jsonl" ] || { echo "FAIL setup: no records from the intact response"; exit 1; }
rc=0
mutant() { # mutant NAME 'sed script on the response'
  sed "$2" "$R" > "$O/mut-$1.jsonl"; rm -f "$O/12.sweep.jsonl"
  if .work/sweep "$GG" "$O" "${args[@]}" resp="$O/mut-$1.jsonl" >/dev/null 2>&1; then echo "FAIL mutant $1: the driver accepted a damaged response"; rc=1
  else echo "ok   mutant $1 rejected"; fi
}
mutant droptok     '0,/"id":3,"tok"/{/"id":3,"tok"/d}'
mutant dropdone    '/"id":4,"done"/d'
mutant duptok      '0,/"id":2,"tok"/{p}'
mutant dupdone     '/"id":2,"done"/p'
mutant garbage     's/"id":5,"tok":[0-9]*/"id":5,"tok"/'
mutant outofrange  '$a {"id":999,"tok":1}'
mutant unknownline '$a {"id":3,"wat":true}'
[ $rc = 0 ] && echo "PASS sweep-complete-check"; exit $rc
