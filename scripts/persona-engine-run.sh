#!/usr/bin/env bash
# The GPU part of persona-engine-check.sh, run inside ONE gpu-wait job: three engine launches (BARO_STEER_SERVE=1, one request each with
# per-request steer_file, the way baro-serve will pick a persona): unsteered, persona vector, recipe vector. Dumps row-0.bin (first step logits).
# usage: persona-engine-run.sh OUTDIR IDS_JSON PERSONA_F32 RECIPE_F32
set -euo pipefail
O=$1; IDS=$2; PF=$3; RF=$4
ENGINE=${ENGINE:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2/.work/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
export PATH HOME BARO_SPEC=0 BARO_MEGA=0 BARO_PACK=$PACK BARO_SERVE=1 BARO_STEER_SERVE=1
for r in none persona recipe; do
  case $r in none) sf="";; persona) sf=",\"steer_file\":\"$PF\"";; recipe) sf=",\"steer_file\":\"$RF\"";; esac
  mkdir -p "$O/$r"; echo "{\"id\":1,\"prompt\":[$IDS],\"n\":1,\"top_logprobs\":1$sf}" > "$O/$r.req"
  BARO_DUMP_LOGITS_DIR="$O/$r" timeout 240 "$ENGINE" < "$O/$r.req" > "$O/$r.log" 2>&1 || true
  [ -f "$O/$r/row-0.bin" ] || { echo "FAIL engine row $r: $(grep -E 'rror' "$O/$r.log" | tail -2 | cut -c1-200)"; exit 1; }
done
