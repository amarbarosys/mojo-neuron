#!/usr/bin/env bash
# Gate F4 (docs/prereg-f4.md), CPU only: enrol qwen2.5-0.5B, Llama-3.2-1B and granite-4.2-3B from the committed probe text (tccap -> C3 sound -> 64-band
# signature), match each from a fresh run, then a held-out model (vibethinker-1.5B) must be flagged unknown; WAVs stay in $D for listening.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
timeout 280 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_fp 2> .work/nx_fp.log || { grep -A5 -m3 error .work/nx_fp.log | cut -c1-240; echo "FAIL build: nx_fp"; exit 1; }
[ -x "${NX_TCCAP:-$HOME/Projects/mojo/mojo-neuron-lanes/features/.work/tccap}" ] || { echo "FAIL setup: tccap (lane features) missing, set NX_TCCAP"; exit 1; }
D=$PWD/.work/fingerprints; rm -rf "$D"; mkdir -p "$D/out"; export FP_DIR=$D NX_RLOG=$D/rlog
declare -A G=([qwen05]=$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf [llama1b]=$HOME/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf [granite3b]=$HOME/Models/granite-4.2-3b-bf16/granite-4.2-3b-BF16-BARO-3c6c25c.gguf [heldout]=${HELDOUT:-$(ls $HOME/Models/vibethinker-1.5b-gguf/*.gguf | head -1)})
declare -A L=([qwen05]=12 [llama1b]=8 [granite3b]=20 [heldout]=14)
call() { gpu-wait run --room cpu --cores 4 --mem 8 --timeout 600 --label "fp-$2-$1" -- bash -c "export FP_DIR='$D' NX_RLOG='$D/rlog' PATH='$PATH' HOME='$HOME'; exec .work/nx_fp fingerprint '{\"op\":\"$1\",\"name\":\"$2\",\"gguf\":\"${G[$2]}\",\"layer\":${L[$2]}${3:+,$3}}'" > "$D/out/$([ $1 = match ] && echo m- || echo e-)$2.json"; }
for n in qwen05 llama1b granite3b; do call enroll $n; done
for n in qwen05 llama1b granite3b; do [ -s "$D/$n.json" ] || { echo "FAIL fingerprint: enroll wrote no $D/$n.json"; exit 1; }; call match $n; done
noise=$(python3 -I -c "import json;print(max(json.load(open('$D/out/m-%s.json'%n))['all'][0]['dist'] for n in 'qwen05 llama1b granite3b'.split()))")
call match heldout "\"noise\":$noise"
python3 -I tools/fp-check.py "$D" "$D/out" qwen05 llama1b granite3b | tee "$D/receipt.txt"
echo "PASS fingerprint: 3 enrolled models matched themselves, the held-out model flagged unknown, shuffled-band control matched none; WAVs in $D"
