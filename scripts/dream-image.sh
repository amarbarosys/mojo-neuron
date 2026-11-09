#!/usr/bin/env bash
# dream-image.sh RUN [UNIT]: one ComfyUI image of the final dream text of UNIT (default: first feature), opened detached.
# One resident ComfyUI inside a gpu-wait job (21 GB), never beside the engine.
set -euo pipefail
cd "$(dirname "$0")/.."; RUN=${1:?usage: dream-image.sh RUN [UNIT]}; O=$PWD/.work/dreams/$RUN; U=${2:-1}
[ -s "$O/dream.jsonl" ] || { echo "FAIL setup: no $O/dream.jsonl"; exit 1; }
jq -c -n --arg p "$(sed -n "$((U+1))p" "$O/dream.jsonl" | jq -r .text | tr '\n' ' ' | cut -c1-300)" \
  '[{id:"dream",kind:"zimage",prompt:$p,size:[768,768],seed:1}]' > "$O/jobs.json"
gpu-wait run --vram 21 --timeout 900 -- "$HOME/iTools/bin/with-comfy" -- /usr/bin/python3 -I "$HOME/iTools/design/comfy-run/comfy_run.py" "$O/jobs.json" --out "$O/img" || { echo "FAIL image: comfy-run"; exit 1; }
f=$(ls "$O"/img/dream*.{png,jpg} 2>/dev/null | head -1); [ -n "$f" ] && [ "$(stat -c %s "$f")" -gt 10240 ] || { echo "FAIL image: no output over 10 KB in $O/img"; exit 1; }
echo "$f"; setsid xdg-open "$f" >/dev/null 2>&1 &
