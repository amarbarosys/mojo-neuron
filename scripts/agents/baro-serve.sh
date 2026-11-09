#!/usr/bin/env bash
# baro-serve.sh up|down|status: one resident baro-serve for nx chat and the agent CLIs (E2, E3), queued through gpu-wait.
# Starts the Rust server directly on the engine and server built by scripts/agents/baro-e1.sh from the E1 commit (steer + capture over HTTP; receipt .work/baro-e1/SHA)
# and the Qwythos-9B pack. It refuses to start on any other binary: a pre-E1 lane build ignores the steer field. `tools/baro serve`
# is not used: it rebuilds the engine from the sources embedded in a BARO GGUF, and that build fails on this toolchain (read(2) signature clash in serve_proto.mojo).
#   STEER_SERVE=1 (default) lets requests carry `steer` (the engine refuses gain otherwise) at the price of batched prefill and speculation; 0 = fast prefill, no steer.
#   env: BARO (checkout), ENGINE, PACK, HF (dir with tokenizer.json + chat_template.jinja), BARO_PORT (18090), BARO_VRAM (12 GB declared), BARO_WAIT (900 s)
set -euo pipefail
cd "$(dirname "$0")/../.."; mkdir -p .work/baro-serve
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro}; PORT=${BARO_PORT:-18090}; URL=http://127.0.0.1:$PORT; L=.work/baro-serve/serve.log
E1=$PWD/.work/baro-e1; ENGINE=${ENGINE:-$E1/engine-9b}; SERVER=${SERVER:-$E1/baro-serve}; PACK=${PACK:-$BARO/.work/e11b-qwythos-pack}
HF=${HF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf}; O=$PWD/.work/baro-serve/pack
fail() { echo "FAIL baro-serve $1: $2"; exit 1; }
up_p() { curl -sf -m 2 $URL/health >/dev/null 2>&1; }
mid() { curl -s $URL/v1/models | python3 -I -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'; }
job() { gpu-wait list --json | python3 -I -c "import json,sys;print(*[j['id'] for j in json.load(sys.stdin)['jobs'] if (j.get('label') or '')=='mneuron-baro-serve-$PORT' and j['state'] in ('waiting','running')])"; }
case ${1:-} in
up)
  up_p && { echo "up (already): $URL model $(mid)"; exit 0; }
  for f in "$ENGINE" "$SERVER" "$PACK/pack.bin" "$HF/tokenizer.json"; do [ -e "$f" ] || fail up "missing $f (scripts/agents/baro-e1.sh builds the engine and server)"; done
  [ "$ENGINE" != "$E1/engine-9b" ] || [ -s "$E1/SHA" ] || fail up "no receipt $E1/SHA: the binaries were not built by baro-e1.sh"
  # the pack dir gets a tokenizer.json beside the pack files (symlinks; the shared pack is not touched)
  rm -rf "$O"; mkdir -p "$O"; for f in "$PACK"/*; do ln -s "$f" "$O/$(basename "$f")"; done; cp "$HF/tokenizer.json" "$O/tokenizer.json"
  if [ -z "$(job)" ]; then : > $L; setsid gpu-wait run --label mneuron-baro-serve-$PORT --priority 10 --vram "${BARO_VRAM:-12}" --timeout 14400 -- \
      env BARO_SPEC=0 BARO_STEER_SERVE=${STEER_SERVE:-1} "$SERVER" --engine "$ENGINE" --pack "$O" --chat-template-file "$HF/chat_template.jinja" --port $PORT > $L 2>&1 & fi  # a queued or running job is joined, never doubled
  for n in $(seq 1 ${BARO_WAIT:-900}); do up_p && break; [ -n "$(job)" ] || [ $n -lt 20 ] || fail up "the server job ended: $(tail -3 $L | tr '\n' ' ')"; sleep 1; done
  up_p || fail up "no /health on $URL after ${BARO_WAIT:-900} s: $(tail -3 $L | tr '\n' ' ')"
  echo "up: $URL model $(mid) (baro $(cat "$E1/SHA" 2>/dev/null | cut -c1-7))";;
down)
  for j in $(job); do gpu-wait cancel "$j" >/dev/null; done
  for _ in $(seq 1 30); do up_p || { echo "down: $URL closed"; exit 0; }; sleep 1; done
  fail down "$URL still answers";;
status) up_p && echo "up: $URL" || echo "down";;
*) fail usage "baro-serve.sh up|down|status";;
esac
