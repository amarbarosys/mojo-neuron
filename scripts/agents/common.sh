# sourced by qwen-code.sh, omp.sh, hermes.sh: the server URL, its model id, and the smoke task (read README.md, name its first heading).
# The heading is random per run: only a real file-read tool call can produce it, so the answer proves the tool call.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."; REPO=$PWD
PORT=${BARO_PORT:-18090}; URL=http://127.0.0.1:$PORT; AGENT=$(basename "$0" .sh)
fail() { echo "FAIL agents/$AGENT: $1"; exit 1; }
[ -z "${BARO_STEER_PROFILE:-}" ] || fail "a steer profile needs a server-side default steer; baro-serve takes steer per request only (plan E1), the CLIs cannot add that field"
MODEL=$(curl -sf -m 5 "$URL/v1/models" | python3 -I -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])') || fail "no baro-serve on $URL (scripts/agents/baro-serve.sh up)"
AGENT_URL=$URL  # what the CLI talks to: a logging tap between it and the server
HEAD="Zebra-Quartz-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
SMOKE=$PWD/.work/agents/smoke-$AGENT; rm -rf "$SMOKE"; mkdir -p "$SMOKE"
printf '# %s\n\nA small project.\n\n## Install\n\nnone\n' "$HEAD" > "$SMOKE/README.md"
TASK=${TASK:-"Read the file $SMOKE/README.md with a tool and tell me its first heading, exactly as written."}  # absolute path: an agent that starts in the repo root reads the wrong README.md
OUT=$SMOKE/answer.txt
TAPPORT=$((PORT + 1)); TAP=$SMOKE/tap.jsonl; : > "$TAP"
python3 -I $REPO/tools/openai-tap.py $TAPPORT "$URL" "$TAP" & TAPPID=$!; trap 'kill $TAPPID 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf -m 1 http://127.0.0.1:$TAPPORT/health >/dev/null 2>&1 && break; sleep 0.1; done
AGENT_URL=http://127.0.0.1:$TAPPORT
verdict() {  # two separate checks: the tool calls asked for the right file (tap log), and the answer names its heading
  local p=FAIL a=FAIL rd; rd=$(python3 -I $REPO/tools/tap-reads.py "$TAP" "$SMOKE") && p=PASS; grep -qF "$HEAD" "$OUT" && a=PASS
  echo "$AGENT: read-path $p ($rd) answer $a"
  [ $p = PASS ] && [ $a = PASS ] && { echo "PASS $AGENT: $MODEL read $SMOKE/README.md and named $HEAD"; return 0; }
  tail -3 "$OUT" | cut -c1-200; echo "FAIL agents/$AGENT: read-path $p, answer $a (read-path FAIL = wrong or no file, answer FAIL alone = the model misread the right file)"; exit 1; }
