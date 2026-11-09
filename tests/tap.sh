#!/usr/bin/env bash
# Gate for the agent read-path check (CPU): the tap passes requests and replies through unchanged and logs the request body; tap-reads tells a right
# README.md read from a wrong-file read and from no tool call at all (the cases the answer text alone cannot separate).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/tapgate; D=$PWD/.work/tapgate; rm -f $D/*; fail() { echo "FAIL tap: $1"; exit 1; }
python3 -I tools/mock-openai.py 18192 $D/up.jsonl & A=$!; python3 -I tools/openai-tap.py 18193 http://127.0.0.1:18192 $D/tap.jsonl & B=$!; trap 'kill $A $B 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf -m 1 http://127.0.0.1:18193/health >/dev/null 2>&1 && break; sleep 0.1; done
req='{"model":"m","messages":[{"role":"user","content":"hi"}]}'
r1=$(curl -s -m 5 -H 'content-type: application/json' -d "$req" http://127.0.0.1:18192/v1/chat/completions); r2=$(curl -s -m 5 -H 'content-type: application/json' -d "$req" http://127.0.0.1:18193/v1/chat/completions)
[ "$r1" = "$r2" ] || fail "reply through the tap differs: $r1 vs $r2"
[ "$(wc -l < $D/tap.jsonl)" = 1 ] && grep -q '"content": "hi"' $D/tap.jsonl || fail "tap did not log the request body"
S=/x/smoke
mk() { python3 -I - "$1" <<'PY'
import json, sys
calls = [json.loads(sys.argv[1])] if sys.argv[1] != "none" else []
msgs = [{"role": "user", "content": "q"}] + [{"role": "assistant", "content": "", "tool_calls": [{"id": "c%d" % i, "type": "function", "function": {"name": "read", "arguments": json.dumps(c)}}]} for i, c in enumerate(calls)]
print(json.dumps({"messages": msgs}))
PY
}
chk() { mk "$1" > $D/l.jsonl; python3 -I tools/tap-reads.py $D/l.jsonl $S >/dev/null && echo pass || echo fail; }
[ "$(chk '{"path":"/x/smoke/README.md"}')" = pass ] || fail "right absolute path rejected"
[ "$(chk '{"file_path":"README.md"}')" = pass ] || fail "relative path rejected"
[ "$(chk '{"path":"/repo/README.md"}')" = fail ] || fail "the repo-root README.md (wrong file) accepted"
[ "$(chk none)" = fail ] || fail "no tool call accepted"
[ "$(chk '{"path":"/x/smoke/notes.txt"}')" = fail ] || fail "a read of another file accepted"
echo "PASS tap: pass-through identical, body logged, right path / relative accepted, wrong file / no call / other file rejected"
