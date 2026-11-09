#!/usr/bin/env bash
# Gate E2 (mock server, CPU): nx chat keeps the history (1, 3, 5 messages), sends the steer field as [{layer, neuron, scale}], prints each reply,
# writes the transcript, ends every turn with an rlog line (3 normal), and a reply without choices gives an anomaly line and a non-zero exit.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/chatgate
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
D=$PWD/.work/chatgate; fail() { echo "FAIL chat: $1"; exit 1; }; PORT=18191
rm -f .work/nx_chat; timeout 400 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_chat 2>&1 | grep -E "error" -A4 && fail "build nx" || true
[ -x .work/nx_chat ] || fail "no .work/nx_chat"
rm -rf $D/* .work/chat; python3 -I tools/mock-openai.py $PORT $D/req.jsonl & MP=$!; trap 'kill $MP 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf -m 1 http://127.0.0.1:$PORT/health >/dev/null && break; sleep 0.1; done
printf 'hello\nsecond\nthird\n' > $D/s3.txt; export NX_RLOG=$D/rlog
out=$(.work/nx_chat chat --url http://127.0.0.1:$PORT --script $D/s3.txt --steer 16:100:+0.5,8:7:-1.25 --max 8)
[ "$out" = "$(printf 'model> reply 1\nmodel> reply 2\nmodel> reply 3')" ] || fail "replies: $out"
python3 -I - $D/req.jsonl <<'PY' || exit 1
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
assert [len(x["messages"]) for x in r] == [1, 3, 5], [len(x["messages"]) for x in r]
assert r[2]["messages"][1] == {"role": "assistant", "content": "reply 1"} and r[2]["messages"][4]["content"] == "third", "history order"
assert all(x["steer"] == [{"layer": 16, "neuron": 100, "scale": 0.5}, {"layer": 8, "neuron": 7, "scale": -1.25}] for x in r), r[0].get("steer")
assert all(x["max_tokens"] == 8 and x["stream"] is False and x["model"] == "mock-model" for x in r)
PY
n=$(grep -c '"kind":"normal".*"tool":"chat"\|"tool":"chat".*"kind":"normal"' $D/rlog/2*.jsonl); [ "$n" = 3 ] || fail "rlog normal lines: $n (want 3)"
grep -q 'tok_s=' $D/rlog/2*.jsonl && grep -q 'steer=16:100:+0.5' $D/rlog/2*.jsonl || fail "rlog line lacks tok_s or the steer"
[ "$(wc -l < $(ls .work/chat/2*.jsonl | head -1))" = 6 ] || fail "transcript is not 3 user + 3 assistant lines"
rm -f $D/req.jsonl; sin=$(printf 'hello\nbye\n/quit\n' | .work/nx_chat chat --url http://127.0.0.1:$PORT | tr -d '\n')
[ "$sin" = "you> model> reply 1you> model> reply 2you> " ] && [ "$(wc -l < $D/req.jsonl)" = 2 ] || fail "piped stdin chat: $sin"
msg=$(.work/nx_chat chat --url http://127.0.0.1:1 --script $D/s3.txt 2>&1 || true); echo "$msg" | grep -q "no server answers" || fail "a missing server gave no clear error: $msg"
printf 'FAIL\n' > $D/bad.txt
if .work/nx_chat chat --url http://127.0.0.1:$PORT --script $D/bad.txt >/dev/null 2>&1; then fail "an empty reply did not fail the chat"; fi
grep -q '"kind":"anomaly"' $D/rlog/anomalies.jsonl || fail "no anomaly line for the reply without choices"
# lane product (F1c, docs/prereg-f1c.md): --persona NAME posts steer {"file": <persona steer.f32>}; --persona with --steer is refused
export PERSONA_DIR=$D/personas; mkdir -p $PERSONA_DIR/calm; printf 'vec' > $PERSONA_DIR/calm/steer.f32; printf 'cv' > $PERSONA_DIR/calm/steer.cvec.gguf
printf '{"v":1,"name":"calm","model":"","gguf":"","recipe":"1:1:1","sha_f32":"%s","sha_cvec":"%s"}' "$(sha256sum < $PERSONA_DIR/calm/steer.f32 | cut -c1-16)" "$(sha256sum < $PERSONA_DIR/calm/steer.cvec.gguf | cut -c1-16)" > $PERSONA_DIR/calm/persona.json
rm -f $D/req.jsonl; .work/nx_chat chat --url http://127.0.0.1:$PORT --script $D/s3.txt --persona calm >/dev/null
python3 -I - $D/req.jsonl $PERSONA_DIR/calm/steer.f32 <<'PY' || fail "persona steer field"
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
assert len(r) == 3 and all(x["steer"] == {"file": sys.argv[2]} for x in r), r[0].get("steer")
PY
if .work/nx_chat chat --url http://127.0.0.1:$PORT --script $D/s3.txt --persona calm --steer 16:1:1 >/dev/null 2>&1; then fail "--persona with --steer was accepted"; fi
if .work/nx_chat chat --url http://127.0.0.1:$PORT --script $D/s3.txt --persona nosuch >/dev/null 2>&1; then fail "an unknown persona was accepted"; fi
echo "PASS chat: persona steer field, 3 turns, piped stdin, missing-server error, history 1/3/5, steer field, 3 normal rlog lines, no-choices reply -> anomaly + non-zero exit"
