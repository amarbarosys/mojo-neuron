#!/usr/bin/env bash
# Gate F1c (docs/prereg-f1c.md): a persona picked by name goes to baro-serve as E1's steer object {"file": ...} and acts exactly like the engine run
# directly with steer_file. Engine and server are the E1-verified binaries (mojo-baro-lanes/verify, built from baro neuron2 7c37795); the gate prints
# and checks that commit. One gpu-wait job: direct engine row, then the server with three HTTP requests (plain, persona 1, persona 2).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG_SRC=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/verify}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}; SERVE=${SERVE:-$BARO/serve/target/release/baro-serve}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; PORT=${PORT:-18473}
GGUF=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
sha=$(git -C "$BARO" rev-parse --short HEAD); case "$sha" in 7c37795*) ;; *) echo "FAIL setup: $BARO is at $sha, want the E1 sha 7c37795"; exit 1;; esac
[ -x "$ENGINE" ] && [ -x "$SERVE" ] || { echo "FAIL setup: engine $ENGINE or server $SERVE missing"; exit 1; }
echo "binaries: baro $sha, engine $(stat -c %y "$ENGINE" | cut -c1-16), server $(stat -c %y "$SERVE" | cut -c1-16)"
timeout 280 "$M" build src/nx.mojo -I src -I "$GG_SRC" -I "$J" -I "$LG" -o .work/nx_persona 2> .work/nx_persona.log || { grep -A5 -m3 error .work/nx_persona.log | cut -c1-240; echo "FAIL build: nx_persona"; exit 1; }
timeout 280 "$M" build src/cvec.mojo -I src -I "$GG_SRC" -o .work/cvec >/dev/null 2>&1 || { echo "FAIL build: cvec"; exit 1; }
O=$PWD/.work/persona-serve; rm -rf "$O"; mkdir -p "$O/row"; export PERSONA_DIR=$O/personas NX_CVEC=$PWD/.work/cvec NX_RLOG=$O/rlog
R1=8:100:30,20:50:120; R2=20:50:120
for p in "p1 $R1" "p2 $R2"; do set -- $p; .work/nx_persona persona save "{\"name\":\"$1\",\"gguf\":\"$GGUF\",\"recipe\":\"$2\"}" >/dev/null; done
.work/cvec "$GGUF" "$R1" "$O/direct" >/dev/null
f1=$(.work/nx_persona persona apply '{"name":"p1"}' | python3 -I -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["steer"]))')
f2=$(.work/nx_persona persona apply '{"name":"p2"}' | python3 -I -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["steer"]))')
IDS=$(gpu-wait run --room cpu --cores 2 --mem 2 --timeout 60 --label persona-serve-tok -- bash -c "~/Models/llama.cpp/build/bin/llama-tokenize -m '$GGUF' --ids -p 'What is the capital of France?' 2>/dev/null | tail -1 | tr -d '[] '" | tail -1)
[ -n "$IDS" ] || { echo "FAIL tokenize"; exit 1; }
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
set -uo pipefail
export PATH='$PATH' HOME='$HOME'
echo '{"id":1,"prompt":[$IDS],"n":1,"top_logprobs":1,"steer_file":"$O/direct.f32"}' > '$O/row.req'
env BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_MEGA=0 BARO_PACK='$PACK' BARO_DUMP_LOGITS_DIR='$O/row' '$ENGINE' < '$O/row.req' > '$O/row.log' 2>&1
env BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK' '$SERVE' --engine '$ENGINE' --pack '$PACK' --port $PORT > '$O/serve.log' 2>&1 &
SP=\$!
for i in \$(seq 1 150); do curl -sf localhost:$PORT/health > /dev/null && break; sleep 1; done
req() { curl -s -m 120 localhost:$PORT/v1/completions -H 'content-type: application/json' -d "\$1"; }
req '{"prompt":[$IDS],"max_tokens":1,"temperature":0,"logits_topk":20}' > '$O/plain.json'
req '{"prompt":[$IDS],"max_tokens":1,"temperature":0,"logits_topk":20,"steer":$f1}' > '$O/p1.json'
req '{"prompt":[$IDS],"max_tokens":1,"temperature":0,"logits_topk":20,"steer":$f2}' > '$O/p2.json'
kill \$SP; wait \$SP 2>/dev/null
exit 0
EOS
chmod +x "$O/run.sh"
gpu-wait run --timeout 900 --vram 12 --label persona-serve -- "$O/run.sh" || true
for f in plain p1 p2; do [ -s "$O/$f.json" ] || { echo "FAIL http $f: no response: $(tail -2 "$O/serve.log" | cut -c1-200)"; exit 1; }; done
[ -f "$O/row/row-0.bin" ] || { echo "FAIL direct engine row: $(grep -E 'rror' "$O/row.log" | tail -2 | cut -c1-200)"; exit 1; }
python3 -I tools/serve-diff.py "$O" | tee "$O/receipt.txt"
echo "PASS persona-serve: persona picked by name over HTTP == direct steer_file run, differs from plain and from another persona (baro $sha)"
