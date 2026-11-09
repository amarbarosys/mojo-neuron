#!/usr/bin/env bash
# Gates E2 (live) and E3: one resident baro-serve on Qwythos-9B (Q4_0-pure pack, engine-9b), a scripted 3-turn `nx chat` with rlog lines, then the smoke task
# (read README.md, name its first heading) through qwen-code, omp and hermes. Every agent runs even if one fails; the server is taken down at the end.
set -uo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/agentsgate
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
D=$PWD/.work/agentsgate; bad=0; fail() { echo "FAIL agents-live: $1"; bad=1; }
rm -f .work/nx_live; timeout 400 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_live 2>&1 | grep -E "error" -A4 && { echo "FAIL agents-live: build nx"; exit 1; }
[ -x .work/nx_live ] || { echo "FAIL agents-live: no .work/nx_live"; exit 1; }
BARO_WAIT=${BARO_WAIT:-2400} scripts/agents/baro-serve.sh up || exit 1
trap 'scripts/agents/baro-serve.sh down' EXIT
printf 'What is the capital of France?\nAnd of Italy?\nWhich of the two is larger?\n' > $D/s3.txt; rm -rf $D/rlog; export NX_RLOG=$D/rlog
out=$(.work/nx_live chat --script $D/s3.txt --max 48) || fail "nx chat exited non-zero: $out"
[ "$(echo "$out" | grep -c '^model> .\+')" = 3 ] || fail "chat did not print 3 non-empty replies: $out"
python3 -I - $D/rlog <<'PY' || bad=1
import glob, json, re, sys
r = [json.loads(l) for f in glob.glob(sys.argv[1] + "/2*.jsonl") for l in open(f)]; r = [x for x in r if x["tool"] == "chat"]
pt = [int(re.search(r"prompt_tokens=(\d+)", x["result"]).group(1)) for x in r]
print("chat rlog: %d lines, kinds %s, prompt tokens %s" % (len(r), sorted({x["kind"] for x in r}), pt))
ok = len(r) == 3 and all(x["kind"] == "normal" for x in r) and pt[0] < pt[1] < pt[2]
sys.exit(0 if ok else print("FAIL agents-live: chat rlog is not 3 normal lines with growing prompt tokens") or 1)
PY
echo "$out" | sed 's/^/  /' | cut -c1-160
# E1 steer over HTTP: the same prompt with and without a steer item gives different replies, and both turns are in the rlog
printf 'What is the capital of France?\n' > $D/s1.txt
plain=$(.work/nx_live chat --script $D/s1.txt --max 40) || fail "plain chat failed"
steered=$(.work/nx_live chat --script $D/s1.txt --max 40 --steer "${STEER:-16:100:100}") || fail "steered chat failed"
[ "$plain" != "$steered" ] && echo "ok   steer ${STEER:-16:100:100}: reply changed ($(echo "$plain" | head -c 50 | tr '\n' ' ') -> $(echo "$steered" | head -c 50 | tr '\n' ' '))" || fail "the steer field did not change the reply (server without E1?)"
echo "  baro build: $(cat .work/baro-e1/SHA | cut -c1-7)"
: > $D/agents.out
for a in qwen-code omp hermes; do scripts/agents/$a.sh 2>&1 | tee -a $D/agents.out || bad=1; done
rp=$(grep -c "read-path PASS" $D/agents.out || true); an=$(grep -c "answer PASS" $D/agents.out || true)
echo "agents: read-path correct $rp/3, answer correct $an/3"
[ $bad = 0 ] && echo "PASS agents-live: chat 3 turns, steer changes the reply, qwen-code/omp/hermes read the right file ($rp/3) and named its heading ($an/3)" || echo "FAIL agents-live: see lines above"
exit $bad
