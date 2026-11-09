#!/usr/bin/env bash
# Gate (GPU, about 30 s, logged in the GPU ledger): the live steered chat call runs the engine lane's resident sweep for one neuron at
# -dose and +dose plus the unsteered row, and returns the same texts as invoking the engine's sweep directly (greedy decoding, so equal).
# Also: a missing live.json and a broken sweep path are errors, not empty answers; an MCP client cannot start a GPU job.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
E=${ENGINE:-$HOME/Projects/mojo/mojo-neuron-lanes/engine}; BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}
GG=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}; MOD=qwen2.5-0.5b-instruct-q4_k_m
for f in "$E/.work/sweep" "$BARO/.work/steer/spark-q05" "$E/.work/steer/q05/rho.f32" "$GG"; do [ -e "$f" ] || { echo "FAIL live: $f missing (build the engine lane first)"; exit 1; }; done
export MNEURON_DIR=$PWD/.work/neurons-live; rm -rf "$MNEURON_DIR" .work/live-direct
export MNEURON_LEDGER=${LEDGER:-$HOME/Projects/mojo/mojo-neuron/.work/gpu-ledger.md}
.work/lens "$Q" 12 8 --out "$MNEURON_DIR" >/dev/null
[ -x .work/nx ] || { echo "FAIL live: build nx first (tests/calls.sh)"; exit 1; }
ARGS="{\"model\":\"$MOD\",\"layer\":12,\"neuron\":3351,\"prompt\":\"Who are you?\",\"dose\":0.5,\"n\":24}"
if .work/nx live "$ARGS" >.work/live.err 2>&1; then echo "FAIL live: ran without live.json"; exit 1; fi
grep -q "live.json" .work/live.err || { echo "FAIL live: error does not name live.json: $(cat .work/live.err)"; exit 1; }
CFG="{\"sweep\":\"$E/.work/sweep\",\"gguf\":\"$GG\",\"spark\":\"$BARO/.work/steer/spark-q05\",\"pack\":\"$BARO/.work/abl/q/pack\",\"baro\":\"$BARO\",\"rho\":\"$E/.work/steer/q05/rho.f32\"}"
echo "$CFG" | sed 's#"sweep":"[^"]*"#"sweep":"/nonexistent/sweep"#' > "$MNEURON_DIR/$MOD/live.json"
if .work/nx live "$ARGS" >.work/live.err 2>&1; then echo "FAIL live: broken sweep path returned an answer"; exit 1; fi
echo "$CFG" > "$MNEURON_DIR/$MOD/live.json"
.work/nx live "$ARGS" > .work/live.out || { echo "FAIL live: call failed: $(cat .work/live.err)"; exit 1; }
# the oracle: the engine's sweep invoked directly with the same arguments
D=$PWD/.work/live-direct; mkdir -p "$D"; printf 'Who are you?\n' > "$D/p.txt"
gpu-wait run --label app-live-direct --timeout 300 -- "$E/.work/sweep" "$GG" "$D" layer=12 neurons=3351 doses=-0.5,0.5 prompts="$D/p.txt" ctrl=0 n=24 \
  rho="$E/.work/steer/q05/rho.f32" spark="$BARO/.work/steer/spark-q05" pack="$BARO/.work/abl/q/pack" baro="$BARO" >/dev/null 2>&1 || { echo "FAIL live: direct engine run failed"; exit 1; }
python3 -I - .work/live.out .work/live-direct/12.sweep.jsonl <<'PY' || { echo "FAIL live: texts differ from the direct engine run"; exit 1; }
import json, sys
r = json.load(open(sys.argv[1])); rows = [json.loads(l) for l in open(sys.argv[2])]
want = {"base": [x for x in rows if x["neuron"] == "base"][0]["text"], "neg": [x for x in rows if x["neuron"] == 3351 and x["dose"] < 0][0]["text"],
        "pos": [x for x in rows if x["neuron"] == 3351 and x["dose"] > 0][0]["text"]}
for k, v in want.items(): assert r[k] == v and v.strip(), (k, r[k], v)
assert len({want["base"], want["neg"], want["pos"]}) == 3, "steering changed nothing"
assert r["prompt"] == "Who are you?" and r["wall_s"] > 0
PY
tail -n 5 "$MNEURON_LEDGER" | grep -q "app | live chat sweep $MOD L12 N3351" || { echo "FAIL live: GPU ledger line missing"; exit 1; }
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"live","arguments":{}}}' | .work/mcp | grep -q "not offered over MCP" || { echo "FAIL live: MCP can start a GPU job"; exit 1; }
echo "PASS live: engine sweep via nx live = direct engine run (base, -0.5, +0.5 texts), errors loud, ledger line written, not offered over MCP"
