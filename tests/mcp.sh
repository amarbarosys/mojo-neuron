#!/usr/bin/env bash
# Gate: MCP server over a pipe. initialize, tools/list (9 tools with schemas), tools/call of lens / connections / recipe / fires equals
# the nx CLI answer for the same call, a failing call is an isError result, an unknown method is a -32601 error, a notification gets no reply.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}
"$M" build src/mcp.mojo -I src -I "$GG" -I "$J" -I "${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}" -o .work/mcp 2>&1 | grep -E "error" -A4 && { echo "FAIL build: mcp"; exit 1; } || true
[ -x .work/nx ] || { echo "FAIL mcp: build nx first (tests/calls.sh)"; exit 1; }
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}; MOD=qwen2.5-0.5b-instruct-q4_k_m
export MNEURON_DIR=$PWD/.work/neurons-mcp; rm -rf "$MNEURON_DIR"; .work/lens "$Q" 12 8 --out "$MNEURON_DIR" >/dev/null
export MNEURON_BRAIN=$PWD/.work/brain-mcp; rm -rf "$MNEURON_BRAIN" .work/sync.marker; export MNEURON_SYNC="touch $PWD/.work/sync.marker"
rpc() { printf '%s\n' "$@" | .work/mcp; }
call() { printf '{"jsonrpc":"2.0","id":%s,"method":"tools/call","params":{"name":"%s","arguments":%s}}' "$1" "$2" "$3"; }
out=$(rpc '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  "$(call 3 lens "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":5}")" \
  "$(call 4 recipe "{\"model\":\"$MOD\",\"recipe\":\"12:5:+1.0\"}")" \
  "$(call 5 recipe "{\"model\":\"$MOD\",\"recipe\":\"0:5:1.0\"}")" \
  "$(call 6 fires "{\"model\":\"$MOD\",\"layer\":12,\"text\":\"The snake slithered away\",\"k\":3}")" \
  "$(call 7 connections "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":5,\"k\":3}")" \
  '{"jsonrpc":"2.0","id":8,"method":"nope"}' \
  '{"jsonrpc":"2.0","id":9,"method":"ping"}' \
  '{"jsonrpc":"2.0","id":10,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}' \
  '{"jsonrpc":"2.0","id":11,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}' \
  "$(call 12 export_brain "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":5}")")
echo "$out" > .work/mcp.out
python3 -I - "$MOD" <<'PY' || { echo "FAIL mcp: protocol check"; exit 1; }
import json, subprocess, sys, os
mod = sys.argv[1]
rs = {}
for l in open('.work/mcp.out'):
    d = json.loads(l); rs[d['id']] = d
assert sorted(rs) == [1,2,3,4,5,6,7,8,9,10,11,12], f"replies for ids {sorted(rs)} (a notification must get none)"
assert rs[1]['result']['serverInfo']['name'] == 'mojo-neuron' and 'tools' in rs[1]['result']['capabilities']
tl = rs[2]['result']['tools']
assert [t['name'] for t in tl] == ['models','lens','search_lens','rank','connections','fires','dossier','recipe','export_brain'], [t['name'] for t in tl]
assert all(t['inputSchema']['type'] == 'object' for t in tl)
def txt(i): return rs[i]['result']['content'][0]['text']
for i, name, args in [(3,'lens',{"model":mod,"layer":12,"neuron":5}), (4,'recipe',{"model":mod,"recipe":"12:5:+1.0"}),
                      (7,'connections',{"model":mod,"layer":12,"neuron":5,"k":3}),
                      (6,'fires',{"model":mod,"layer":12,"text":"The snake slithered away","k":3})]:
    assert rs[i]['result']['isError'] is False, (name, txt(i))
    cli = subprocess.run(['.work/nx', name, json.dumps(args)], capture_output=True, text=True, check=True).stdout
    assert json.loads(txt(i)) == json.loads(cli), f"{name}: MCP differs from the CLI"
assert rs[5]['result']['isError'] is True and 'layer' in txt(5), txt(5)
assert rs[8]['error']['code'] == -32601
assert rs[9]['result'] == {}
ok = {'2024-11-05', '2025-03-26', '2025-06-18'}
assert rs[10]['result']['protocolVersion'] in ok and rs[10]['result']['protocolVersion'] != '1999-01-01', rs[10]['result']['protocolVersion']
assert rs[11]['result']['protocolVersion'] == '2025-03-26', "a supported version must be echoed"
assert rs[12]['result']['isError'] is False, txt(12)
assert os.path.exists('.work/sync.marker'), "export_brain over MCP did not index the note (its description promises it)"
print("ok protocol, 9 tools, 4 calls equal the CLI, error paths")
PY
echo "PASS mcp"
