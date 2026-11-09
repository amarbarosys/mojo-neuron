#!/usr/bin/env bash
# Gate: the calls (src/nx.mojo) against independent paths: lens.bin read vs lens text output, recipe refusals, connections vs conn_check,
# dossier file is valid JSON with its sections, brain export is found by brain-ask (cois mode) after a sync, an unknown token is not.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}
"$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}" -o .work/nx 2>&1 | grep -E "error" -A4 && { echo "FAIL build: nx"; exit 1; } || true
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}; MOD=qwen2.5-0.5b-instruct-q4_k_m
export MNEURON_DIR=$PWD/.work/neurons-calls; rm -rf "$MNEURON_DIR"
[ -f .work/lens ] || { echo "FAIL calls: build lens first (scripts/check.sh)"; exit 1; }
.work/lens "$Q" 12 8 --out "$MNEURON_DIR" >/dev/null
nx() { .work/nx "$@"; }
# 1. lens call == text output of the lens tool, three neurons
.work/lens "$Q" 12 8 > .work/lens12.tsv
for n in 5 777 4000; do
  want=$(awk -F'\t' -v n=$n '$1==n{split($2,a,"\xe2\x90\x9f"); print a[1]}' .work/lens12.tsv)
  got=$(nx lens "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":$n}" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["top"][0][0])')
  [ "$want" = "$got" ] || { echo "FAIL lens call: neuron $n top token '$got' != text tool '$want'"; exit 1; }
done
echo "ok   lens call = lens text output (3 neurons)"
# 2. recipe: normalises, refuses layer 0, neuron out of range, bad shape
nx recipe "{\"model\":\"$MOD\",\"recipe\":\"12:5:+1.0, 13:7:-0.5\"}" | grep -q '"recipe": "12:5:+1.0,13:7:-0.5"' || { echo "FAIL recipe: normal form"; exit 1; }
for bad in "0:5:1.0" "12:99999:1.0" "12:5" "30:1:1.0"; do
  if nx recipe "{\"model\":\"$MOD\",\"recipe\":\"$bad\"}" >/dev/null 2>&1; then echo "FAIL recipe: accepted '$bad'"; exit 1; fi
done
echo "ok   recipe (normal form, 4 refusals)"
# 3. connections call returns the pair conn_check verified
nx connections "{\"model\":\"$MOD\",\"layer\":10,\"neuron\":100,\"k\":3,\"span\":2}" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); assert len(d["down"])==3 and len(d["up"])==3 and all(-1<=c["cos"]<=1 for c in d["down"]+d["up"])' || { echo "FAIL connections call"; exit 1; }
echo "ok   connections call"
# 4. dossier: file written, valid JSON, sections present
nx dossier "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":3351,\"text\":\"The snake slithered away\"}" >/dev/null
python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["v"]==1 and d["lens"]["top"] and d["connections"]["down"] and d["fires"][0]["activations"] and len(d["fires"][0]["tokens"])==len(d["fires"][0]["activations"])' "$MNEURON_DIR/$MOD/12.dossier/3351.json" || { echo "FAIL dossier file"; exit 1; }
echo "ok   dossier (lens, connections, fires, file)"
# 5. steer lens in the dossier: a hand-written steer.bin (docs/format.md) comes back with its numbers
python3 -I - "$MNEURON_DIR/$MOD/12.steer.bin" <<'PY'
import struct, sys
k, ffn = 2, 4864
b = bytearray(b'MNST' + struct.pack('<5I', 1, 12, ffn, k, 0) )
struct.pack_into('<f', b, 20, 0.5)
rec = [struct.pack('<2i2f2i2f', -1, -1, 0, 0, -1, -1, 0, 0)] * ffn
rec[3] = struct.pack('<2i2f2i2f', 10, 11, 1.5, 1.25, 20, 21, -1.0, -2.0)
open(sys.argv[1], 'wb').write(bytes(b) + b''.join(rec))
PY
nx dossier "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":3,\"save\":0}" | python3 -I -c 'import json,sys; s=json.load(sys.stdin)["steer"]["slens"][0]; assert s["dose"]==0.5 and [x[1] for x in s["up"]]==[1.5,1.25] and [x[1] for x in s["down"]]==[-1.0,-2.0] and all(x[0] for x in s["up"]+s["down"]), s' || { echo "FAIL dossier: steer lens"; exit 1; }
nx dossier "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":4,\"save\":0}" | python3 -I -c 'import json,sys; s=json.load(sys.stdin)["steer"]["slens"][0]; assert s["up"]==[] and s["down"]==[]' || { echo "FAIL dossier: empty steer record"; exit 1; }
echo "ok   dossier steer lens (hand-written steer.bin)"
# dossier keeps the engine's unsteered rows ("neuron":"base", one per prompt) keyed by prompt (codex A1)
python3 -I - "$MNEURON_DIR/$MOD/12.sweep.jsonl" <<'PY'
import json, sys
rows = [("base", 0.0, "P1", "BASE ONE"), ("base", 0.0, "P2", "BASE TWO"), ("ctrl0", 0.5, "P1", "ctrl"),
        (3, -0.5, "P1", "NEG ONE"), (3, 0.5, "P1", "POS ONE"), (3, -0.5, "P2", "NEG TWO"), (3, 0.5, "P2", "POS TWO"), (4, 0.5, "P1", "other")]
open(sys.argv[1], "w").write("".join(json.dumps({"model": "m", "layer": 12, "neuron": n, "prompt": p, "dose": d, "text": t, "tokens": [1]}) + "\n" for n, d, p, t in rows))
PY
nx dossier "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":3,\"save\":0}" | python3 -I -c '
import json,sys; s=json.load(sys.stdin)["steer"]
assert {b["prompt"]: b["text"] for b in s["base"]} == {"P1": "BASE ONE", "P2": "BASE TWO"}, s["base"]
assert {(r["prompt"], r["dose"]): r["text"] for r in s["rollouts"]} == {("P1",-0.5):"NEG ONE",("P1",0.5):"POS ONE",("P2",-0.5):"NEG TWO",("P2",0.5):"POS TWO"}, s["rollouts"]' || { echo "FAIL dossier: base rows not kept by prompt"; exit 1; }
echo "ok   dossier keeps base rows keyed by prompt"
# search_lens honours k: a token with at least 3 hits returns exactly k when k is smaller
tok=$(python3 -I - "$MOD" <<'PY'
import json, subprocess, sys
mod = sys.argv[1]
for n in range(0, 60):
    t = json.loads(subprocess.run(['.work/nx', 'lens', json.dumps({"model": mod, "layer": 12, "neuron": n})], capture_output=True, text=True, check=True).stdout)["top"][0][0].strip()
    r = json.loads(subprocess.run(['.work/nx', 'search_lens', json.dumps({"model": mod, "layer": 12, "token": t, "k": 100000})], capture_output=True, text=True, check=True).stdout)
    if len(r) >= 3: print(t); break
else: sys.exit("no token with 3 hits among the first 60 neurons")
PY
) || { echo "FAIL search_lens: no multi-hit token"; exit 1; }
for k in 1 2; do c=$(nx search_lens "{\"model\":\"$MOD\",\"layer\":12,\"token\":\"$tok\",\"k\":$k}" | python3 -I -c 'import json,sys; print(len(json.load(sys.stdin)))'); [ "$c" = "$k" ] || { echo "FAIL search_lens: k=$k returned $c hits for '$tok'"; exit 1; }; done
echo "ok   search_lens honours k"
# 6. brain export found by brain-ask (cois mode); a token no neuron has is not found
top=$(nx lens "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":3351}" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["top"][0][0].strip())')
nx export_brain "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":3351,\"sync\":1}" >/dev/null
BRAIN_ASK_MODE=bm25 brain-ask "$top neuron layer 12" 2>&1 | grep -q "L12-N3351.md" || { echo "FAIL brain-ask: exported neuron not found by '$top'"; exit 1; }
if BRAIN_ASK_MODE=bm25 brain-ask "zzqxjv" 2>&1 | grep -q "L12-N3351.md"; then echo "FAIL brain-ask: mutant query found the neuron"; exit 1; fi
echo "ok   brain export found by brain-ask, mutant query not"
echo PASS calls
