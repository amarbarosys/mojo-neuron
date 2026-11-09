#!/usr/bin/env bash
# Gate F1 CPU part (docs/prereg-f1.md): save -> files + persona.json, steer.f32 byte-identical to a direct cvec run, cvec checked by
# tools/cvec-check.py and loaded by llama.cpp; list, apply; mutants: flipped byte -> apply refuses, truncated cvec -> llama.cpp load fails,
# duplicate save and bad name refused. Engine logits part: scripts/persona-engine-check.sh.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
PY=${PY:-$HOME/Projects/mojo/baro-miner/.venv/bin/python}
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}; MOD=qwen2.5-0.5b-instruct-q4_k_m
B() { timeout 280 "$M" build "$1" -I src -I "$GG" -I "$J" -I "$LG" -o "$2" 2> "$2.log" || { grep -A5 -m3 error "$2.log" | cut -c1-240; echo "FAIL build: $2"; exit 1; }; }
B src/nx.mojo .work/nx_persona & B src/cvec.mojo .work/cvec & B src/lens.mojo .work/lens & wait
D=.work/persona-test; rm -rf "$D"; mkdir -p "$D"
export MNEURON_DIR=$PWD/$D/neurons PERSONA_DIR=$PWD/$D/personas NX_CVEC=$PWD/.work/cvec NX_RLOG=$PWD/$D/rlog
.work/lens "$Q" 12 8 --out "$MNEURON_DIR" >/dev/null
nx() { .work/nx_persona "$@"; }
R="12:5:+1.0, 13:7:-0.5"
nx persona save "{\"name\":\"calm\",\"model\":\"$MOD\",\"recipe\":\"$R\",\"labels\":{\"12:5\":\"soft\"}}" > "$D/save.json"
P=$PERSONA_DIR/calm
for f in persona.json steer.f32 steer.cvec.gguf; do [ -s "$P/$f" ] || { echo "FAIL persona: $f missing"; exit 1; }; done
.work/cvec "$Q" "12:5:+1.0,13:7:-0.5" "$D/direct" >/dev/null
cmp -s "$P/steer.f32" "$D/direct.f32" && cmp -s "$P/steer.cvec.gguf" "$D/direct.cvec.gguf" || { echo "FAIL persona: saved vector differs from a direct cvec run"; exit 1; }
$PY -I tools/cvec-check.py "$P/steer.cvec.gguf" "$P/steer.f32" 24 896 >/dev/null || { echo "FAIL persona: cvec-check"; exit 1; }
$PY -I tools/persona-check.py "$P/persona.json" 12:5 soft || exit 1
ll() { gpu-wait run --room cpu --cores 4 --mem 4 --timeout 90 --label "persona-$1" -- tests/llama-cvec-load.sh "$Q" "$2" "$D/$1.log" >/dev/null 2>&1; }
ll good "$P/steer.cvec.gguf" || { echo "FAIL persona: llama.cpp did not load the saved cvec ($D/good.log)"; exit 1; }
head -c 5000 "$P/steer.cvec.gguf" > "$D/trunc.cvec.gguf"
if ll trunc "$D/trunc.cvec.gguf"; then echo "FAIL persona: truncated cvec still loaded (load check does not bite)"; exit 1; fi
nx persona list | grep -q '"name": "calm"' || { echo "FAIL persona: list"; exit 1; }
nx persona apply '{"name":"calm"}' > "$D/apply.json"
grep -q "BARO_STEER=$P/steer.f32" "$D/apply.json" && grep -q -- "--control-vector $P/steer.cvec.gguf" "$D/apply.json" || { echo "FAIL persona: apply output"; exit 1; }
if nx persona save "{\"name\":\"calm\",\"model\":\"$MOD\",\"recipe\":\"$R\"}" >/dev/null 2>&1; then echo "FAIL persona: duplicate save accepted"; exit 1; fi
if nx persona save "{\"name\":\"../x\",\"model\":\"$MOD\",\"recipe\":\"$R\"}" >/dev/null 2>&1; then echo "FAIL persona: bad name accepted"; exit 1; fi
printf '\x7f' | dd of="$P/steer.f32" bs=1 seek=100 conv=notrunc 2>/dev/null
if nx persona apply '{"name":"calm"}' >/dev/null 2>&1; then echo "FAIL persona: apply accepted a corrupted vector"; exit 1; fi
[ "$(grep -c '"tool":"persona"' "$D"/rlog/2*-*.jsonl)" = 3 ] && [ "$(grep -c '"kind":"anomaly"' "$D"/rlog/anomalies.jsonl)" = 1 ] || { echo "FAIL persona: want 3 Rlog lines (save, apply, refused apply) and 1 anomaly"; exit 1; }
echo "PASS persona: save/list/apply, vector byte-identical to direct cvec, cvec-check + llama.cpp load, corruption + truncation + duplicate + bad name refused"
