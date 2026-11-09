#!/usr/bin/env bash
# Gate: build lens, then the f64 brute-force golden + neighbour mutant on two archs (qwen2, llama).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}  # ggufx needs f5da9cf (heap thread ctx) or deq_rows returns zeros
cap() { systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- timeout 300 "$@"; }
cap "$M" build src/lens.mojo -I src -I "$GG" -I "$JS" -o .work/lens || { echo "FAIL build: lens"; exit 1; }
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}
L=${LLAMA:-$HOME/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf}
for m in "$Q:12:8" "$L:8:4"; do IFS=: read -r f l n <<< "$m"
  [ -f "$f" ] || { echo "FAIL model: $f missing"; exit 1; }
  cap .work/lens "$f" "$l" 10 "$n" | tail -1 | grep -q "^PASS" || { echo "FAIL lens check: $f layer $l"; exit 1; }
  echo "ok   lens $(basename "$f") layer $l"
done
tests/rank.sh | tail -1 | grep -q "^PASS rank" || { echo "FAIL rank gate"; exit 1; }
echo "ok   rank golden + mutants"
tests/rlog.sh | tail -1 | grep -q "^PASS rlog" || { echo "FAIL rlog gate"; exit 1; }
echo "ok   rlog normal + anomaly lines, mutant"
cap .work/lens "$Q" 12 8 --out .work/neurons-gate >/dev/null || { echo "FAIL lens --out"; exit 1; }
python3 -I tools/check_lens_bin.py .work/neurons-gate/qwen2.5-0.5b-instruct-q4_k_m/12.lens.bin 151936 || exit 1
tests/modeljson.sh | tail -1 | grep -q "^PASS modeljson" || { echo "FAIL modeljson gate"; exit 1; }
echo "ok   model.json merge (rho kept, gguf absolute)"
cap "$M" build src/conn_check.mojo -I src -I "$GG" -o .work/conn_check || { echo "FAIL build: conn_check"; exit 1; }
cap .work/conn_check "$Q" 10 100 | tail -1 | grep -q "^PASS" || { echo "FAIL conn_check: $Q"; exit 1; }
echo "ok   connections golden + mutant"
cap "$M" build src/fires_check.mojo -I src -I "$GG" -o .work/fires_check || { echo "FAIL build: fires_check"; exit 1; }
QB=${QWEN_BF16:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}
for m in "$QB:The snake slithered away" "$L:The snake slithered"; do f=${m%%:*}; p=${m#*:}
  python3 -I tools/swiglu_oracle.py "$f" "$p" .work/oracle.txt || { echo "FAIL oracle: $f"; exit 1; }
  full=""; if [ "$f" = "$QB" ]; then python3 -I tools/swiglu_full.py "$f" .work/oracle.txt .work/full.f32 || { echo "FAIL full oracle: $f"; exit 1; }; full=.work/full.f32; fi
  cap .work/fires_check "$f" .work/oracle.txt $full | tail -1 | grep -q "^PASS" || { echo "FAIL fires check: $f"; exit 1; }
  echo "ok   fires $(basename "$f") vs llama.cpp ffn_swiglu${full:+ + every activation vs the numpy oracle}"
done
tests/rlens.sh | tail -1 | grep -q "^PASS rlens" || { echo "FAIL rlens gate"; exit 1; }
echo "ok   rlens"
tests/calls.sh | tail -1 | grep -q "^PASS calls" || { echo "FAIL calls gate"; exit 1; }
echo "ok   calls"
tests/sound.sh | tail -1 | grep -q "^PASS sound" || { echo "FAIL sound gate"; exit 1; }
echo "ok   sound (golden WAV, pitch per layer)"
tests/chat.sh | tail -1 | grep -q "^PASS chat" || { echo "FAIL chat gate"; exit 1; }
echo "ok   nx chat (mock server)"
tests/tap.sh | tail -1 | grep -q "^PASS tap" || { echo "FAIL tap gate"; exit 1; }
echo "ok   agent read-path tap (wrong file / no call rejected)"
# lane product
tests/radar.sh | tail -1 | grep -q "^PASS radar" || { echo "FAIL radar gate"; exit 1; }
echo "ok   radar (3 planted clusters, tool-only mutant)"
tests/persona.sh | tail -1 | grep -q "^PASS persona" || { echo "FAIL persona gate"; exit 1; }
echo "ok   persona store (cvec byte-identical, llama.cpp load, corruption refused)"
tests/wiki.sh | tail -1 | grep -q "^PASS wiki" || { echo "FAIL wiki gate"; exit 1; }
echo "ok   wiki map (known unit linked, nonsense unlinked)"
tests/replay.sh | tail -1 | grep -q "^PASS replay" || { echo "FAIL replay gate"; exit 1; }
echo "ok   replay (frames equal capture records, shifted mutant)"
# ## lane probe (CPU halves; the engine halves are scripts/probe-run.sh, scripts/concept-engine-check.sh, scripts/concept-lang-check.sh)
for g in lie lie-steer lie-eval concept circuit; do
  tests/$g.sh | tail -1 | grep -q "^PASS $g" || { echo "FAIL $g gate"; exit 1; }
  echo "ok   $g (lane probe)"
done
echo PASS check
