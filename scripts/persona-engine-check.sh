#!/usr/bin/env bash
# Gate F1 engine part (docs/prereg-f1.md): a saved persona applied as a per-request steer_file gives the same first-step logits as the
# recipe vector built directly, and not the unsteered logits. One gpu-wait job, three engine launches on Qwythos-9B.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
PY=${PY:-$HOME/Projects/mojo/baro-miner/.venv/bin/python}
GGUF=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}; REC=${REC:-8:100:30,20:50:120}
for f in .work/nx_persona .work/cvec; do [ -x $f ] || { echo "FAIL setup: $f missing (run tests/persona.sh)"; exit 1; }; done
O=$PWD/.work/persona-engine; rm -rf "$O"; mkdir -p "$O"
export PERSONA_DIR=$O/personas NX_CVEC=$PWD/.work/cvec NX_RLOG=$O/rlog
.work/nx_persona persona save "{\"name\":\"engine\",\"gguf\":\"$GGUF\",\"recipe\":\"$REC\"}" >/dev/null
.work/cvec "$GGUF" "$REC" "$O/direct" >/dev/null
PF=$(.work/nx_persona persona apply '{"name":"engine"}' | $PY -I -c 'import json,sys; print(json.load(sys.stdin)["steer_file"])')
IDS=$(gpu-wait run --room cpu --cores 2 --mem 2 --timeout 60 --label persona-tok -- bash -c "~/Models/llama.cpp/build/bin/llama-tokenize -m '$GGUF' --ids -p 'What is the capital of France?' 2>/dev/null | tail -1 | tr -d '[] '" | tail -1)
[ -n "$IDS" ] || { echo "FAIL tokenize"; exit 1; }
gpu-wait run --timeout 900 --vram 12 --label persona-engine -- scripts/persona-engine-run.sh "$O" "$IDS" "$PF" "$O/direct.f32" > "$O/run.log" 2>&1 || { tail -3 "$O/run.log"; echo "FAIL persona-engine: engine job"; exit 1; }
$PY -I tools/logit-diff.py "$O" | tee "$O/receipt.txt" || { echo "FAIL persona-engine: logit bars (docs/prereg-f1.md)"; exit 1; }
echo "PASS persona-engine: $(cat "$O/receipt.txt")"
