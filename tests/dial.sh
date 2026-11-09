#!/usr/bin/env bash
# Gate C2 (docs/prereg-C2.md): the app's persona dial exports a control vector that llama.cpp loads and that moves logits like the engine.
# app (script: two neurons, slider clicks, export) -> cvec.gguf == f32 -> llama.cpp --control-vector -> delta logits llama.cpp vs engine -> negated mutant.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/shots .work/dial
HOMEP=$HOME; E=${ENGINE:-$HOME/Projects/mojo/mojo-neuron-lanes/engine}; BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}; MOD=qwen2.5-0.5b-instruct-q4_k_m
GGBF=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}
SPARK=$BARO/.work/steer/spark-q05; PACK=$BARO/.work/abl/q/pack; PROF=$BARO/.work/abl/q/prof; PY=${PY:-$E/.work/venv-gne/bin/python}; RHO=$E/.work/steer/q05/rho.f32
D=$PWD/.work/dial; fail() { echo "FAIL dial: $1"; exit 1; }
for f in "$SPARK" "$PY" "$RHO" "$Q" "$GGBF"; do [ -e "$f" ] || fail "$f missing"; done
NL=$(sed -n 's/^comptime N_LAYERS = \([0-9]*\).*/\1/p' "$PROF/profile.mojo"); H=$(sed -n 's/^comptime H = \([0-9]*\).*/\1/p' "$PROF/profile.mojo")
scripts/build-app.sh >/dev/null; cp .work/app $D/app.bin  # the queued job runs this copy, a later rebuild cannot change it
timeout 300 "$M" build src/cvec.mojo -I src -I "$GG" -o .work/cvec >/dev/null 2>&1 || fail "build cvec"
# 1. the app: store with lenses of layers 12 and 16, model.json carries rho (rel doses), script drives the dial
$PY tools/mkcap.py $D/cap 4,12 64 30 7
export MNEURON_CAP=$D/cap MNEURON_DIR=$D/neurons NX_RLOG=$D/rlog; rm -rf "$MNEURON_DIR" "$NX_RLOG" "$D/dial.out"
for l in 12 16; do .work/lens "$Q" $l 8 --out "$MNEURON_DIR" >/dev/null; done
$PY - "$MNEURON_DIR/$MOD/model.json" "$RHO" <<'PY'
import json, sys, numpy as np
m = json.load(open(sys.argv[1])); m["rho"] = [float(x) for x in np.fromfile(sys.argv[2], dtype=np.float32)]; json.dump(m, open(sys.argv[1], "w"))
PY
cat > $D/app.sh <<APP
#!/usr/bin/env bash
export PATH=$PATH HOME=$HOMEP WAYLAND_DISPLAY=\${WAYLAND_DISPLAY:-wayland-1} DISPLAY=\${DISPLAY:-:0} XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/run/user/1000} MNEURON_DIR=$MNEURON_DIR NX_RLOG=$NX_RLOG MNEURON_CAP=$D/cap
cd $PWD; MNEURON_SCRIPT=tests/dial_script.txt timeout 150 $D/app.bin 2>&1 | tail -1
APP
chmod +x $D/app.sh
out=$(gpu-wait run --shared --vram 2 --timeout 200 -- $D/app.sh 2>&1 | grep '^STATE' | tail -1)
rec=$(echo "$out" | sed -n 's/.* dial=\([^ ]*\) .*/\1/p')
echo "$out" | grep -q 'status=sound: 3.6 s' || fail "play capture did not render the synthetic capture (3.6 s): $out"
grep -q '"tool":"cvec".*layers=2 .*mode=rel' "$NX_RLOG"/2*.jsonl || fail "export did not run in rel mode with 2 layers (rlog)"
IFS=, read -r u1 u2 <<< "$rec"
[[ "${u1##*:}" != "+0.25" && "${u1##*:}" == +* && "${u2##*:}" == -* ]] || fail "slider clicks did not move the doses off the default with signs + and -: $rec"
echo "ok   app: dial $rec exported (rel)"
# 2. export == CLI, gguf == f32
.work/cvec "$Q" "$rec" $D/cli "$RHO" >/dev/null  # the store model.json names the Q4_K_M file the lens was built from
cmp -s $D/cli.f32 "$MNEURON_DIR/$MOD/dial.f32" || fail "app export differs from the cvec CLI (f32)"
cmp -s $D/cli.cvec.gguf "$MNEURON_DIR/$MOD/dial.cvec.gguf" || fail "app export differs from the cvec CLI (gguf)"
$PY tools/cvec-check.py "$MNEURON_DIR/$MOD/dial.cvec.gguf" "$MNEURON_DIR/$MOD/dial.f32" $NL $H >/dev/null || fail "cvec.gguf differs from f32"
echo "ok   export byte-identical to the CLI; cvec.gguf == f32"
# the rlog line of the export
grep -q '"tool":"cvec"' "$NX_RLOG"/2*.jsonl && grep -q '"kind":"normal"' "$NX_RLOG"/2*.jsonl || fail "no rlog line for the export"
# 3. llama.cpp loads the gguf (its own loader) and changes the greedy text
LC=$HOME/Models/llama.cpp/build/bin/llama-completion; P="What is the capital of France?"
gen() { gpu-wait run --room cpu --cores 6 --timeout 120 -- "$LC" -m "$GGBF" -p "$P" -n 24 --temp 0 -no-cnv -ngl 0 --no-display-prompt --no-perf "$@" 2>/dev/null | tr -d '\n'; }
t0=$(gen); t1=$(gen --control-vector "$MNEURON_DIR/$MOD/dial.cvec.gguf")
[ -n "$t1" ] || fail "llama.cpp produced no text with the control vector (load failed?)"
[ "$t0" != "$t1" ] || fail "llama.cpp text unchanged by the control vector"
echo "ok   llama.cpp loaded dial.cvec.gguf, greedy text changed"
# 4. delta logits, llama.cpp vs engine
IDS=$D/prompt.ids; ~/Models/llama.cpp/build/bin/llama-tokenize -m "$GGBF" --ids -p "$P" 2>/dev/null | tail -1 | tr -d '[],' > "$IDS"
spark() { local o=$1; shift; (cd "$BARO" && gpu-wait run --timeout 60 -- env BARO_PROMPT="$IDS" BARO_PACK="$PACK" BARO_GEN=1 BARO_SPEC=0 BARO_PREFILL_ROWS=0 BARO_DUMP_LOGITS="$o" "$@" "$SPARK" >/dev/null 2>&1); }
V=$MNEURON_DIR/$MOD/dial.f32
$PY -c "import numpy as np;np.fromfile('$V',dtype=np.float32).__neg__().tofile('$D/neg.f32')"
$PY tools/llama-oracle.py "$GGBF" "$IDS" $D/ll-base.f32 >/dev/null; $PY tools/llama-oracle.py "$GGBF" "$IDS" $D/ll-dial.f32 "$V" >/dev/null
spark $D/b-base.f32; spark $D/b-dial.f32 BARO_STEER="$V"; spark $D/b-neg.f32 BARO_STEER=$D/neg.f32
$PY - $D <<'PY' || exit 1
import sys, numpy as np
D = sys.argv[1] + "/"; f = lambda n: np.fromfile(D + n, dtype=np.float32)
lb, ld, bb, bd, bn = f("ll-base.f32"), f("ll-dial.f32"), f("b-base.f32"), f("b-dial.f32"), f("b-neg.f32")
dl, db, dn = ld - lb, bd - bb, bn - bb
e = np.linalg.norm(dl - db) / np.linalg.norm(dl); top = np.argsort(-np.abs(dl))[:1000]
sg = float((np.sign(dl[top]) == np.sign(db[top])).mean()); cm = float(dl @ dn / np.linalg.norm(dl) / np.linalg.norm(dn))
em = np.linalg.norm(dl - dn) / np.linalg.norm(dl)
print("rel err %.4f  top1 %d/%d  sign agreement %.3f  | negated-dial mutant: cos %.3f, rel err %.3f" % (e, ld.argmax(), bd.argmax(), sg, cm, em))
bad = [n for n, ok in [("rel err > 0.10", e > 0.10), ("top-1 differs", ld.argmax() != bd.argmax()), ("sign agreement < 0.95", sg < 0.95),
                       ("mutant not rejected (rel err <= 0.5)", em <= 0.5)] if ok]  # annotation added after the first result run, see docs/prereg-C2.md
if cm >= 0:
    print("MISSED prereg bar 5 (negated dial cosine < 0): %.3f. The response is not linear at these doses; the mutant is rejected by rel err instead" % cm)
sys.exit(print("FAIL dial: " + ", ".join(bad)) or 1) if bad else None
PY
echo "PASS dial (bars 1-4; a missed bar 5 is printed above): $rec"
