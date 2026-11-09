#!/usr/bin/env bash
# Gate for BARO_STEER in engine.mojo (qwen35 family), tolerances pre-registered in docs/prereg-engine-steer.md.
#   Gate A (decides): the engine's own residual dump, steered vs unsteered: equal below the first steered layer, differing by
#                     exactly the vector at it, non-zero above; mutants 0.6 v, 0.95 v, v shifted one layer, zero vector must fail.
#   Gate B: agreement with llama.cpp's control vector, printed as a diagnostic (v1 pre-registered and failed; v2 post hoc).
# Needs: ENGINE (built from the baro `ablate` worktree), PACK, GGUF (the pack's source file), LAYERS, H.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; TAG=${TAG:-qwythos9}
ENGINE=${ENGINE:-$BARO/.work/steer/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
LAYERS=${LAYERS:-32}; H=${H:-4096}
PY=${PY:-.work/venv-gne/bin/python}; M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}
S=$PWD/.work/steer/$TAG; mkdir -p "$S/cal"
if [ -z "${REC:-}" ]; then for r in ${RECS:-"8:100:30,20:50:120" "24:100:150"}; do REC=$r "$0"; done; echo "PASS engine-steer-check $TAG"; exit 0; fi
fail() { echo "FAIL $1: $2"; exit 1; }
[ -x "$ENGINE" ] || fail setup "no engine $ENGINE"
[ -x .work/cvec ] || timeout 300 "$M" build src/cvec.mojo -I src -I "${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}" -o .work/cvec >/dev/null 2>&1 || fail build cvec
TOK=~/Models/llama.cpp/build/bin/llama-tokenize
ids_of() { [ -f "$2" ] || $TOK -m "$GG" --ids -p "$1" 2>/dev/null | tail -1 | tr -d '[],' > "$2"; }
ids_of "What is the capital of France?" "$S/prompt.ids"
ids_json() { tr -s ' \n' ',' < "$1" | sed 's/,$//'; }
ENVBASE="BARO_SPEC=0 BARO_MEGA=${MEGA:-0} ${EXTRA_ENV:-} BARO_PACK='$PACK'"
engine_row() { # engine_row DIR IDS [STEER.f32] -> DIR/row-0.bin (serve mode, raw logits of the first generated step)
  mkdir -p "$1"; local st=""; [ -n "${3:-}" ] && st="BARO_STEER=$3"
  echo "{\"id\":1,\"prompt\":[$(ids_json "$2")],\"n\":1,\"top_logprobs\":1}" > "$1.req"
  gpu-wait run --timeout 300 -- bash -c "env BARO_SERVE=1 $ENVBASE BARO_DUMP_LOGITS_DIR='$1' $st '$ENGINE' < '$1.req' 2>&1" > "$1.log" || true
  [ -f "$1/row-0.bin" ] || fail "engine row" "$(grep -E 'rror' "$1.log" | tail -2 | cut -c1-200)"
}
engine_dump() { # engine_dump NAME [STEER.f32] -> $S/NAME.dump (one-shot, residual after every layer at the first generated step)
  local st=""; [ -n "${2:-}" ] && st="BARO_STEER=$2"
  gpu-wait run --timeout 300 -- bash -c "env BARO_PROMPT='$S/prompt.ids' BARO_GEN=2 $ENVBASE BARO_DUMP='$S/$1.dump' $st '$ENGINE' 2>&1" > "$S/$1.log" || true
  [ -s "$S/$1.dump" ] || fail "engine dump" "$(grep -E 'rror' "$S/$1.log" | tail -2 | cut -c1-200)"
}
# ---- calibration (once per pack): unsteered engine vs llama.cpp on four other prompts
if [ ! -f "$S/cal.json" ]; then
  i=0; tasks="["; for p in "The weather today is" "Write a poem about the sea." "def fib(n):" "Bonjour, comment"; do i=$((i+1))
    ids_of "$p" "$S/cal/p$i.ids"; engine_row "$S/cal/e$i" "$S/cal/p$i.ids"; cp "$S/cal/e$i/row-0.bin" "$S/cal/e$i.f32"
    tasks="$tasks{\"ids\":\"$S/cal/p$i.ids\",\"out\":\"$S/cal/l$i.f32\"},"; done
  echo "${tasks%,}]" > "$S/cal/tasks.json"; $PY tools/llama-batch.py "$GG" "$S/cal/tasks.json" >/dev/null
  $PY - "$S" <<'PYEOF'
import sys, json, numpy as np
S = sys.argv[1]; ns, n99, rel = [], [], []
for i in range(1, 5):
    e = np.fromfile(f"{S}/cal/e{i}.f32", dtype=np.float32); l = np.fromfile(f"{S}/cal/l{i}.f32", dtype=np.float32)
    ns.append(float(np.linalg.norm(e - l))); n99.append(float(np.percentile(np.abs(e - l), 99))); rel.append(ns[-1] / float(np.linalg.norm(l)))
json.dump({"N": float(np.median(ns)), "n99": float(np.median(n99)), "rel": float(np.median(rel))}, open(f"{S}/cal.json", "w"))
print("calibration: N %.1f n99 %.2f rel %.4f" % (np.median(ns), np.median(n99), np.median(rel)))
PYEOF
fi
# ---- vectors
.work/cvec "$GG" "$REC" "$S/v" >/dev/null
$PY - "$S" "$LAYERS" "$H" <<'PYEOF'
import sys, numpy as np
S, nl, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
v = np.fromfile(f"{S}/v.f32", dtype=np.float32); assert v.size == nl * h, (v.size, nl, h)
np.roll(v.reshape(nl, h), 1, axis=0).tofile(f"{S}/vs.f32"); (v * np.float32(0.6)).tofile(f"{S}/v60.f32"); (v * np.float32(0.95)).tofile(f"{S}/v95.f32")
np.zeros_like(v).tofile(f"{S}/vz.f32")
PYEOF
# ---- Gate A
engine_dump base; engine_dump steer "$S/v.f32"; engine_dump m60 "$S/v60.f32"; engine_dump m95 "$S/v95.f32"; engine_dump mshift "$S/vs.f32"; engine_dump mzero "$S/vz.f32"
$PY - "$S" "$LAYERS" "$H" <<'PYEOF'
import sys, numpy as np
S, nl, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
v = np.fromfile(f"{S}/v.f32", dtype=np.float32).reshape(nl, h)
def res(name):  # residual after every layer at the first generated step
    d = np.fromfile(f"{S}/{name}.dump", dtype=np.float32)[:2 * nl * h].reshape(nl, 2, h)
    return d[:, 1, :]
base = res("base")
Lf = int(np.flatnonzero(np.abs(v).sum(axis=1))[0])
def gate_a(name):
    x = res(name); why = []
    if not np.array_equal(x[:Lf], base[:Lf]): why.append("a layer below %d differs from the unsteered run" % Lf)
    d = x[Lf] - base[Lf]
    e = np.linalg.norm(d - v[Lf]); tol = 1e-6 * np.linalg.norm(base[Lf]) + 1e-4 * np.linalg.norm(v[Lf])
    if e > tol: why.append("layer %d: |delta - v| = %.3g > %.3g (|v| %.3g)" % (Lf, e, tol, np.linalg.norm(v[Lf])))
    above = [np.linalg.norm(x[l] - base[l]) for l in range(Lf + 1, nl)]
    if not all(np.isfinite(a) and a > 0 for a in above): why.append("a layer above %d is unchanged or non-finite" % Lf)
    return why
bad = gate_a("steer")
if bad: raise SystemExit("FAIL gate A: " + "; ".join(bad))
for m in ("m60", "m95", "mshift", "mzero"):
    if not gate_a(m): raise SystemExit("FAIL gate A mutant %s: not rejected" % m)
print("gate A ok: below layer %d identical, |delta - v| at layer %d under %.1e of |v|; mutants 0.6v, 0.95v, shifted, zero rejected" % (Lf, Lf, 1e-4))
PYEOF
# ---- Gate B: diagnostic only. v1 (pre-registered) failed on all three readings; v2 was written after them, so it is post hoc and does not decide.
engine_row "$S/rb" "$S/prompt.ids"; engine_row "$S/rs" "$S/prompt.ids" "$S/v.f32"; engine_row "$S/rsh" "$S/prompt.ids" "$S/vs.f32"
tasks="[{\"ids\":\"$S/prompt.ids\",\"out\":\"$S/ll-base.f32\"},{\"ids\":\"$S/prompt.ids\",\"out\":\"$S/ll-steer.f32\",\"vec\":\"$S/v.f32\"}"
for i in 1 2 3 4; do engine_row "$S/cal/es$i" "$S/cal/p$i.ids" "$S/v.f32"; tasks="$tasks,{\"ids\":\"$S/cal/p$i.ids\",\"out\":\"$S/cal/ls$i.f32\",\"vec\":\"$S/v.f32\"}"; done
echo "$tasks]" > "$S/tasks.json"; $PY tools/llama-batch.py "$GG" "$S/tasks.json" >/dev/null
$PY - "$S" <<'PYEOF'
import sys, json, numpy as np
S = sys.argv[1]; cal = json.load(open(f"{S}/cal.json"))
f = lambda p: np.fromfile(p, dtype=np.float32)
lb, ls = f(f"{S}/ll-base.f32"), f(f"{S}/ll-steer.f32"); eb, es, esh = f(f"{S}/rb/row-0.bin"), f(f"{S}/rs/row-0.bin"), f(f"{S}/rsh/row-0.bin")
dl = ls - lb
err = lambda e: float(np.linalg.norm((e - eb) - dl)); cos = lambda e: float(((e - eb) @ dl) / np.linalg.norm(e - eb) / np.linalg.norm(dl))
NB = float(np.median([np.linalg.norm(f(f"{S}/cal/e{i}.f32") - f(f"{S}/cal/l{i}.f32")) for i in range(1, 5)]))
NS = float(np.median([np.linalg.norm(f(f"{S}/cal/es{i}/row-0.bin") - f(f"{S}/cal/ls{i}.f32")) for i in range(1, 5)]))
N, n99 = cal["N"], cal["n99"]
gap = float(ls.max() - ls[es.argmax()]); top1 = es.argmax() == ls.argmax() or gap <= n99
r = err(es) / np.linalg.norm(dl); bound = NS + NB
print("gate B v1 (diagnostic): error %.2f N (bound 1.6 N)" % (err(es) / N))
print("gate B v2: error %.1f vs NS + NB = %.1f + %.1f = %.1f, cos %.4f, top-1 %d/%d (llama gap %.2f, n99 %.2f), |dl| %.1f, shifted cos %.4f error %.1f" % (err(es), NS, NB, bound, cos(es), es.argmax(), ls.argmax(), gap, n99, np.linalg.norm(dl), cos(esh), err(esh)))
conds = {"error <= NS + NB": err(es) <= bound, "cos >= 0.95": cos(es) >= 0.95, "top-1 (equal or inside n99)": top1,
         "shifted mutant cos 0.05 lower or error > NS + NB": cos(esh) <= cos(es) - 0.05 or err(esh) > bound}
if cal["rel"] <= 0.05: conds["exact pack: rel err <= 0.10 and same top-1"] = r <= 0.10 and es.argmax() == ls.argmax()
bad = [k for k, ok in conds.items() if not ok]
print("gate B v2 (post hoc, diagnostic, not deciding): " + ("agrees" if not bad else "would FAIL: " + "; ".join(bad)))
PYEOF
echo "PASS engine-steer-check $TAG $REC (Gate A decides)"
