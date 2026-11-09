#!/usr/bin/env bash
# Gate for steps 2 and 3: BARO_STEER in mojo-baro spark, and the cvec writer, against llama.cpp.
#   zero vector -> byte-identical logits; two recipes -> delta logits match llama.cpp's control vector
#   (relative error <= 0.10, same top-1); a one-layer-shifted mutant must fail; the .cvec.gguf holds the vector.
# Needs: SPARK (binary built from the baro `ablate` worktree with the 0.5B profile), PACK (its pack dir).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/steer
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}
TAG=${TAG:-q05}   # model tag: spark binary .work/steer/spark-$TAG in the baro worktree, work files in .work/steer/$TAG
PROF=${PROF:-$BARO/.work/abl/q/prof}   # profile dir of the pack's model (H, N_LAYERS)
SPARK=${SPARK:-$BARO/.work/steer/spark-$TAG}; PACK=${PACK:-$BARO/.work/abl/q/pack}
GG=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}
PY=${PY:-.work/venv-gne/bin/python}; M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}
NL=$(sed -n 's/^comptime N_LAYERS = \([0-9]*\).*/\1/p' "$PROF/profile.mojo"); H=$(sed -n 's/^comptime H = \([0-9]*\).*/\1/p' "$PROF/profile.mojo")
S=$PWD/.work/steer/$TAG; mkdir -p "$S"
IDS=${IDS:-$S/prompt.ids}
[ -f "$IDS" ] || ~/Models/llama.cpp/build/bin/llama-tokenize -m "$GG" --ids -p "What is the capital of France?" 2>/dev/null | tail -1 | tr -d '[],' > "$IDS"
fail() { echo "FAIL $1: $2"; exit 1; }
spark() { local out=$1; shift   # spark OUT.f32 GEN [env...]
  (cd "$BARO" && gpu-wait run --timeout 60 -- env BARO_PROMPT="$IDS" BARO_PACK="$PACK" BARO_GEN="$1" BARO_SPEC=0 \
     BARO_PREFILL_ROWS=${PF:-0} BARO_DUMP_LOGITS="$out" "${@:2}" "$SPARK" 2>&1) | grep -E "^generated|tok/s_gen|rror|Unhandled" || true; }
[ -x "$SPARK" ] || fail setup "no spark binary $SPARK"
cap() { systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- timeout 300 "$@"; }
cap "$M" build src/cvec.mojo -I src -I "${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}" -o .work/cvec >/dev/null 2>&1 || fail build cvec
# rho from the model's own residuals
rm -f "$S/layers.bin"; spark "$S/x.bin" 1 BARO_DUMP_LAYERS="$S/layers.bin" >/dev/null
$PY tools/rho.py $H "$S/rho.f32" "$S/layers.bin" >/dev/null
# 1. zero vector
head -c $((NL * H * 4)) /dev/zero > "$S/zero.f32"
spark "$S/b-base.f32" 1 >/dev/null; spark "$S/b-zero.f32" 1 BARO_STEER="$S/zero.f32" >/dev/null
cmp -s "$S/b-base.f32" "$S/b-zero.f32" || fail zero "logits differ with an all-zero steering vector"
echo "ok   zero vector: logits byte-identical"
$PY tools/llama-oracle.py "$GG" "$IDS" "$S/ll-base.f32" >/dev/null
# 2. recipes vs llama.cpp
cmpq() { $PY - "$S" "$1" "$2" "${3:-}" <<'PYEOF'
import sys, numpy as np
S, t, mode, sfx = sys.argv[1] + "/", sys.argv[2], sys.argv[3], sys.argv[4]
f = lambda n: np.fromfile(S + n, dtype=np.float32)
lb, lt, bb, bt = f("ll-base.f32"), f("ll-%s.f32" % t), f("b-base%s.f32" % sfx), f("b-%s%s.f32" % (t, sfx))
dl, db = lt - lb, bt - bb
e = np.linalg.norm(dl - db) / np.linalg.norm(dl)
ok = lt.argmax() == bt.argmax() and e <= 0.10
print("%s rel err %.4f cos %.5f top1 %d/%d" % (t, e, dl @ db / np.linalg.norm(dl) / np.linalg.norm(db), lt.argmax(), bt.argmax()))
sys.exit(0 if (ok if mode == "pass" else not ok) else 1)
PYEOF
}
for r in "t1:$((NL / 2)):100:+1.0" "t2:$((NL / 4)):50:+0.3,$((NL * 5 / 6)):7:-0.5,3:11:+0.4"; do t=${r%%:*}; rec=${r#*:}
  .work/cvec "$GG" "$rec" "$S/$t" "$S/rho.f32" >/dev/null
  $PY tools/cvec-check.py "$S/$t.cvec.gguf" "$S/$t.f32" $NL $H >/dev/null || fail cvec "gguf differs from f32 for $rec"
  $PY tools/llama-oracle.py "$GG" "$IDS" "$S/ll-$t.f32" "$S/$t.f32" >/dev/null
  spark "$S/b-$t.f32" 1 BARO_STEER="$S/$t.f32" >/dev/null
  cmpq $t pass || fail steer "baro and llama.cpp disagree on recipe $rec"
  echo "ok   recipe $rec matches llama.cpp --control-vector; cvec.gguf == f32"
  # 3. mutant: the same vector one layer up must NOT match
  $PY -c "import numpy as np;np.roll(np.fromfile('$S/$t.f32',dtype=np.float32).reshape($NL,-1),1,axis=0).tofile('$S/${t}s.f32')"
  spark "$S/b-${t}s.f32" 1 BARO_STEER="$S/${t}s.f32" >/dev/null; cp "$S/ll-$t.f32" "$S/ll-${t}s.f32"
  cmpq ${t}s fail || fail mutant "a layer-shifted vector still matches llama.cpp"
  echo "ok   mutant (layer + 1) rejected"
done
# 4. the batched prefill path (prompt rows through WMMA GEMMs) steers too: same delta as token-by-token
t=t1; PF=1 spark "$S/b-base-pf.f32" 1 >/dev/null; PF=1 spark "$S/b-t1-pf.f32" 1 BARO_STEER="$S/t1.f32" >/dev/null
cmpq t1 pass -pf || fail steer-prefill "batched prefill rows disagree with llama.cpp on recipe t1"
echo "ok   batched prefill rows steer like token-by-token"
echo "PASS steer-check $TAG"
