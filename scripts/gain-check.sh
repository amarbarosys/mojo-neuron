#!/usr/bin/env bash
# Gate for neuron gain ablation (L:N:k multiplies a neuron's activation; k = 0 removes it): baro serve `gain` request
# against the HF fp32 model with the same hook; delta logits relative error <= 0.10 and same top-1; a gain on the
# neighbour neuron must NOT match. The HF oracle runs on the CPU only (GPU hidden).
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; SPARK=${SPARK:-$BARO/.work/steer/spark-q05}; PACK=${PACK:-$BARO/.work/abl/q/pack}
HF=${HF:-$HOME/Models/qwen2.5-0.5b-instruct-hf}; PYHF=${PYHF:-$HOME/Projects/mojo/mojo-ablate/.work/oracle/bin/python}; PY=${PY:-.work/venv-gne/bin/python}
IDS=${IDS:-$HOME/Projects/mojo/mojo-ablate/tests/golden/qwen05-capital.ids}
O=$PWD/.work/gain-check; rm -rf "$O"; mkdir -p "$O"
P=$(tr -s ' \n' ',' < "$IDS" | sed 's/,$//')
baro_row() { # baro_row NAME GAIN|"" -> $O/b_NAME.f32, the raw logits of the first generated step
  mkdir -p "$O/$1"; local g=""; [ -n "$2" ] && g=",\"gain\":\"$2\""
  echo "{\"id\":1,\"prompt\":[$P],\"n\":1,\"top_logprobs\":1$g}" > "$O/$1.req"
  (cd "$BARO" && gpu-wait run --timeout 60 -- bash -c "env BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK' BARO_DUMP_LOGITS_DIR='$O/$1' '$SPARK' < '$O/$1.req' 2>&1") > "$O/$1.log" || true
  if grep -q '"error"' "$O/$1.log"; then echo "FAIL baro $1: $(grep '"error"' "$O/$1.log")"; exit 1; fi
  cp "$O/$1/row-0.bin" "$O/b_$1.f32"
}
hf_row() { CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=-1 $PYHF tools/gain-oracle.py "$HF" "$IDS" "$O/h_$1.f32" "${@:2}" >/dev/null; }
cmpq() { $PY - "$O" "$1" "$2" "$3" <<'PYEOF'
import sys, numpy as np
O, bn, hn, mode = sys.argv[1] + "/", sys.argv[2], sys.argv[3], sys.argv[4]
f = lambda n: np.fromfile(O + n + ".f32", dtype=np.float32)
b0, bt, h0, ht = f("b_base"), f("b_" + bn), f("h_base"), f("h_" + hn)
db, dh = bt - b0, ht - h0
e = np.linalg.norm(db - dh) / np.linalg.norm(dh)
ok = bt.argmax() == ht.argmax() and e <= 0.10
print("baro %s vs hf %s: rel err %.4f top1 %d/%d |dh| %.2f" % (bn, hn, e, bt.argmax(), ht.argmax(), np.linalg.norm(dh)))
sys.exit(0 if (ok if mode == "pass" else not ok) else 1)
PYEOF
}
baro_row base "0:0:1"; hf_row base   # a no-op gain keeps the base on the same token-by-token path as the gain requests
for g in "12:100:0" "12:100:4" "5:300:0,18:42:3"; do t=$(echo "$g" | tr ':,' '__')
  baro_row "$t" "$g"; hf_row "$t" "$g"
  cmpq "$t" "$t" pass || { echo "FAIL gain: baro and HF disagree on $g"; exit 1; }
  echo "ok   gain $g matches the HF fp32 hook"
done
baro_row mut "12:101:0"
cmpq mut 12_100_0 fail || { echo "FAIL mutant: neighbour neuron gain still matches"; exit 1; }
echo "ok   mutant (neuron + 1) rejected"
echo PASS gain-check
