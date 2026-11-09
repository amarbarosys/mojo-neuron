#!/usr/bin/env bash
# Gate (boss review item 6): `sweep mode=slens req=1` against engine.mojo writes a steer.bin whose up AND down records and KL equal
# numpy computed from the engine's own raw logits rows (base and BARO_STEER), for two neurons, plus the neighbour mutant.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; TAG=${TAG:-qwythos9}
ENGINE=${ENGINE:-$BARO/.work/steer/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
LAYERS=${LAYERS:-32}; LAYER=${LAYER:-16}; DOSE=${DOSE:-60}; PY=${PY:-.work/venv-gne/bin/python}
O=$PWD/.work/engine-slens-sweep-$TAG; rm -rf "$O"; mkdir -p "$O"
$PY -c "import numpy as np;np.ones($LAYERS,dtype=np.float32).tofile('$O/rho1.f32')"
printf 'What is the capital of France?\n' > "$O/prompts.txt"
scripts/build-sweep.sh
.work/sweep "$GG" "$O" mode=slens req=1 layer=$LAYER neurons=100,101 doses=$DOSE prompts="$O/prompts.txt" k=8 rho="$O/rho1.f32" spark="$ENGINE" pack="$PACK" baro="$BARO" env="BARO_MEGA=1 ${EXTRA_ENV:-}" >/dev/null
# oracle rows from the same engine: base and one steered run per neuron (BARO_STEER one-shot, same vector as the bank row)
P=$(python3 -c "import json;print(','.join(map(str,json.loads(open('$O/.req.jsonl').readline())['prompt'])))")
rows() { mkdir -p "$O/$1"; echo "{\"id\":1,\"prompt\":[$P],\"n\":1,\"top_logprobs\":1}" > "$O/$1.req"
  gpu-wait run --timeout 300 -- bash -c "env BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_MEGA=1 ${EXTRA_ENV:-} BARO_PACK='$PACK' BARO_DUMP_LOGITS_DIR='$O/$1' ${2:+BARO_STEER=$2} '$ENGINE' < '$O/$1.req' 2>&1" > "$O/$1.log" || true
  [ -f "$O/$1/row-0.bin" ] || { echo "FAIL rows $1"; exit 1; }; }
rows base
# the oracle's steer file is the bank row times the scale in float32, bit for bit what the engine builds from the request: the engine
# rounds activations to bf16, so a vector that differs in the last float32 bit (e.g. one recomputed in f64 by cvec) moves logits by ~1e-3
for i in 0 1; do n=$((100 + i)); $PY -c "
import numpy as np
b = np.fromfile('$O/$LAYER.bank.f32', dtype=np.float32).reshape(-1, 4096)[$i]
v = np.zeros((int('$LAYERS'), 4096), dtype=np.float32); v[$LAYER] = b * np.float32($DOSE); v.tofile('$O/v$n.f32')"; rows "n$n" "$O/v$n.f32"; done
$PY - "$O" $LAYER <<'PYEOF'
import sys, struct, numpy as np
O, layer = sys.argv[1], sys.argv[2]
b = open(f"{O}/{layer}.steer.bin", "rb").read()
magic, ver, ly, ffn, k, dose = struct.unpack("<4sIIIIf", b[:24]); rec = 16 * k
klf = np.fromfile(f"{O}/{layer}.steer.kl.f32", dtype=np.float32)
f = lambda n: np.fromfile(f"{O}/{n}/row-0.bin", dtype=np.float32).astype(np.float64)
def lsm(x): m = x.max(); return x - m - np.log(np.exp(x - m).sum())
base = lsm(f("base"))
def rec_of(n):
    o = 24 + n * rec; g = lambda q, t: struct.unpack(f"<{k}{t}", b[o + q * 4 * k:o + (q + 1) * 4 * k]); return g(0, "i"), g(1, "f"), g(2, "i"), g(3, "f")
def oracle(n):
    s = lsm(f(f"n{n}")); d = s - base
    return d, float(np.sum(np.exp(base) * (base - s)))
def match(n, d, kl):
    ui, uv, di, dv = rec_of(n); bad = []
    for ids, vs, order, name in ((ui, uv, np.argsort(-d)[:k], "up"), (di, dv, np.argsort(d)[:k], "down")):
        if set(ids) != set(int(i) for i in order): bad.append(f"{name} ids differ")
        else:
            for i, v in zip(ids, vs):
                if abs(v - d[i]) > 1e-3: bad.append(f"{name} {i}: {v} vs {d[i]}")
    if abs(klf[n] - kl) > 1e-3 * max(1.0, kl): bad.append(f"kl {klf[n]} vs {kl}")
    return bad
for n in (100, 101):
    d, kl = oracle(n); bad = match(n, d, kl)
    if bad: raise SystemExit(f"FAIL neuron {n}: " + "; ".join(bad[:3]))
    other = 101 if n == 100 else 100; d2, kl2 = oracle(other)
    if not match(n, d2, kl2): raise SystemExit(f"FAIL mutant: neuron {n}'s record matches neuron {other}'s oracle")
print("PASS: up, down and KL of neurons 100 and 101 equal the engine's own rows (KL %.4f, %.4f); neighbour mutant rejected" % (klf[100], klf[101]))
PYEOF
echo "PASS engine-slens-sweep-check $TAG"
