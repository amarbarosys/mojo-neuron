#!/usr/bin/env bash
# Gate (boss review item 6): the steer lens on engine.mojo (per request: slens_ctx / slens_k / slens_flags next to steer_file).
# Same-numerics oracle: the engine's own raw logits rows (BARO_DUMP_LOGITS_DIR) unsteered and under BARO_STEER, top-8 up/down of the
# log-prob change and KL computed in numpy f64; the in-engine line must equal them (ids, values 1e-3, KL 1e-3 relative).
# A line against the shifted vector's oracle must not match.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}; TAG=${TAG:-qwythos9}
ENGINE=${ENGINE:-$BARO/.work/steer/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
S=$PWD/.work/steer/$TAG; PY=${PY:-.work/venv-gne/bin/python}; O=$S/slens; rm -rf "$O"; mkdir -p "$O"
[ -f "$S/v.f32" ] && [ -f "$S/vs.f32" ] && [ -f "$S/prompt.ids" ] || { echo "FAIL setup: run engine-steer-check.sh first (v.f32, vs.f32, prompt.ids)"; exit 1; }
P=$(tr -s ' \n' ',' < "$S/prompt.ids" | sed 's/,$//')
ENVB="BARO_SPEC=0 BARO_MEGA=1 ${EXTRA_ENV:-} BARO_PACK='$PACK'"
{ echo "{\"id\":1,\"prompt\":[$P],\"n\":1,\"slens_ctx\":0,\"slens_k\":0}"
  echo "{\"id\":2,\"prompt\":[$P],\"n\":1,\"slens_ctx\":0,\"slens_k\":8,\"steer_file\":\"$S/v.f32\"}"
  echo "{\"id\":3,\"prompt\":[$P],\"n\":1,\"slens_ctx\":0,\"slens_k\":8,\"steer_file\":\"$S/vs.f32\"}"; } > "$O/a.req"
gpu-wait run --timeout 300 -- bash -c "env BARO_SERVE=1 BARO_STEER_SERVE=1 $ENVB '$ENGINE' < '$O/a.req' 2>&1" > "$O/a.out" || true
rows() { # rows DIR [STEER] : raw logits of the first generated step
  mkdir -p "$O/$1"; echo "{\"id\":1,\"prompt\":[$P],\"n\":1,\"top_logprobs\":1}" > "$O/$1.req"
  gpu-wait run --timeout 300 -- bash -c "env BARO_SERVE=1 BARO_STEER_SERVE=1 $ENVB BARO_DUMP_LOGITS_DIR='$O/$1' ${2:+BARO_STEER=$2} '$ENGINE' < '$O/$1.req' 2>&1" > "$O/$1.log" || true
  [ -f "$O/$1/row-0.bin" ] || { echo "FAIL rows $1: $(grep -E 'rror' "$O/$1.log" | tail -1 | cut -c1-160)"; exit 1; }
}
rows base; rows steer "$S/v.f32"; rows shift "$S/vs.f32"
$PY - "$O" <<'PYEOF'
import sys, json, numpy as np
O = sys.argv[1]
lines = [json.loads(l) for l in open(f"{O}/a.out") if l.startswith('{"id":') and '"slens"' in l]
if len(lines) != 2: raise SystemExit(f"FAIL slens: expected 2 slens lines from the engine, got {len(lines)}")
f = lambda n: np.fromfile(f"{O}/{n}/row-0.bin", dtype=np.float32).astype(np.float64)
def lsm(x): m = x.max(); return x - m - np.log(np.exp(x - m).sum())
b = lsm(f("base"))
def oracle(n):
    s = lsm(f(n)); d = s - b
    up = np.argsort(-d)[:8]; dn = np.argsort(d)[:8]
    return {"up": [(int(i), float(d[i])) for i in up], "dn": [(int(i), float(d[i])) for i in dn], "kl": float(np.sum(np.exp(b) * (b - s)))}
def same(line, o):
    e = line["slens"]; bad = []
    for key in ("up", "dn"):
        ids = [i for i, _ in e[key]]; oid = [i for i, _ in o[key]]
        if set(ids) != set(oid): bad.append(f"{key} ids differ")
        else:
            for i, v in e[key]:
                if abs(v - dict(o[key])[i]) > 1e-3: bad.append(f"{key} value {i}: {v} vs {dict(o[key])[i]}")
    if abs(e["kl"] - o["kl"]) > 1e-3 * max(1.0, o["kl"]): bad.append(f"kl {e['kl']} vs {o['kl']}")
    return bad
bad = same(lines[0], oracle("steer"))
if bad: raise SystemExit("FAIL slens: " + "; ".join(bad[:4]))
if not same(lines[0], oracle("shift")): raise SystemExit("FAIL mutant: the engine's line matches the shifted-vector oracle")
if same(lines[1], oracle("shift")): raise SystemExit("FAIL slens: the shifted-vector request does not match its own oracle")
print("PASS engine-slens-check: up/down top-8 and KL equal the engine's own rows (KL %.4f); shifted-vector mutant rejected" % lines[0]["slens"]["kl"])
PYEOF
