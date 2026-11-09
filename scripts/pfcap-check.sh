#!/usr/bin/env bash
# Batched-prefill capture (docs/prereg-pfcap.md): 2048 tokens, layer 16, batched against one-position capture of the same engine; rate from the engine's prefill_s.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2b}; ENGINE=${ENGINE:-$BARO/.work/engine-9b-pfcap}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
TOKJ=${TOKJ:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf/tokenizer.json}; PYB=${PYB:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python3}; L=${LAYER:-16}
O=$PWD/.work/pfcap-check; rm -rf "$O"; mkdir -p "$O/capb" "$O/cap1"
[ -x "$ENGINE" ] || { echo "FAIL setup: no engine at $ENGINE"; exit 1; }
"$PYB" -I -c "
from tokenizers import Tokenizer
t = open('$HOME/Projects/mojo/mojo-neuron/docs/plans/2026-11-09-neuron-v2.md').read()
ids = Tokenizer.from_file('$TOKJ').encode(t, add_special_tokens=False).ids[:2048]
assert len(ids) == 2048, len(ids)
open('$O/ids.txt', 'w').write(' '.join(map(str, ids)) + '\n'); print('ids', len(ids))"
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_SPEC=0 BARO_PACK='$PACK' BARO_PROMPT='$O/ids.txt' BARO_TMAX=4096 BARO_CAPTURE_LAYERS=$L BARO_CAPTURE_FIELDS=xyr
env BARO_CAPTURE='$O/capb' '$ENGINE' > '$O/batched.log' 2>&1
env BARO_STEER_SERVE=1 BARO_CAPTURE='$O/cap1' '$ENGINE' > '$O/single.log' 2>&1
EOS
chmod +x "$O/run.sh"; gpu-wait run --timeout 900 -- "$O/run.sh" || true
for d in capb cap1; do [ -s "$O/$d/L$L.f16" ] || { echo "FAIL run $d: $(grep -iE 'error|abort' "$O/${d/capb/batched}.log" "$O/${d/cap1/single}.log" 2>/dev/null | tail -n 1 | cut -c1-200)"; exit 1; }; done
grep -h "prefill chunk\|prefill_s" "$O/batched.log" "$O/single.log" | cut -c1-140
"$PYB" -I - "$O" $L <<'PYEOF'
import sys, re, numpy as np
O, L = sys.argv[1], sys.argv[2]
rate = {}
for n in ("batched", "single"):
    m = re.search(r"tokens: \d+\s+prefill_s: ([0-9.e+-]+)", open(f"{O}/{n}.log").read())
    rate[n] = 2048 / float(m[1])  # all 2048 prompt positions (batched chunk rows + the last position + replay rows)
cap = {d: np.fromfile(f"{O}/{d}/L{L}.f16", dtype=np.float16) for d in ("capb", "cap1")}
nrec = cap["capb"].size // (3 * 4096)
if cap["capb"].size != cap["cap1"].size or cap["capb"].size % (3 * 4096) or nrec < 2048: raise SystemExit(f"FAIL bar 1: sizes {cap['capb'].size} vs {cap['cap1'].size} ({nrec} records)")  # the engine also generates 63 tokens, whose positions are recorded too: compare the first 2048
tb, t1 = (np.fromfile(f"{O}/{d}/tokens.i32", dtype=np.int32) for d in ("capb", "cap1"))
if not np.array_equal(tb[:2048], t1[:2048]) or open(f"{O}/capb/req.i32", "rb").read() != open(f"{O}/cap1/req.i32", "rb").read(): raise SystemExit("FAIL bar 1: the prompt tokens or the request ids differ between the runs")
print(f"generated continuation: {int((tb[2048:] != t1[2048:]).sum())} of {len(tb) - 2048} fed tokens differ between the runs (greedy decoding after numerically different prefills)")
a = cap["capb"].reshape(nrec, 3, 4096)[:2048].astype(np.float64); b = cap["cap1"].reshape(nrec, 3, 4096)[:2048].astype(np.float64)
rel = lambda x, y: np.linalg.norm(x - y, axis=1) / np.maximum(np.linalg.norm(y, axis=1), 1e-12); cos = lambda x, y: (x * y).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12)
cr = cos(a[:, 2], b[:, 2]); rx, ry, rr = (np.median(rel(a[:, i], b[:, i])) for i in (0, 1, 2))
print(f"agreement over 2048 positions: r cosine min {cr.min():.5f}, median rel L2: x {rx:.4f}, y {ry:.4f}, r {rr:.4f}")
print(f"rate: batched {rate['batched']:.0f} tok/s, one position per step {rate['single']:.0f} tok/s (engine prefill_s, includes the file writes)")
if cr.min() < 0.999 or rr > 0.02 or ry > 0.05 or rx > 0.02: raise SystemExit("FAIL bar 2: agreement with the one-position capture below the prereg bars")
print("RATE", int(rate["batched"]))
PYEOF
echo "PASS pfcap-check: batched capture equals the one-position capture within the bars"
