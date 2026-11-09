"""A5 comparison (docs/prereg-a5.md): engine expert trace (BARO_EXPERTS, ids + routing weights) against llama.cpp's router tensors from tools/moe-topk-oracle.
usage: moe-router-compare.py TRACE.txt ORACLE.txt NTOKENS"""
import sys, collections
trace, oracle, nt = sys.argv[1], sys.argv[2], int(sys.argv[3])
eng = {}  # (token, layer) -> (ids, gates)
k = None
for l in open(trace):
    if l.startswith("#"):
        for w in l.split():
            if w.startswith("topk="): k = int(w[5:])
        continue
    p = l.split()
    if len(p) < 2 + 2 * k: continue
    tk, ly = int(p[0]), int(p[1]); eng[(tk, ly)] = ([int(x) for x in p[2:2 + k]], [float(x) for x in p[2 + k:2 + 2 * k]])
ora = collections.defaultdict(dict)  # name -> (token, layer) -> values
for l in open(oracle):
    p = l.split()
    if len(p) < 4 or not p[0].startswith("ffn_moe"): continue
    ora[p[0]][(int(p[2]), int(p[1]))] = [float(x) for x in p[3:]] if p[0] != "ffn_moe_topk" else [int(x) for x in p[3:]]
layers = sorted({ly for (_, ly) in ora["ffn_moe_topk"]})
if not layers: raise SystemExit("FAIL oracle: no ffn_moe_topk lines")
miss = [(t, ly) for t in range(nt) for ly in layers if (t, ly) not in eng]
if miss: raise SystemExit(f"FAIL engine trace: no rows for {len(miss)} (position, layer) pairs, first {miss[:3]} (rows present: {len(eng)})")
bad = [(t, ly) for t in range(nt) for ly in layers if set(eng[(t, ly)][0]) != set(ora["ffn_moe_topk"][(t, ly)])]
if bad: raise SystemExit(f"FAIL ids: {len(bad)} of {nt * len(layers)} (position, layer) top-{k} sets differ, first {bad[:4]}")
shift = [(t, ly) for t in range(nt) for ly in layers[:-1] if set(eng[(t, ly)][0]) != set(ora["ffn_moe_topk"][(t, ly + 1)])]
if not shift: raise SystemExit("FAIL mutant: engine layer l equals llama layer l+1 everywhere")
res = {}
for name in ("ffn_moe_weights", "ffn_moe_weights_norm"):
    if not ora[name]: continue
    worst = 0.0
    for t in range(nt):
        for ly in layers:
            ids, g = eng[(t, ly)]; oid = ora["ffn_moe_topk"][(t, ly)]; ow = ora[name][(t, ly)]; ref = dict(zip(oid, ow))
            for i, x in zip(ids, g): worst = max(worst, abs(x - ref[i]) / max(abs(ref[i]), 1e-9))
    res[name] = worst
print(f"ids: all {nt * len(layers)} (position, layer) top-{k} sets equal llama.cpp ffn_moe_topk over {len(layers)} layers; layer-shift mutant differs at {len(shift)} pairs")
print("routing weights, worst relative error against llama.cpp: " + ", ".join(f"{n} {w:.3e}" for n, w in res.items()))
if not res or min(res.values()) > 1e-3: raise SystemExit(f"FAIL weights: no llama tensor within 1e-3 relative ({res})")
print(f"weights match {min(res, key=res.get)} within 1e-3")
