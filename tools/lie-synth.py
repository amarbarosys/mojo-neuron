"""Synthetic capture for the lie_fit gate: layer 3 carries a planted truth direction, layer 7 is pure noise. Usage: lie-synth.py OUTDIR [N pairs]"""
import json, os, sys
import numpy as np
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
h, T = 64, 3
rng = np.random.default_rng(7)
os.makedirs(out, exist_ok=True)
d = rng.normal(size=h); d /= np.linalg.norm(d)
y = np.tile([1, 0], n)                                 # request 2p = true statement of pair p, 2p+1 = its false twin
recs = {3: [], 7: []}
for i in range(2 * n):
    for t in range(T):
        base3 = rng.normal(size=h) * 1.0 + (3.0 * d * (1 if y[i] else -1) if t == T - 1 else 0)
        recs[3].append(base3); recs[7].append(rng.normal(size=h))
for l in (3, 7):
    np.asarray(recs[l], dtype=np.float16).tofile(f"{out}/L{l}.f16")
np.repeat(np.arange(1, 2 * n + 1), T).astype(np.int32).tofile(f"{out}/req.i32")
np.zeros(2 * n * T, dtype=np.int32).tofile(f"{out}/tokens.i32")
json.dump({"model": "synth", "dtype": "f16", "h": h, "n_layers": 8, "fields": "r", "layers": [3, 7]}, open(f"{out}/meta.json", "w"))
with open(f"{out}/labels.tsv", "w") as f:
    for i in range(2 * n):
        f.write(f"{y[i]}\t{i // 2}\tstatement {i}\n")
np.save(f"{out}/dir.npy", d)
