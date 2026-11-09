"""Synthetic capture + fake steered captures for the lie_steer gate. Layer 3 carries a planted truth direction, layer 7 = layer 3 + noise (so a
steer at L3 propagates to L7 with gain 1). Usage: lie-steer-synth.py cap OUTDIR | steered CAPDIR PREPDIR OUTDIR MODE
MODE: ok (steer vector added at L3 and L7), none (steered rows equal the base), wrongsign (vector negated), applied (added at L3 only)."""
import json, os, sys
import numpy as np
h, T, n = 64, 3, 400
def cap(out):
    rng = np.random.default_rng(11); os.makedirs(out, exist_ok=True)
    d = rng.normal(size=h); d /= np.linalg.norm(d)
    y = np.tile([1, 0], n); r3, r7 = [], []
    for i in range(2 * n):
        for t in range(T):
            x = rng.normal(size=h) + (3.0 * d * (1 if y[i] else -1) if t == T - 1 else 0)
            r3.append(x); r7.append(x + 0.3 * rng.normal(size=h))
    np.asarray(r3, dtype=np.float16).tofile(f"{out}/L3.f16"); np.asarray(r7, dtype=np.float16).tofile(f"{out}/L7.f16")
    np.repeat(np.arange(1, 2 * n + 1), T).astype(np.int32).tofile(f"{out}/req.i32")
    (np.arange(2 * n * T) % 1000).astype(np.int32).tofile(f"{out}/tokens.i32")
    json.dump({"model": "synth", "dtype": "f16", "h": h, "n_layers": 8, "fields": "r", "layers": [3, 7]}, open(f"{out}/meta.json", "w"))
    with open(f"{out}/labels.tsv", "w") as f:
        for i in range(2 * n): f.write(f"{y[i]}\t{i // 2}\tstatement {i}\n")
def steered(capdir, prep, out, mode):
    os.makedirs(out, exist_ok=True); P = json.load(open(f"{prep}/prep.json")); sel = P["sel"]
    req = np.fromfile(f"{capdir}/req.i32", dtype=np.int32)
    last = [i for i in range(len(req)) if i + 1 == len(req) or req[i + 1] != req[i]]
    L = {l: np.fromfile(f"{capdir}/L{l}.f16", dtype=np.float16).reshape(-1, h).astype(np.float32) for l in (3, 7)}
    names = ["base", "plus", "minus", "rand0", "rand1", "rand2"]
    vec = {c: (np.zeros(h, np.float32) if c == "base" else np.fromfile(f"{prep}/{c}.f32", dtype=np.float32)) for c in names}
    r3, r7 = [], []
    for c in names:
        v = vec[c] * (-1 if mode == "wrongsign" else 1) * (0 if mode == "none" else 1)
        for j in range(len(sel)):
            b = last[sel[j]]; r3.append(L[3][b] + v); r7.append(L[7][b] + (0 if mode == "applied" else v))
    np.asarray(r3, dtype=np.float16).tofile(f"{out}/L3.f16"); np.asarray(r7, dtype=np.float16).tofile(f"{out}/L7.f16")
    np.arange(1, len(r3) + 1).astype(np.int32).tofile(f"{out}/req.i32"); np.zeros(len(r3), np.int32).tofile(f"{out}/tokens.i32")
    json.dump({"model": "synth", "dtype": "f16", "h": h, "n_layers": 8, "fields": "r", "layers": [3, 7]}, open(f"{out}/meta.json", "w"))
if sys.argv[1] == "cap": cap(sys.argv[2])
else: steered(*sys.argv[2:6])
