"""implants-write CAPDIR LAYER REC OUT [k=1] [tau=0.95] [gate=100] [vnorm=10] [seed=0] [key=capture|random]: test builder for scripts/implants-check.sh (the real builder is
src/causal/implants.mojo, G2). One set at LAYER with k implants: key = the `x` row of record REC of CAPDIR/L<layer>.f16 (or random), random unit value * vnorm. k=0 writes an empty set."""
import sys, json
import numpy as np
cap, layer, rec, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
o = dict(a.split("=") for a in sys.argv[5:]); k = int(o.get("k", 1)); tau = float(o.get("tau", 0.95)); gate = float(o.get("gate", 100)); vn = float(o.get("vnorm", 10))
rng = np.random.default_rng(int(o.get("seed", 0)))
m = json.load(open(f"{cap}/meta.json")); H, nf, F = m["h"], len(m["fields"]), m["fields"]
a = np.fromfile(f"{cap}/L{layer}.f16", dtype=np.float16).reshape(-1, nf, H)
keys = (np.stack([a[rec + i, F.index("x")].astype(np.float32) for i in range(k)]) if k else np.zeros((0, H), np.float32)) if o.get("key", "capture") == "capture" else rng.standard_normal((k, H)).astype(np.float32)
vals = rng.standard_normal((k, H)).astype(np.float32); vals *= vn / np.linalg.norm(vals, axis=1, keepdims=True)
with open(out, "wb") as f:
    f.write(b"NIMP0001"); f.write(np.array([H, 1], dtype=np.uint32).tobytes())
    f.write(np.array([layer, k], dtype=np.uint32).tobytes()); f.write(np.float32(tau).tobytes())
    f.write(keys.astype(np.float32).tobytes()); f.write(np.full(k, gate, dtype=np.float32).tobytes()); f.write(vals.tobytes())
print("wrote", out, "k", k)
