"""mkcap.py DIR LAYERS H [TOKENS] [SEED]: a deterministic synthetic capture in the docs/format-capture.md layout (fields xy, n_layers 24),
for sound and heat-strip gates that must not depend on a growing engine capture."""
import json, os, sys
import numpy as np
d, layers, h = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3])
n, seed = (int(sys.argv[4]) if len(sys.argv) > 4 else 30), (int(sys.argv[5]) if len(sys.argv) > 5 else 7)
os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(seed)
for l in layers:
    rng.normal(0, 1, (n, 2, h)).astype(np.float16).tofile(f"{d}/L{l}.f16")
np.arange(100, 100 + n, dtype=np.int32).tofile(f"{d}/tokens.i32")
np.zeros(n, dtype=np.int32).tofile(f"{d}/req.i32")
json.dump({"model": "synthetic", "dtype": "f16", "h": h, "n_layers": 24, "fields": "xy", "layers": layers}, open(f"{d}/meta.json", "w"))
