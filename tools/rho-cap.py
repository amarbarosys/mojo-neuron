"""rho-cap DIR LAYER: median residual L2 norm of capture layer LAYER (fields must hold r). Prints the number."""
import json, sys
import numpy as np

d, layer = sys.argv[1], int(sys.argv[2])
m = json.load(open(f"{d}/meta.json"))
f = m["fields"]
assert "r" in f, f"capture fields {f!r} hold no r"
x = np.fromfile(f"{d}/L{layer}.f16", dtype=np.float16).reshape(-1, len(f), m["h"])[:, f.index("r")].astype(np.float32)
print(float(np.median(np.linalg.norm(x, axis=1))))
