"""replay-seed DIR: a capture in format v1 (docs/format-capture.md): 3 requests of 5, 7, 4 tokens, layers 3 and 5, h=8, fields xyr, seeded f16."""
import json, os, sys
import numpy as np
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(7); n = [5, 7, 4]; T = sum(n); h = 8
json.dump({"model": "seed", "dtype": "f16", "h": h, "n_layers": 8, "fields": "xyr", "layers": [3, 5]}, open(f"{d}/meta.json", "w"))
for l in (3, 5): (rng.standard_normal((T, 3, h)) * 3).astype(np.float16).tofile(f"{d}/L{l}.f16")
rng.integers(0, 1000, T).astype(np.int32).tofile(f"{d}/tokens.i32")
np.concatenate([np.full(c, i, dtype=np.int32) for i, c in enumerate(n)]).tofile(f"{d}/req.i32")
