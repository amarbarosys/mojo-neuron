"""replay-oracle NX CAPDIR REQ ID LAYER: for every k of request REQ, `nx replay frame` equals record k of the ORIGINAL files (token and every f16 value,
exactly); exit 1 with FAIL on the first difference. Also checks the replay holds only that request's tokens."""
import json, subprocess, sys
import numpy as np
nx, cap, req, rid, layer = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
m = json.load(open(f"{cap}/meta.json")); h, nf = m["h"], len(m["fields"])
r = np.fromfile(f"{cap}/req.i32", dtype=np.int32); tok = np.fromfile(f"{cap}/tokens.i32", dtype=np.int32)
a = np.fromfile(f"{cap}/L{layer}.f16", dtype=np.float16).reshape(-1, nf, h)
idx = np.nonzero(r == req)[0]
for k, i in enumerate(idx):
    o = json.loads(subprocess.run([nx, "replay", "frame", json.dumps({"id": rid, "k": k, "layer": layer})], check=True, capture_output=True, text=True).stdout)
    got = np.array(o["rows"], dtype=np.float64)
    if o["token"] != int(tok[i]) or not np.array_equal(got, a[i].astype(np.float64)):
        sys.exit(f"FAIL replay frame {k}: token {o['token']} vs {int(tok[i])}, max diff {np.abs(got - a[i].astype(np.float64)).max():.3g}")
print(f"frames {len(idx)} equal")
