"""replay-ui-oracle CAPDIR REQ K STATE_FILE: the app's scrubbed strip (STATE ... rp=k:token:c=v,...) equals record K of request REQ in the ORIGINAL capture
(token, the 6 strongest channels of the y row by |value| and their f16 values, exactly). Prints the original record's index."""
import json, re, sys
import numpy as np
d, req, k, sf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
m = json.load(open(d + "/meta.json")); h = m["h"]; L = m["layers"][0]; nf = len(m["fields"]); row = m["fields"].find("y"); row = row if row >= 0 else nf - 1
r = np.fromfile(d + "/req.i32", dtype=np.int32); tok = np.fromfile(d + "/tokens.i32", dtype=np.int32); i = int(np.nonzero(r == req)[0][k])
a = np.fromfile(d + "/L%d.f16" % L, dtype=np.float16).reshape(-1, nf, h)[i, row].astype(np.float64)
idx = np.argsort(-np.abs(a), kind="stable")[:6]
got = re.search(r" rp=(\S+)", open(sf).read()).group(1).strip(",").split(":")
cells = [c.split("=") for c in got[2].strip(",").split(",")]
ok = int(got[0]) == k and int(got[1]) == int(tok[i]) and [int(c[0]) for c in cells] == [int(x) for x in idx] and all(float(c[1]) == a[int(c[0])] for c in cells)
if not ok: sys.exit(f"FAIL replay-ui: frame {k}: app says {got}, original record {i} has token {int(tok[i])} channels {[int(x) for x in idx]}")
print(i)
