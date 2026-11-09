"""scan-oracle.py CAPDIR N [TOKEN]: numpy twin of src/sense/scan.mojo on the first N tokens of a capture: the strip signature (6 strongest channels
of the y row by |value|, ties to the lower channel), or with TOKEN the strongest channel of that token."""
import sys, json, numpy as np
d, n = sys.argv[1], int(sys.argv[2]); m = json.load(open(d + "/meta.json")); h = m["h"]; L = m["layers"][0]; nf = len(m["fields"]); row = m["fields"].find("y")
row = row if row >= 0 else nf - 1
a = np.fromfile(d + "/L%d.f16" % L, dtype=np.float16, count=n * nf * h).reshape(n, nf, h)[:, row].astype(np.float64)
idx = np.argsort(-np.abs(a), axis=1, kind="stable")[:, :6]; v = np.take_along_axis(a, idx, 1)
print("n=%d sumabs=%.3f sumch=%d" % (n, round(float(np.abs(v).sum()), 3), int(idx.sum())) if len(sys.argv) < 4 else int(idx[int(sys.argv[3])][0]))
