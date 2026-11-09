"""serve-diff DIR: DIR/{plain,p1,p2}.json (baro-serve completions with logits_topk) and DIR/row/row-0.bin (the engine run directly with steer_file = the recipe
vector). Bars (docs/prereg-f1c.md): p1 == direct within 1e-3, plain vs p1 >= 0.1, p1 vs p2 >= 0.1 on the shared top-k tokens (no shared token counts as different)."""
import json, sys
import numpy as np
d = sys.argv[1]
top = lambda n: {t["id"]: t["logit"] for t in json.load(open(f"{d}/{n}.json"))["logits_topk"][0]}
plain, p1, p2 = top("plain"), top("p1"), top("p2")
row = np.fromfile(f"{d}/row/row-0.bin", dtype=np.float32)
bad = max(abs(float(row[i]) - v) for i, v in p1.items())
def gap(a, b):
    s = [i for i in a if i in b]
    return max(abs(a[i] - b[i]) for i in s) if s else 1e9
g1, g2 = gap(plain, p1), gap(p1, p2)
print(f"HTTP persona vs direct steer_file max |logit diff| {bad:.3e}; plain vs persona {g1:.4g}; persona 1 vs persona 2 {g2:.4g}")
if bad > 1e-3: sys.exit(f"FAIL persona-serve: HTTP persona logits differ from the direct steer_file run by {bad:.3e}")
if g1 < 0.1: sys.exit(f"FAIL persona-serve: persona acts on the logits by only {g1:.3g} (< 0.1)")
if g2 < 0.1: sys.exit(f"FAIL persona-serve: two personas differ by only {g2:.3g} (< 0.1)")
