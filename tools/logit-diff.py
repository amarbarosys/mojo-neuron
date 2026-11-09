"""logit-diff DIR: DIR/{none,persona,recipe}/row-0.bin (f32 logits). Prints max|persona-recipe| and max|persona-none|; exit 1 when the prereg
bars fail (docs/prereg-f1.md): persona == recipe exactly, persona differs from unsteered by >= 0.1."""
import sys
import numpy as np
d = sys.argv[1]
r = {k: np.fromfile(f"{d}/{k}/row-0.bin", dtype=np.float32) for k in ("none", "persona", "recipe")}
assert r["none"].size == r["persona"].size == r["recipe"].size and r["none"].size > 1000, {k: v.size for k, v in r.items()}
assert all(np.isfinite(v).all() for v in r.values()), "non-finite logits"
pr, pn = float(np.abs(r["persona"] - r["recipe"]).max()), float(np.abs(r["persona"] - r["none"]).max())
print(f"max|persona-recipe| {pr:.6g}  max|persona-unsteered| {pn:.6g}  argmax none {int(r['none'].argmax())} persona {int(r['persona'].argmax())}")
sys.exit(0 if pr == 0.0 and pn >= 0.1 else 1)
