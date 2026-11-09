"""Oracle for rlens: refit the ridge map of the dumped layer with numpy (float64) and compare with the Mojo map.
Usage: check_ridge.py DUMPDIR   (x.f32, y.f32, m.f32 from `rlens --dump`). A mutant with 100x the ridge must be far from the Mojo map."""
import sys, numpy as np
d = sys.argv[1]
hdr = np.fromfile(f'{d}/m.f32', dtype=np.uint32, count=4); h, n, layer, nf = map(int, hdr)
M = np.fromfile(f'{d}/m.f32', dtype=np.float32, offset=16).reshape(h, h).astype(np.float64)
X = np.fromfile(f'{d}/x.f32', dtype=np.float32).reshape(n, h).astype(np.float64)
Y = np.fromfile(f'{d}/y.f32', dtype=np.float32).reshape(n, h).astype(np.float64)
Xf, Yf = X[:nf], Y[:nf]
A = Xf.T @ Xf
def fit(rel):
    lam = rel * np.trace(A) / h
    return np.linalg.solve(A + lam * np.eye(h), Xf.T @ Yf).T
ref = fit(1e-3)
err = np.abs(M - ref).max() / np.abs(ref).max()
mut = np.abs(M - fit(1e-1)).max() / np.abs(ref).max()
res = np.linalg.norm(Y[nf:] - X[nf:] @ M.T) / np.linalg.norm(Y[nf:])
idn = np.linalg.norm(Y[nf:] - X[nf:]) / np.linalg.norm(Y[nf:])
print(f"layer {layer} hidden {h} positions {n}: max rel err vs numpy {err:.2e}, mutant(100x ridge) {mut:.2e}, heldout residual {res:.3f} vs identity {idn:.3f}")
# the Mojo Gram and X^T Y are f32 sums: 5e-3 relative is the bar, the mutant sits near 0.67
if err > 5e-3:
    sys.exit("FAIL ridge: Mojo map differs from numpy lstsq")
if mut < 50 * err:
    sys.exit("FAIL ridge: mutant not distinguishable")
if nf < n and res >= idn:
    sys.exit("FAIL ridge: the map does not beat the identity on held-out positions")
