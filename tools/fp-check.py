"""fp-check DIR OUTDIR enrolled.. -- heldout.json noise: F4 bars (docs/prereg-f4.md) from `nx fingerprint match` outputs in OUTDIR/m-<name>.json (enrolled
models matched from a fresh run), OUTDIR/m-heldout.json, and the stored signatures: shuffled-band control done here from the stored files."""
import json, random, sys
d, o = sys.argv[1], sys.argv[2]; names = sys.argv[3:]
sig = {n: json.load(open(f"{d}/{n}.json"))["sig"] for n in names}
def dist(a, b): return 1 - sum(x * y for x, y in zip(a, b))
selfd = {}
for n in names:
    m = json.load(open(f"{o}/m-{n}.json")); a = {r["name"]: r["dist"] for r in m["all"]}
    others = sorted(v for k, v in a.items() if k != n)
    selfd[n] = a[n]
    print(f"{n}: nearest {m['nearest']} self {a[n]:.3g} second-nearest {others[0]:.3g}")
    assert m["nearest"] == n, f"FAIL fingerprint: {n} matched {m['nearest']}"
    assert a[n] < 0.5 * others[0], f"FAIL fingerprint: {n} self {a[n]:.3g} not below half of {others[0]:.3g}"
noise = max(selfd.values()); h = json.load(open(f"{o}/m-heldout.json"))
print(f"re-run noise (largest self distance) {noise:.3g}; held-out nearest {h['nearest']} at {h['dist']:.3g}, unknown={h['unknown']}")
assert h["unknown"] == 1 and h["dist"] > 3 * max(noise, 1e-9), "FAIL fingerprint: the held-out model was not flagged unknown"
pair = min(dist(sig[a], sig[b]) for a in names for b in names if a < b)
random.seed(1); s = sig[names[0]][:]; random.shuffle(s)
sd = min(dist(s, sig[n]) for n in names)
print(f"smallest enrolled pair distance {pair:.3g}; shuffled-band control nearest {sd:.3g}")
assert sd > 3 * max(noise, 1e-9), "FAIL fingerprint: the shuffled-band control matched an enrolled model"
