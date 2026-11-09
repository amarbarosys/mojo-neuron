"""A2b verifier (docs/prereg-a2b.md): batch vs single-process re-runs on kl, kl_ctl and dnll, identity and gain 0.999 scoring 0, wrong-neuron mutant, floor control."""
import sys, statistics
O, single, L = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
def rd(p):
    rows = [l.split("\t") for l in open(p).read().split("\n")[1:] if l]
    return {int(r[0]): tuple(float(x) for x in r[1:]) for r in rows}  # kl, dnll[, kl_ctl, excess]
close = lambda x, y: abs(x - y) <= 1e-6 * max(abs(x), abs(y)) + 1e-12
b = rd(f"{O}/batch/L{L}.scan.tsv")
for n in single:
    s = rd(f"{O}/single-{n}/L{L}.scan.tsv")[int(n)]
    if not all(close(s[i], b[int(n)][i]) for i in (0, 1, 2)): raise SystemExit(f"FAIL single {n}: single {s} vs batch {b[int(n)]}")
for name in ("ident", "g999"):
    r = rd(f"{O}/{name}/L{L}.scan.tsv")
    if max(v[0] for v in r.values()) > 1e-9: raise SystemExit(f"FAIL {name}: scores {r}")
a, c = int(single[0]), int(single[1])
sa = rd(f"{O}/single-{a}/L{L}.scan.tsv")[a]
if close(sa[0], b[c][0]) or close(sa[2], b[c][2]): raise SystemExit(f"FAIL mutant: neuron {a} single equals batch neuron {c} in a column")
med = statistics.median(v[2] for v in b.values())
off = [n for n, v in b.items() if not 0.5 * med <= v[2] <= 2 * med]
if off: raise SystemExit(f"FAIL floor: kl_ctl of {off} outside 0.5x..2x of the median {med:.3e}")
det = [n for n, v in b.items() if v[3] > 2 * med]
print(f"{len(b)} neurons, floor (median kl_ctl) {med:.3e}, kl {min(v[0] for v in b.values()):.3e}..{max(v[0] for v in b.values()):.3e}, detectable (excess > 2x floor): {det or 'none'}")
print("singles equal the batch to 1e-6 relative on kl, dnll, kl_ctl; identity and gain 0.999 score 0; mutant rejected; floor within 0.5x..2x")
