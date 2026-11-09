"""A2c-group helper (docs/prereg-a2c-group.md). `make RANK.tsv OUT.tsv` writes the groups file; `verify DIR` prints the floor first, then the bars F S R O D P from DIR/L16.group.tsv and DIR/pre.de.f32."""
import sys, random
import numpy as np
KS = (8, 32, 128)
def make(rank, out):
    ns = [int(l.split("\t")[0]) for l in open(rank) if l.strip()]
    rows = []
    for rep in "AB":  # the same five random groups of 128 at gain 0.99, twice (bar F)
        for d in range(5): rows.append((f"f{rep}{d}", random.Random(100 + d).sample(ns, 128), 0.99))
    for k in KS:
        rows.append((f"top{k}", ns[:k], 0)); rows.append((f"bot{k}", ns[::-1][:k], 0))
        for d in range(5): rows.append((f"rnd{k}_{d}", random.Random(1000 * k + d).sample(ns, k), 0))
    open(out, "w").write("".join(f"{l}\t{','.join(map(str, g))}\t{k}\n" for l, g, k in rows))
def load(d):
    rows = [l.rstrip("\n").split("\t") for l in open(f"{d}/L16.group.tsv").read().split("\n")[1:] if l]
    out = {}
    for lab, p, kl, dn, dl in rows: out.setdefault(lab, {})[int(p)] = (float(kl), float(dn), float(dl))
    return out
def mean(g, i): return float(np.mean([v[i] for v in g.values()]))
def vec(g, i): return np.array([g[p][i] for p in sorted(g)])
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
def verify(d):
    g = load(d)
    fa = [mean(g[f"fA{i}"], 2) for i in range(5)]; fb = [mean(g[f"fB{i}"], 2) for i in range(5)]
    floor = max(abs(x) for x in fa)
    print(f"FLOOR (logit-difference change under 5 random groups of 128 at gain 0.99): max |mean| = {floor:.4e}; values {[f'{x:.2e}' for x in fa]}")
    F = all(abs(x - y) <= 1e-6 * max(abs(x), abs(y)) + 1e-12 for x, y in zip(fa, fb)); fl = max(floor, 1e-9)
    de = np.fromfile(f"{d}/pre.de.f32", dtype=np.float32); npr = len(g["top8"]); de = de.reshape(npr, -1)
    res = {}
    for k in KS:
        top, bot = mean(g[f"top{k}"], 2), mean(g[f"bot{k}"], 2); rnd = [mean(g[f"rnd{k}_{i}"], 2) for i in range(5)]
        res[k] = dict(top=top, bot=bot, rnd=rnd, klt=mean(g[f"top{k}"], 0), klb=mean(g[f"bot{k}"], 0), klr=float(np.mean([mean(g[f"rnd{k}_{i}"], 0) for i in range(5)])))
    S = [k for k in KS if res[k]["top"] <= -3 * fl]
    kk = None
    for k in S:
        r = res[k]
        if all(abs(x) <= 2 * fl for x in r["rnd"]) and abs(r["top"]) >= 3 * max(abs(x) for x in r["rnd"]) and r["bot"] >= 3 * fl: kk = k; break
    D = all(abs(res[KS[i]]["top"]) <= abs(res[KS[i + 1]]["top"]) for i in range(2))
    grp = {l: [int(x) for x in n.split(",")] for l, n, _ in (ln.split("\t") for ln in open(f"{d}/groups.tsv") if ln.strip())}
    P = None
    if kk:
        pred = -de[:, grp[f"top{kk}"]].sum(1); meas = vec(g[f"top{kk}"], 2); P = spearman(pred, meas)
    print("| k | supporters (top by de) | opposers (bottom) | random x5 | KL top / bottom / random |")
    for k in KS:
        r = res[k]; print(f"| {k} | {r['top']:+.3e} | {r['bot']:+.3e} | {', '.join(f'{x:+.1e}' for x in r['rnd'])} | {r['klt']:.2e} / {r['klb']:.2e} / {r['klr']:.2e} |")
    print(f"F repeat identical: {F}   S (top <= -3x floor at k in {S or 'none'}), R+O jointly at k={kk}   D dose non-decreasing: {D}   P spearman at k={kk}: {P}")
    ok = F and kk is not None and D
    print(("VERDICT PASS" if ok else "VERDICT MISSED") + f" (floor {floor:.2e}; P {'ok' if (P is not None and P >= 0.3) else 'reported, below 0.3 or n/a'})")
if __name__ == "__main__":
    make(sys.argv[2], sys.argv[3]) if sys.argv[1] == "make" else verify(sys.argv[2])
