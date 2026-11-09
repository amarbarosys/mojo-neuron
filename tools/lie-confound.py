"""CPU confound hunt for the F24 probe (boss request 2026-11-09). Usage: lie-confound.py PAIRS_TSV CAPDIR LAYER [OUT.json]
1. Surface-only probes (token count, answer position, template, object frequency, object length, role counts, all together): test accuracy.
2. A numpy replica of lie_fit at LAYER (same split, standardise, lambda 1e-2, lr 0.5, 300 its): real accuracy, then shuffled-label controls three ways:
   global, within template (object frame), per pair flip; with cos(w, mean twin difference) per seed to test the shared-truth-direction reading.
Split by pair as in src/probe/lie.mojo part(): (p*7919+13)%10 < 6 train, < 8 val, else test."""
import json, re, sys
from collections import Counter
import numpy as np

pairs_path, capdir, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [l.rstrip("\n").split("\t") for l in open(pairs_path)]
meta = json.load(open(f"{capdir}/meta.json")); h = meta["h"]; nf = len(meta["fields"]); fld = meta["fields"].index("r")
P = len(rows); part = lambda p: (p * 7919 + 13) % 10
split = np.array([0 if part(p) < 6 else (1 if part(p) < 8 else 2) for p in range(P) for _ in (0, 1)])
y = np.tile([1, 0], P); pair = np.repeat(np.arange(P), 2)
text = [r[0] if k == 0 else r[1] for r in rows for k in (0, 1)]
obj = [r[2] if k == 0 else r[3] for r in rows for k in (0, 1)]
req = np.fromfile(f"{capdir}/req.i32", dtype=np.int32)
ntok = np.bincount(req)[1:].astype(np.float64); assert len(ntok) == 2 * P, (len(ntok), 2 * P)
tr, te = split == 0, split == 2

def logistic(X, yy, lam=1e-2, iters=60):  # IRLS, standardised on the fitted rows
    mu, sd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
    Z = (X - mu) / sd; Z1 = np.c_[Z, np.ones(len(Z))]; w = np.zeros(Z1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z1[tr_i] @ w, -30, 30))); W = p * (1 - p) + 1e-9
        g = Z1[tr_i].T @ (p - yy[tr_i]) / tr_i.sum() + lam * np.r_[w[:-1], 0]
        H = (Z1[tr_i] * W[:, None]).T @ Z1[tr_i] / tr_i.sum() + lam * np.eye(len(w)); w -= np.linalg.solve(H, g)
    return (Z1 @ w > 0).astype(int)
tr_i = tr
lower = [t.lower() for t in text]
def frame(t, o):
    i = t.lower().find(o.lower()); pre = t[:i].split()[-2:]; post = t[i + len(o):].split()[:1]
    return " ".join(w.lower() for w in pre) + " <X> " + " ".join(w.lower().strip(".") for w in post)
first2 = [frame(text[i], obj[i]) for i in range(2 * P)]   # name kept: the template id (frame around the swapped object)
tcount = Counter(first2[i] for i in range(2 * P) if tr[i]); big = [t for t, c in tcount.items() if c >= 15]
F = {}
F["token_count"] = np.c_[ntok]
F["answer_position"] = np.array([[lower[i].find(obj[i].lower()) / max(len(text[i]), 1), len(lower[i][:lower[i].find(obj[i].lower())].split())] for i in range(2 * P)])
F["template(object frame)"] = np.array([[float(first2[i] == t) for t in big] + [float(first2[i] not in big)] for i in range(2 * P)])
freq = Counter(); [freq.update([o.lower()]) for o in obj]
inall = np.array([sum(obj[i].lower() in t for t in lower) - 1 for i in range(2 * P)], dtype=float)
F["object_frequency(statements containing it)"] = np.c_[np.log1p(inall)]
F["object_length"] = np.array([[len(o), len(o.split())] for o in obj], dtype=float)
ans_c = Counter(); wr_c = Counter()
for p in range(P):
    if split[2 * p] == 0: ans_c[rows[p][2].lower()] += 1; wr_c[rows[p][3].lower()] += 1
F["object_role_counts(train pairs, leave own out)"] = np.array([[np.log1p(ans_c[obj[i].lower()] - (1 if (tr[i] and y[i] == 1) else 0)), np.log1p(wr_c[obj[i].lower()] - (1 if (tr[i] and y[i] == 0) else 0))] for i in range(2 * P)])
F["ALL surface features"] = np.hstack(list(F.values()))
res = {"n_pairs": P, "n_test_statements": int(te.sum()), "surface_test_acc": {}}
for k, X in F.items():
    pred = logistic(X, y); res["surface_test_acc"][k] = round(float((pred[te] == y[te]).mean()), 4)
    print(f"surface  {k:48s} test acc {res['surface_test_acc'][k]:.3f}")
# twin-pair view: pairs whose true statement scores above its false twin on a surface feature (pairwise), per family
# --- residual probe replica
raw = np.fromfile(f"{capdir}/L{layer}.f16", dtype=np.float16).reshape(-1, nf, h)[:, fld, :]
last = np.flatnonzero(np.r_[req[1:] != req[:-1], True]); X = raw[last].astype(np.float32); assert len(X) == 2 * P
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Zs = (X - mu) / sd; Zt = Zs[tr]
def fit(labels):
    w = np.zeros(h, np.float32); b = 0.0; yt = labels[tr].astype(np.float32)
    for _ in range(300):
        r = (1 / (1 + np.exp(-(Zt @ w + b))) - yt) / len(yt); w -= 0.5 * (Zt.T @ r + 1e-2 * w); b -= 0.5 * r.sum()
    return w, b
def acc(w, b): return float(((Zs[te] @ w + b > 0) == (y[te] == 1)).mean())
def auroc(w, b):
    sc = Zs[te] @ w + b; yy = y[te]; pos, neg = sc[yy == 1], sc[yy == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
def pairwise(w, b):
    sc = Zs @ w + b; m = (split[0::2] == 2); return float((sc[0::2][m] > sc[1::2][m]).mean())
w0, b0 = fit(y); res["real_test_acc"] = round(acc(w0, b0), 4); res["real_auroc"] = round(auroc(w0, b0), 4); res["real_pairwise"] = round(pairwise(w0, b0), 4); print(f"replica  real probe L{layer} test acc {res['real_test_acc']:.3f}  (Mojo lie_fit: see fit.json)")
twin = (Zs[0::2] - Zs[1::2])[(split[0::2] == 0)].mean(0); twin /= np.linalg.norm(twin)
tmpl = np.array([big.index(f) if f in big else len(big) for f in first2])
def shuffled(kind, seed):
    rng = np.random.default_rng(seed); ys = y.copy(); idx = np.flatnonzero(tr)
    if kind == "global": ys[idx] = rng.permutation(ys[idx])
    elif kind == "within template":
        for t in np.unique(tmpl[idx]):
            g = idx[tmpl[idx] == t]; ys[g] = rng.permutation(ys[g])
    else:  # per pair flip
        fl = {p: rng.integers(2) for p in np.unique(pair[idx])}
        for i in idx:
            if fl[pair[i]]: ys[i] = 1 - ys[i]
    return ys
res["controls"] = {}
for kind in ("global", "within template", "pair flip"):
    out = []
    for s in range(int(sys.argv[5]) if len(sys.argv) > 5 else 30):
        w, b = fit(shuffled(kind, 100 + s)); out.append([round(acc(w, b), 3), round(auroc(w, b), 3), round(pairwise(w, b), 3), round(float(w @ twin / np.linalg.norm(w)), 3)])
    a = np.array(out); dev = np.abs(a[:, 0] - 0.5); realdev = abs(res["real_test_acc"] - 0.5)
    res["controls"][kind] = {"seeds": len(out), "acc_min_max": [float(a[:, 0].min()), float(a[:, 0].max())], "max_abs_dev_from_0.5": round(float(dev.max()), 3),
        "share_seeds_with_abs_dev_ge_real": round(float((dev >= realdev).mean()), 3), "auroc_min_max": [float(a[:, 1].min()), float(a[:, 1].max())], "pairwise_min_max": [float(a[:, 2].min()), float(a[:, 2].max())],
        "sign_match_cos_vs_acc": float(((a[:, 3] > 0) == (a[:, 0] > 0.5)).mean())}
    print(f"control  {kind:16s}", json.dumps(res["controls"][kind]))
res["cos_real_w_twin"] = round(float(w0 @ twin / np.linalg.norm(w0)), 3); print("real probe cos(w, twin direction):", res["cos_real_w_twin"])
if len(sys.argv) > 4: json.dump(res, open(sys.argv[4], "w"), indent=1)
