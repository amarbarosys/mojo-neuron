"""Splice gate verifier, v2 bars (docs/prereg-splice.md run 1 FAILED bars 3 and 4 on NLL; docs/prereg-splice-b.md measures the effect by KL). usage: splice-verify.py DIR"""
import sys, json
import numpy as np
O = sys.argv[1]; meta = json.load(open(f"{O}/meta.json")); got = {}
for l in open(f"{O}/main.out"):
    if l.startswith('{"id":') and '"error"' in l: raise SystemExit("FAIL engine error: " + l[:240])
    if l.startswith('{"id":') and '"slens"' in l: j = json.loads(l); got[j["id"]] = j["slens"]
res = {}
for name, ids in meta.items():
    if any(i not in got for i in ids): raise SystemExit(f"FAIL {name}: missing slens lines")
    res[name] = (np.array([got[i]["nll"] for i in ids]), np.array([got[i]["kl"] for i in ids]))
base = res["none"][0]
def top(n):
    for l in open(f"{O}/{n}.out"):
        if '"logits_topk"' in l: return {e["id"]: e["logit"] for e in json.loads(l)["logits_topk"]}
    raise SystemExit(f"FAIL {n}: no logits_topk line")
new, old = top("row-new"), top("row-old")
if any(new[i] != old[i] for i in new if i in old) or len(set(new) & set(old)) < 15: raise SystemExit("FAIL bar 1: the splice code path changes an unspliced request")
if res["none"][1].max() > 1e-9: raise SystemExit("FAIL bar 1: unspliced request has KL against its own base")
k_id = res["ident"][1].mean(); d_id = np.abs(res["ident"][0] - base).mean()
if d_id > 0.01 or k_id > 1e-3: raise SystemExit(f"FAIL bar 2: identity splice |dNLL| {d_id:.4f}, KL {k_id:.2e}")
for name, bar in (("zero", "3"), ("shift", "4")):
    k = res[name][1].mean()
    if k < 1e-2 or k < 100 * k_id: raise SystemExit(f"FAIL bar {bar}: {name} splice KL {k:.2e} (needs >= 1e-2 and >= 100x identity {k_id:.2e})")
if not (np.array_equal(res["zero"][0], res["zfile"][0]) and np.array_equal(res["zero"][1], res["zfile"][1])): raise SystemExit("FAIL bar 3: splice_zero differs from a zeros file")
print(f"identity KL {k_id:.2e} |dNLL| {d_id:.4f}; zero KL {res['zero'][1].mean():.2e} dNLL {(res['zero'][0] - base).mean():+.4f}; shifted rows KL {res['shift'][1].mean():.2e} dNLL {(res['shift'][0] - base).mean():+.4f}; zeros file == splice_zero exactly; unspliced row equals the old engine's")
