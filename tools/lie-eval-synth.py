"""Synthetic inputs for tests/lie-eval.sh. Usage: lie-eval-synth.py OUTDIR MODE   MODE: ok | surface (the probe-visible signal follows the surface
form, so the inverted set S4 reads backwards) | noprobe (no truth signal in the sets: every AUROC near 0.5).
Writes main/ (capture layers 15,19, labels.tsv), sets/ (capture, all.tsv, judge.out, tokids.txt), unemb15.f32, unemb19.f32."""
import json, os, sys
import numpy as np
out, mode = sys.argv[1], sys.argv[2]
h, T, nl = 64, 3, 32
rng = np.random.default_rng(5); os.makedirs(f"{out}/main", exist_ok=True); os.makedirs(f"{out}/sets", exist_ok=True)
d = rng.normal(size=h); d /= np.linalg.norm(d)
def rows(signs, strength):  # last token carries sign * strength * d, both band layers
    L = {15: [], 19: []}
    for s in signs:
        for t in range(T):
            x = rng.normal(size=h) + (s * strength * d if t == T - 1 else 0)
            L[15].append(x); L[19].append(x + 0.3 * rng.normal(size=h))
    return L
def write_cap(path, signs, strength):
    L = rows(signs, strength)
    for l in (15, 19): np.asarray(L[l], dtype=np.float16).tofile(f"{path}/L{l}.f16")
    np.repeat(np.arange(1, len(signs) + 1), T).astype(np.int32).tofile(f"{path}/req.i32")
    np.zeros(len(signs) * T, dtype=np.int32).tofile(f"{path}/tokens.i32")
    json.dump({"model": "synth", "dtype": "f16", "h": h, "n_layers": nl, "fields": "r", "layers": [15, 19]}, open(f"{path}/meta.json", "w"))
NP = 400
y = np.tile([1, 0], NP); write_cap(f"{out}/main", np.where(y == 1, 1, -1), 3.0)
open(f"{out}/main/labels.tsv", "w").write("".join(f"{y[i]}\t{i // 2}\tstatement {i}\n" for i in range(2 * NP)))
# the test pairs the sets are built from: pair ids with part == 2 are (p*7919+13)%10 >= 8; take 60 of them
test = [p for p in range(NP) if (p * 7919 + 13) % 10 >= 8][:60]
names, labs, signs = [], [], []
for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
    for p in test:
        for truth in (1, 0):
            lab = (1 - truth) if s == "S4" else truth
            names.append(s); labs.append(lab)
            sign = (1 if truth else -1) if mode == "surface" and s == "S4" else (1 if lab else -1)   # surface: S4 shows the PLAIN truth, not the flipped label
            signs.append(0 if mode == "noprobe" else sign)
write_cap(f"{out}/sets", signs, 3.0)
open(f"{out}/sets/all.tsv", "w").write("".join(f"{n}\t{l}\ttext {i}\n" for i, (n, l) in enumerate(zip(names, labs))))
open(f"{out}/sets/tokids.txt", "w").write("_True 1000\n_true 1001\n_False 1002\n_false 1003\n")
lines = []
for i, l in enumerate(labs):  # output baseline: informative, with noise
    z = (4.0 if l else -4.0) + rng.normal() * 3.0
    top = [{"id": 1000 if z > 0 else 1002, "logit": 10.0 + abs(z)}, {"id": 1002 if z > 0 else 1000, "logit": 10.0 - abs(z)}] + [{"id": 2000 + k, "logit": 5.0 - k * 0.1} for k in range(18)]
    lines.append(json.dumps({"id": i + 1, "logits_topk": top}))
open(f"{out}/sets/judge.out", "w").write("\n".join(lines) + "\n")
u = np.zeros((nl, h), np.float32)
for l in (15, 19): u[l] = d + 4.0 * rng.normal(size=h) / np.sqrt(h) * 3   # a noisy truth direction: the training-free baseline is weaker than the probe
u.tofile(f"{out}/unemb15.f32"); u.tofile(f"{out}/unemb19.f32")
