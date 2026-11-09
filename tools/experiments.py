"""experiments X1 X2 X3 (docs/experiments.md): analysis of the layer-12 steer lens and persona contrast of Qwen2.5-0.5B bf16. CPU only."""
import sys, json, math, struct
import numpy as np
from gguf import GGUFReader

GG = "$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf"
B2, RV = ".work/bench2", ".work/rv1"
L, K = 12, 12

def bf16(t):
    a = np.ascontiguousarray(t.data)
    if a.dtype == np.uint8: a = a.view(np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32)

def steer(path):
    b = open(path, "rb").read()
    _, _, _, ffn, k, dose = struct.unpack("<4sIIIIf", b[:24])
    a = np.frombuffer(b[24:], dtype=np.uint8).reshape(ffn, 16 * k)
    ii = lambda q: a[:, q * 4 * k:(q + 1) * 4 * k].copy().view(np.int32)
    return ii(0), ii(2)   # up ids, down ids

def load():
    r = GGUFReader(GG)
    T = {t.name: t for t in r.tensors}
    down = bf16(T[f"blk.{L}.ffn_down.weight"]).reshape(-1, T[f"blk.{L}.ffn_down.weight"].shape[0])  # (H, ffn)
    U = bf16(T["token_embd.weight"]).reshape(-1, down.shape[0])                                      # (V, H), tied head
    nw = np.asarray(T["output_norm.weight"].data, dtype=np.float32)
    toks = [bytes(r.fields["tokenizer.ggml.tokens"].parts[i]).decode("utf-8", "replace") for i in r.fields["tokenizer.ggml.tokens"].data]
    return down, U, nw, toks

def unbyte(t):
    low = [b for b in range(256) if not ((33 <= b <= 126) or (161 <= b <= 172) or b >= 174)]
    out = bytearray()
    for ch in t:
        c = ord(ch)
        out.append(low[c - 256] if 256 <= c < 256 + len(low) else c & 0xFF) if (c >= 256 and c - 256 < len(low)) or c < 256 else out.extend(ch.encode())
    return out.decode("utf-8", "replace")

def lens_top(down, U, nw):
    ffn = down.shape[1]; G = (down / np.linalg.norm(down, axis=0, keepdims=True)).T * nw   # (ffn, H)
    out = np.zeros((ffn, K), dtype=np.int32)
    for s in range(0, ffn, 512):
        Z = U @ G[s:s + 512].T                                                              # (V, chunk)
        idx = np.argpartition(-Z, K, axis=0)[:K]
        o = np.argsort(-np.take_along_axis(Z, idx, axis=0), axis=0)
        out[s:s + 512] = np.take_along_axis(idx, o, axis=0).T
    return out

def shared(a, b):  # per row: number of ids shared
    return (a[:, :, None] == b[:, None, :]).any(axis=2).sum(axis=1)

def x1():
    down, U, nw, _ = load(); lens = lens_top(down, U, nw)
    up, _ = steer(f"{B2}/{L}.steer.0.5.bin"); rng = np.random.default_rng(0)
    obs = float((shared(lens, up) >= 1).mean())
    sh = np.mean([(shared(lens[rng.permutation(len(lens))], up) >= 1).mean() for _ in range(500)])
    print(f"X1 observed {obs:.4f} shuffled {sh:.4f} ratio {obs / max(sh, 1e-9):.2f}  mean shared tokens obs {shared(lens, up).mean():.4f}")
    kl = np.fromfile(f"{B2}/{L}.steer.0.5.kl.f32", dtype=np.float32); top = kl >= np.quantile(kl, 0.95)
    print(f"X1 post hoc (not pre-registered): share among the 5 % highest-KL neurons {(shared(lens, up) >= 1)[top].mean():.4f}, among the rest {(shared(lens, up) >= 1)[~top].mean():.4f}")

def x2():
    up_p, dn_p = steer(f"{B2}/{L}.steer.0.5.bin"); up_m, dn_m = steer(f"{B2}/{L}.steer.-0.5.bin")
    klp = np.fromfile(f"{B2}/{L}.steer.0.5.kl.f32", dtype=np.float32); klm = np.fromfile(f"{B2}/{L}.steer.-0.5.kl.f32", dtype=np.float32)
    p50 = json.load(open(f"{B2}/{L}.steer.0.5.json"))["null_kl"]["p50"]
    sel = klp > p50
    ov = shared(up_p, dn_m)[sel]; ratio = (klp / np.maximum(klm, 1e-9))[sel]
    print(f"X2 neurons above control median KL: {sel.sum()} of {len(sel)}; overlap>=6/12: {(ov >= 6).mean():.3f} (median {np.median(ov):.0f}, mean {ov.mean():.2f}); "
          f"KL ratio in [0.5,2]: {((ratio >= 0.5) & (ratio <= 2)).mean():.3f} (median {np.median(ratio):.2f})")
    ovr = shared(up_p, dn_m[np.random.default_rng(1).permutation(len(dn_m))])[sel]
    print(f"X2 chance (down rows shuffled across neurons): mean overlap {ovr.mean():.3f}")

LEX = {"arr", "arrr", "matey", "ahoy", "aye", "ye", "pirate", "pirates", "ship", "sail", "sea", "treasure", "captain", "plunder", "seas"}
def x3():
    down, U, nw, toks = load()
    d = np.fromfile(f"{RV}/contrast.f32", dtype=np.float32).reshape(-1, down.shape[0])[L]
    cos = (down / np.linalg.norm(down, axis=0, keepdims=True)).T @ d / np.linalg.norm(d)
    up, _ = steer(f"{B2}/{L}.steer.0.5.bin")
    txt = [unbyte(t).strip().lower() for t in toks]
    hit = np.array([any(txt[i] in LEX for i in row if i >= 0) for row in up])
    top = np.argsort(-cos)[:20]
    pop = hit.mean(); rng = np.random.default_rng(2)
    base = np.mean([hit[rng.choice(len(hit), 200, replace=False)].mean() for _ in range(200)])
    obs = hit[top].mean()
    n, Kp, nn = len(hit), int(hit.sum()), 20; k = int(hit[top].sum())
    p = sum(math.comb(Kp, i) * math.comb(n - Kp, nn - i) for i in range(k, min(nn, Kp) + 1)) / math.comb(n, nn)
    print(f"X3 top-20 by cosine: {k} of 20 with a pirate token ({obs:.3f}); population {pop:.4f} (random-200 mean {base:.4f}); enrichment {obs / max(base, 1e-9):.2f}; one-sided hypergeometric p {p:.4f}")
    print("X3 top-5 cos:", [(int(i), round(float(cos[i]), 3)) for i in top[:5]], "lexicon hits in population:", Kp)


def x3b():
    down, U, nw, toks = load(); ids = json.load(open(".work/x3b/ids.json"))
    up, _ = steer(".work/x3b/12.steer.bin")
    txt = [unbyte(t).strip().lower() for t in toks]
    hit = lambda n: any(txt[i] in LEX for i in up[n] if i >= 0)
    top = np.array([hit(n) for n in ids["top"]]); rnd = np.array([hit(n) for n in ids["rnd"]])
    k, nn, Kp, n = int(top.sum()), len(top), int(top.sum() + rnd.sum()), len(top) + len(rnd)
    p = sum(math.comb(Kp, i) * math.comb(n - Kp, nn - i) for i in range(k, min(nn, Kp) + 1)) / math.comb(n, nn)
    print(f"X3b top-100 by cosine: {k} with a lexicon token ({top.mean():.3f}); random 400: {int(rnd.sum())} ({rnd.mean():.4f}); enrichment {top.mean() / max(rnd.mean(), 1e-9):.2f}; one-sided hypergeometric p {p:.4f}")
    ex = [(n, [txt[i] for i in up[n] if txt[i] in LEX]) for n in ids["top"] if hit(n)][:5]
    print("X3b examples (neuron, lexicon tokens raised):", ex)


if __name__ == "__main__":
    {"x1": x1, "x2": x2, "x3": x3, "x3b": x3b}[sys.argv[1]]()
