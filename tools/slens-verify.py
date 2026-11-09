"""slens-verify STEER.bin GGUF RHO CVEC PROMPTS N...: steer.bin records of neurons N... against llama.cpp.

Exact mean (over the prompts) of the log-prob change at the last position under the neuron's vector is computed from
full llama.cpp logits; the file's top-up / top-down ids must overlap the exact top-8 by >= 6 of 8 and the shared tokens'
values agree within 15 %. Mutant: neuron N's record against neuron N+1's exact answer must overlap < 4 of 8."""
import struct, subprocess, sys, tempfile, ctypes
import numpy as np
import llama_cpp as lc

bin_, gguf, rho, cvec, prompts = sys.argv[1:6]
ns = [int(x) for x in sys.argv[6:]]
b = open(bin_, "rb").read()
magic, ver, layer, ffn, k, dose = struct.unpack("<4sIIIIf", b[:24])
assert magic == b"MNST" and ver == 1, magic
rec = 4 * k * 4
llm = lc.Llama(model_path=gguf, n_ctx=256, n_gpu_layers=0, logits_all=True, verbose=False, n_threads=8)
nl, nh = lc.llama_model_n_layer(llm._model.model), llm.n_embd()
ctxs = [llm.tokenize(l.encode(), add_bos=True) for l in open(prompts).read().split("\n") if l]
tmp = tempfile.mkdtemp()

def last_lsm(ids, vec):
    llm.reset()
    if vec is None:
        lc.llama_set_adapter_cvec(llm.ctx, None, 0, nh, 0, 0)
    else:
        buf = np.ascontiguousarray(vec[nh:])
        lc.llama_set_adapter_cvec(llm.ctx, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buf.size, nh, 1, nl - 1)
    llm.eval(ids)
    x = np.asarray(llm.scores[len(ids) - 1], dtype=np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())

base = [last_lsm(c, None) for c in ctxs]
klf = np.fromfile(bin_.replace(".steer.bin", ".steer.kl.f32"), dtype=np.float32)
def exact_kl(n):
    v = np.fromfile(f"{tmp}/v{n}.f32", dtype=np.float32)
    return float(np.mean([np.sum(np.exp(bs) * (bs - last_lsm(c, v))) for c, bs in zip(ctxs, base)]))

def exact(n):
    subprocess.run([cvec, gguf, f"{layer}:{n}:{dose}", f"{tmp}/v{n}", rho], check=True, capture_output=True)
    v = np.fromfile(f"{tmp}/v{n}.f32", dtype=np.float32)
    return np.mean([last_lsm(c, v) - bs for c, bs in zip(ctxs, base)], axis=0)

def record(n):
    o = 24 + n * rec
    f = lambda q, t: struct.unpack(f"<{k}{t}", b[o + q * 4 * k:o + (q + 1) * 4 * k])
    return (f(0, "i"), f(1, "f")), (f(2, "i"), f(3, "f"))   # (up ids, values), (down ids, values)

def overlap(n, d, half=0):
    """half 0: up half against the exact top-8 rises of d; half 1: down half against the exact top-8 falls."""
    ids, vs = record(n)[half]
    top = set(np.argsort(-d if half == 0 else d)[:8])
    sh = [(i, v) for i, v in zip(ids[:8], vs[:8]) if i in top]
    err = max((abs(v - d[i]) / max(abs(d[i]), 1e-6) for i, v in sh), default=9)
    return len(sh), err

bad = 0
for n in ns:
    d = exact(n)
    ek = exact_kl(n)
    dn = exact(n + 1)
    for half, name in ((0, "up"), (1, "down")):
        o, e = overlap(n, d, half)
        print(f"neuron {n} {name}: top-8 overlap {o}/8, worst value error {e:.3f}")
        if o < 6 or e > 0.15:
            bad += 1
        om, _ = overlap(n, dn, half)
        if om >= 4:
            print(f"  mutant ({name} half against neuron {n+1}'s answer) overlaps {om}/8"); bad += 1
    print(f"  KL {klf[n]:.4f} vs exact {ek:.4f}")
    if abs(klf[n] - ek) > max(0.15 * ek, 0.02):
        bad += 1
if bad:
    raise SystemExit(f"FAIL slens-verify: {bad} problems")
print("PASS slens-verify:", ns)
