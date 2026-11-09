"""tcstab TC_A TC_B NF H [MODEL.gguf LAYER]: seed stability of two transcoder runs (docs/prereg-b4.md): per feature of A the best cosine of its decoder row
against all decoder rows of B; share >= 0.7. With MODEL and LAYER also the neuron-feature share: share of A's features with cosine >= 0.5 to a unit W_down column."""
from std.sys import argv
from std.math import sqrt
from ggufx.model import Gguf
from mneuron.ggio import tensor, load_rows
from mneuron.fwd import matmul
from mneuron.rlog import Rlog


def unit_dec(path: String, nf: Int, h: Int) raises -> List[Float32]:
    var b = open(path + "/tc.bin", "r").read_bytes()
    var p = b.unsafe_ptr().unsafe_bitcast[Float32]()
    var U = List[Float32](length=nf * h, fill=0)
    for f in range(nf):
        var s: Float64 = 0
        for j in range(h):
            s += Float64(p[h * nf + nf + f * h + j]) ** 2
        var inv = Float32(1.0 / sqrt(s))
        for j in range(h):
            U[f * h + j] = p[h * nf + nf + f * h + j] * inv
    _ = len(b)
    return U^


def best_share(A: List[Float32], B: List[Float32], na: Int, nb: Int, h: Int, thr: Float32) -> Float64:
    var c = matmul(B, A, nb, h, na)  # c[a][b] = cos
    var hit = 0
    for a in range(na):
        var best = Float32.MIN
        for b in range(nb):
            best = max(best, c[a * nb + b])
        if best >= thr:
            hit += 1
    return Float64(hit) / Float64(na)


def main() raises:
    var av = argv()
    var nf = Int(String(av[3]))
    var h = Int(String(av[4]))
    var rl = Rlog("tcstab", String(av[1]) + " " + String(av[2]), "")
    var A = unit_dec(String(av[1]), nf, h)
    var B = unit_dec(String(av[2]), nf, h)
    var stab = best_share(A, B, nf, nf, h, 0.7)
    var res = "seed_stability_share_cos>=0.7=" + String(stab)
    var nfs: Float64 = -1
    if len(av) > 6:
        var g = Gguf(String(av[5]))
        var di = tensor(g, "blk." + String(Int(String(av[6]))) + ".ffn_down.weight")
        var ffn = g.tensors[di].dims[0]
        var D = load_rows(g, di, 0, h)
        var N = List[Float32](length=ffn * h, fill=0)
        for n in range(ffn):
            var s: Float64 = 0
            for j in range(h):
                s += Float64(D[j * ffn + n]) ** 2
            var inv = Float32(1.0 / sqrt(s)) if s > 0 else Float32(0)
            for j in range(h):
                N[n * h + j] = D[j * ffn + n] * inv
        nfs = best_share(A, N, nf, ffn, h, 0.5)
        res += " neuron_feature_share_cos>=0.5=" + String(nfs)
    print(res)
    var vals = List[Float64]()
    vals.append(stab)
    rl.done(res, "", vals, 0.0, 1.0)
    if stab < 0.30:
        rl.event("anomaly", "prereg bar missed: seed stability " + String(stab) + " < 0.30 | " + res)
        print("BAR MISSED: seed stability", stab, "< 0.30")
    else:
        print("PASS seed stability", stab, ">= 0.30")
