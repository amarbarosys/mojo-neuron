"""xmap A.gguf B.gguf CORPUS LA LB [--windows N] [--seg S] [--sample K] [--chunk W] [--skip0]: cross-model residual map (docs/prereg-b3.md).
Both models are dense llama/qwen2/granite with ONE tokenizer (same ids). Ridge map M (hB x hA, `ridge` of rlens.mojo) from the residual
after block LA of A to the residual after block LB of B, fitted on the first 90 % of the windows, scored (R2) on the rest, with a control
fitted on B's rows permuted across positions. Neuron match: K sampled A neurons' unit W_down columns through M against all unit W_down
columns of B (cosine, top-1), real map vs control map. Prints the numbers and ends with rlog.done (bar missed -> anomaly)."""
from std.sys import argv, exit
from std.math import sqrt
from std.random import seed, random_float64
from ggufx.model import Gguf
from mneuron.ggio import kv_int, kv_str, tensor, load_rows, llama_ids
from mneuron.fwd import forward, matmul
from mneuron.store import read_text
from mneuron.rlog import Rlog
from rlens import ridge


def fit(X: List[Float32], Y: List[Float32], n: Int, ha: Int, hb: Int) raises -> List[Float32]:
    """Ridge map hb x ha from X (n x ha) to Y (n x hb)."""
    var xt = List[Float32](length=ha * n, fill=0)
    var yt = List[Float32](length=hb * n, fill=0)
    for p in range(n):
        for j in range(ha):
            xt[j * n + p] = X[p * ha + j]
        for j in range(hb):
            yt[j * n + p] = Y[p * hb + j]
    var A = matmul(xt, xt, ha, n, ha)
    var B = matmul(xt, yt, ha, n, hb)  # B[t][o], t over hb, o over ha
    var Bo = List[Float32](length=ha * hb, fill=0)
    for o in range(ha):
        for t in range(hb):
            Bo[o * hb + t] = B[t * ha + o]
    return ridge(A, Bo, ha, 1e-3, hb)


def r2(M: List[Float32], X: List[Float32], Y: List[Float32], n0: Int, n: Int, ha: Int, hb: Int) -> Float64:
    var P = matmul(M, List[Float32](X[n0 * ha : n * ha]), hb, ha, n - n0)
    var mean = List[Float64](length=hb, fill=0)
    for p in range(n0, n):
        for j in range(hb):
            mean[j] += Float64(Y[p * hb + j]) / Float64(n - n0)
    var se: Float64 = 0
    var st: Float64 = 0
    for p in range(n0, n):
        for j in range(hb):
            var e = Float64(P[(p - n0) * hb + j]) - Float64(Y[p * hb + j])
            var d = Float64(Y[p * hb + j]) - mean[j]
            se += e * e
            st += d * d
    return 1.0 - se / st


def unit_down(g: Gguf, layer: Int, h: Int) raises -> List[Float32]:
    """Rows n = unit W_down column n of block `layer` (ffn x h)."""
    var di = tensor(g, "blk." + String(layer) + ".ffn_down.weight")
    var ffn = g.tensors[di].dims[0]
    var D = load_rows(g, di, 0, h)
    var U = List[Float32](length=ffn * h, fill=0)
    for n in range(ffn):
        var s: Float64 = 0
        for j in range(h):
            s += Float64(D[j * ffn + n]) ** 2
        var inv = Float32(1.0 / sqrt(s)) if s > 0 else Float32(0)
        for j in range(h):
            U[n * h + j] = D[j * ffn + n] * inv
    return U^


def top1(M: List[Float32], U: List[Float32], idx: List[Int], ha: Int, hb: Int, ub: List[Float32], nb: Int) -> Float64:
    """Mean over the sampled A neurons of max_b cos(M u_a, u_b)."""
    var q = List[Float32](length=len(idx) * ha, fill=0)
    for r in range(len(idx)):
        for j in range(ha):
            q[r * ha + j] = U[idx[r] * ha + j]
    var T = matmul(M, q, hb, ha, len(idx))
    var sc = matmul(ub, T, nb, hb, len(idx))
    var tot: Float64 = 0
    for r in range(len(idx)):
        var s: Float64 = 0
        for j in range(hb):
            s += Float64(T[r * hb + j]) ** 2
        var best = Float32.MIN
        for b in range(nb):
            best = max(best, sc[r * nb + b])
        tot += Float64(best) / sqrt(s)
    return tot / Float64(len(idx))


def main() raises:
    var a = List[String]()
    var windows = 800
    var seg = 128
    var sample = 512
    var chunk = 64
    var skip0 = False
    var av = argv()
    var i = 1
    while i < len(av):
        var t = String(av[i])
        if t == "--windows" and i + 1 < len(av):
            windows = Int(String(av[i + 1]))
            i += 2
        elif t == "--seg" and i + 1 < len(av):
            seg = Int(String(av[i + 1]))
            i += 2
        elif t == "--sample" and i + 1 < len(av):
            sample = Int(String(av[i + 1]))
            i += 2
        elif t == "--skip0":
            skip0 = True
            i += 1
        elif t == "--chunk" and i + 1 < len(av):
            chunk = Int(String(av[i + 1]))
            i += 2
        else:
            a.append(t)
            i += 1
    if len(a) < 5:
        print("usage: xmap A.gguf B.gguf CORPUS LA LB [--windows N] [--seg S] [--sample K] [--chunk W]")
        exit(1)
    var la = Int(a[3])
    var lb = Int(a[4])
    var rl = Rlog("xmap", String(" ").join(a), a[0])
    var ga = Gguf(a[0])
    var gb = Gguf(a[1])
    var text = read_text(a[2])
    var ids = llama_ids(a[0], text, ".work/xmap-text.txt")
    var idb = llama_ids(a[1], text, ".work/xmap-text.txt")
    if len(ids) != len(idb):
        raise Error("tokenizers differ: " + String(len(ids)) + " vs " + String(len(idb)) + " tokens for the same text")
    for t in range(len(ids)):
        if ids[t] != idb[t]:
            raise Error("tokenizers differ at token " + String(t))
    var nw = min(windows, len(ids) // seg)
    var X = List[Float32]()
    var Y = List[Float32]()
    var ha = 0
    var hb = 0
    var w0 = 0
    while w0 < nw:
        var w1 = min(w0 + chunk, nw)
        var seq = List[Int]()
        for t in range(w0 * seg, w1 * seg):
            seq.append(ids[t])
        var ra = forward(ga, seq, List[Int](), [la], seg)
        var rb = forward(gb, seq, List[Int](), [lb], seg)
        ha = len(ra[1][0]) // len(seq)
        hb = len(rb[1][0]) // len(seq)
        X.extend(ra[1][0].copy())
        Y.extend(rb[1][0].copy())
        w0 = w1
    var n = nw * seg
    var nfit = (nw - max(1, nw // 10)) * seg
    if skip0:  # post-hoc diagnostic (not the prereg number): drop position 0 of every window, the attention-sink row with a massive residual
        var X2 = List[Float32]()
        var Y2 = List[Float32]()
        for p in range(n):
            if p % seg != 0:
                X2.extend(List[Float32](X[p * ha : (p + 1) * ha]))
                Y2.extend(List[Float32](Y[p * hb : (p + 1) * hb]))
        nfit = nfit - nfit // seg
        n = n - nw
        X = X2^
        Y = Y2^
    var M = fit(X, Y, nfit, ha, hb)
    var real = r2(M, X, Y, nfit, n, ha, hb)
    seed(11)
    var Ys = List[Float32](length=n * hb, fill=0)  # control: B's rows permuted across positions
    var perm = List[Int]()
    for p in range(n):
        perm.append(p)
    for p in range(n - 1, 0, -1):
        var r = min(Int(random_float64(0, Float64(p + 1))), p)
        var tmp = perm[p]
        perm[p] = perm[r]
        perm[r] = tmp
    for p in range(n):
        for j in range(hb):
            Ys[p * hb + j] = Y[perm[p] * hb + j]
    var Mc = fit(X, Ys, nfit, ha, hb)
    var ctrl = r2(Mc, X, Ys, nfit, n, ha, hb)
    var ctrl_real = r2(Mc, X, Y, nfit, n, ha, hb)  # control map scored against the real held-out targets
    var ua = unit_down(ga, la, ha)
    var ub = unit_down(gb, lb, hb)
    var na = len(ua) // ha
    var nb = len(ub) // hb
    var idx = List[Int]()
    var stp = max(1, na // sample)
    for q in range(0, na, stp):
        idx.append(q)
    var m_real = top1(M, ua, idx, ha, hb, ub, nb)
    var m_ctrl = top1(Mc, ua, idx, ha, hb, ub, nb)
    var res = "r2=" + String(real) + " r2_shuffled_control=" + String(ctrl) + " r2_control_map_on_real_targets=" + String(ctrl_real) + " match_top1_real=" + String(m_real) + " match_top1_control=" + String(m_ctrl) + " ratio=" + String(m_real / m_ctrl) + " skip0=" + String(skip0) + " positions=" + String(n) + " ha=" + String(ha) + " hb=" + String(hb) + " LA=" + String(la) + " LB=" + String(lb)
    print(res)
    var vals = List[Float64]()
    vals.append(real)
    vals.append(ctrl)
    vals.append(m_real)
    rl.done(res, "", vals, -1.0, 1.0)
    if skip0:
        print("(skip0 is a post-hoc diagnostic; the prereg numbers are the run without it)")
        return
    var miss = String()
    if real < 0.50:
        miss += " r2 " + String(real) + " < 0.50;"
    if ctrl > 0.05:
        miss += " control r2 " + String(ctrl) + " > 0.05;"
    if m_real < 1.5 * m_ctrl:
        miss += " match ratio " + String(m_real / m_ctrl) + " < 1.5;"
    if miss.byte_length() > 0:
        rl.event("anomaly", "prereg bar missed:" + miss + " | " + res)
        print("BARS MISSED:" + miss)
    else:
        print("PASS xmap: all prereg-b3 bars 1 and 2")
