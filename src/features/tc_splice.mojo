"""tc_splice MODEL.gguf CORPUS LAYER TC_DIR NF K [--w0 A] [--w1 B] [--seg S] [--max W] [--bar R]: CPU gate of a transcoder (tc_train's tc.bin) on a dense model:
held-out NLL of windows [A, B) (last W of them, same windowing as tccap) with the block's MLP kept, zeroed, and replaced by the transcoder
output. Prints fvu (CPU, own variance) and the ratio (splice - base) / (zero - base); ratio > R (prereg bar) logs an rlog anomaly."""
from std.sys import argv, exit
from std.math import exp, log
from ggufx.model import Gguf
from mneuron.ggio import kv_int, kv_str, kv_f32, tensor, load_rows, llama_ids
from mneuron.fwd import forward, matmul, rmsnorm
from mneuron.store import read_text
from mneuron.rlog import Rlog


def f32s(path: String, off: Int, n: Int) raises -> List[Float32]:
    var b = open(path, "r").read_bytes()
    var r = List[Float32](length=n, fill=0)
    var p = b.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(n):
        r[i] = p[off + i]
    _ = len(b)
    return r^


def nll(g: Gguf, res: List[Float32], ids: List[Int], seg: Int, h: Int, wu: List[Float32], nv: Int, nw_: List[Float32], eps: Float32) -> Float64:
    """Mean next-token NLL inside windows from the final residual `res` (T x h)."""
    var T = len(ids)
    var xn = rmsnorm(res, nw_, T, h, eps)
    var tot: Float64 = 0
    var cnt = 0
    var c0 = 0
    while c0 < T:
        var cn = min(256, T - c0)
        var rows = List[Float32](length=cn * h, fill=0)
        for i in range(cn * h):
            rows[i] = xn[c0 * h + i]
        var lg = matmul(wu, rows, nv, h, cn)
        for t in range(cn):
            var gt = c0 + t
            if gt % seg == seg - 1:
                continue
            var mx = Float32.MIN
            for v in range(nv):
                mx = max(mx, lg[t * nv + v])
            var s: Float64 = 0
            for v in range(nv):
                s += exp(Float64(lg[t * nv + v] - mx))
            tot += Float64(mx) + log(s) - Float64(lg[t * nv + ids[gt + 1]])
            cnt += 1
        c0 += cn
    return tot / Float64(cnt)


def main() raises:
    var a = List[String]()
    var w0 = 0
    var w1 = 0
    var seg = 128
    var mx = 16
    var bar = 0.5
    var av = argv()
    var i = 1
    while i < len(av):
        var t = String(av[i])
        if t == "--w0" and i + 1 < len(av):
            w0 = Int(String(av[i + 1]))
            i += 2
        elif t == "--w1" and i + 1 < len(av):
            w1 = Int(String(av[i + 1]))
            i += 2
        elif t == "--seg" and i + 1 < len(av):
            seg = Int(String(av[i + 1]))
            i += 2
        elif t == "--max" and i + 1 < len(av):
            mx = Int(String(av[i + 1]))
            i += 2
        elif t == "--bar" and i + 1 < len(av):
            bar = Float64(String(av[i + 1]))
            i += 2
        else:
            a.append(t)
            i += 1
    if len(a) < 6 or w1 <= w0:
        print("usage: tc_splice MODEL.gguf CORPUS LAYER TC_DIR NF K --w0 A --w1 B [--seg S] [--max W] [--bar R]")
        exit(1)
    var layer = Int(a[2])
    var nf = Int(a[4])
    var k = Int(a[5])
    var rl = Rlog("tc_splice", String(" ").join(a), a[0])
    var g = Gguf(a[0])
    var arch = kv_str(g, "general.architecture")
    var nl = kv_int(g, arch + ".block_count")
    var all = llama_ids(a[0], read_text(a[1]), a[3] + "/corpus-tmp.txt")
    var wa = max(w0, w1 - mx)
    var ids = List[Int]()
    for t in range(wa * seg, w1 * seg + 1):  # +1: the next token of the last position is never scored (window ends), kept for indexing
        ids.append(all[t])
    var seq = List[Int]()
    for t in range((w1 - wa) * seg):
        seq.append(ids[t])
    var T = len(seq)
    var base = forward(g, seq, List[Int](), [nl - 1], seg, layer)
    ref cap = base[2]
    var h = len(cap[0]) // T
    var wb = String(a[3]) + "/tc.bin"
    var we = f32s(wb, 0, h * nf)
    var be = f32s(wb, h * nf, nf)
    var wd = f32s(wb, h * nf + nf, nf * h)
    var bd = f32s(wb, h * nf + nf + nf * h, h)
    var wet = List[Float32](length=nf * h, fill=0)
    for j in range(h):
        for f in range(nf):
            wet[f * h + j] = we[j * nf + f]
    var z = matmul(wet, cap[0], nf, h, T)
    var yh = List[Float32](length=T * h, fill=0)
    for t in range(T):
        var vals = List[Float32]()
        for f in range(nf):
            z[t * nf + f] += be[f]
            if z[t * nf + f] > 0:
                vals.append(z[t * nf + f])
        var thr = Float32(0)
        if len(vals) > k:
            sort(vals)
            thr = vals[len(vals) - k]
        for j in range(h):
            yh[t * h + j] = bd[j]
        for f in range(nf):
            if z[t * nf + f] > 0 and z[t * nf + f] >= thr:
                for j in range(h):
                    yh[t * h + j] += z[t * nf + f] * wd[f * h + j]
    var mean = List[Float64](length=h, fill=0)
    for t in range(T):
        for j in range(h):
            mean[j] += Float64(cap[1][t * h + j]) / Float64(T)
    var vt: Float64 = 0
    var se: Float64 = 0
    for t in range(T):
        for j in range(h):
            var dv = Float64(cap[1][t * h + j]) - mean[j]
            vt += dv * dv
            var e = Float64(yh[t * h + j]) - Float64(cap[1][t * h + j])
            se += e * e
    var fvu = se / vt
    var ni = tensor(g, "output_norm.weight")
    var ui = tensor(g, "output.weight")
    if ui < 0:
        ui = tensor(g, "token_embd.weight")
    var nv = g.tensors[ui].dims[1]
    var wu = load_rows(g, ui, 0, nv)
    var nw_ = load_rows(g, ni, 0, 1)
    var eps = kv_f32(g, arch + ".attention.layer_norm_rms_epsilon", 1e-5)
    var n0 = nll(g, base[1][0], seq, seg, h, wu, nv, nw_, eps)
    var zr = forward(g, seq, List[Int](), [nl - 1], seg, layer, List[Float32](length=T * h, fill=0))
    var n1 = nll(g, zr[1][0], seq, seg, h, wu, nv, nw_, eps)
    var sr = forward(g, seq, List[Int](), [nl - 1], seg, layer, yh)
    var n2 = nll(g, sr[1][0], seq, seg, h, wu, nv, nw_, eps)
    var ratio = (n2 - n0) / (n1 - n0)
    var res = "fvu=" + String(fvu) + " nll_base=" + String(n0) + " nll_zero_mlp=" + String(n1) + " nll_splice=" + String(n2) + " ratio=" + String(ratio) + " tokens=" + String(T) + " layer=" + String(layer)
    print(res)
    var vals = List[Float64]()
    vals.append(fvu)
    vals.append(n0)
    vals.append(n1)
    vals.append(n2)
    rl.done(res, a[3] + "/tc.bin", vals, 0.0, 30.0)
    if ratio > bar:
        rl.event("anomaly", "prereg bar missed: spliced NLL delta is " + String(ratio) + " of zero-ablation (bar " + String(bar) + ") | " + res, a[3] + "/tc.bin")
    print("PASS" if ratio <= bar else "BAR MISSED", "splice ratio", ratio, "bar", bar)
