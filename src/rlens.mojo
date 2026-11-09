"""Rlens MODEL CORPUS [--layers a,b] [--windows N] [--holdout N] [--out DIR] [--dump DIR]: fit per-layer ridge transport maps.

Our R-style lens, not the LRP-Jacobian R lens of jpp_lens: for source layer l, M_l = argmin sum over positions of
|h_final - M h_l|^2 + lambda |M|^2, where h_l is the residual after block l and h_final the residual after the last block (before
the final norm), both from the CPU forward over CORPUS. `lens --rmap` then reads a neuron's write direction through M_l, the final
norm and W_U. Writes DIR/<model stem>/rlens.bin: "MNRL", u32 version 1, u32 hidden, u32 count, then per layer u32 layer + f32 M[hidden][hidden].
--holdout N: the last N windows are left out of the fit (the oracle scores the map on them). --dump DIR: also X and Y of the first fitted layer as raw f32 (N x hidden) for the numpy oracle.
"""
from std.sys import argv, exit
from std.math import sqrt
from std.os import makedirs
from ggufx.model import Gguf
from ggufx.par import parallelize
from mneuron.ggio import kv_int, kv_str, put_u32, put_f32, write_file, llama_ids
from mneuron.fwd import forward, matmul
from mneuron.store import read_text


@fieldwise_init
struct Chol(Copyable):
    var a: Pointer[Float64, MutUntrackedOrigin]  # lower Cholesky factor, n x n row-major
    var b: Pointer[Float64, MutUntrackedOrigin]  # right-hand sides: n x n, column j is RHS j (stored row-major [j][i])
    var n: Int


def solve_task(ctx: Int, j: Int):
    var c = Pointer[Chol, MutUntrackedOrigin](unsafe_from_address=ctx)
    var n = c[].n
    var x = c[].b.unsafe_offset(j * n)
    for i in range(n):  # L y = b
        var s = x[unsafe_offset=i]
        for k in range(i):
            s -= c[].a[unsafe_offset=i * n + k] * x[unsafe_offset=k]
        x[unsafe_offset=i] = s / c[].a[unsafe_offset=i * n + i]
    for i in range(n - 1, -1, -1):  # L^T x = y
        var s = x[unsafe_offset=i]
        for k in range(i + 1, n):
            s -= c[].a[unsafe_offset=k * n + i] * x[unsafe_offset=k]
        x[unsafe_offset=i] = s / c[].a[unsafe_offset=i * n + i]


def ridge(A: List[Float32], B: List[Float32], n: Int, lam_rel: Float64, m: Int = -1) raises -> List[Float32]:
    """Solve (A + lam I) W = B for symmetric A (n x n) and B[o][t] = sum_n X[n][o] Y[n][t] (n x m outputs, m = n when -1); return M = W^T (m x n, row i = output i)."""
    var mm = m if m > 0 else n
    var tr: Float64 = 0
    for i in range(n):
        tr += Float64(A[i * n + i])
    var lam = lam_rel * tr / Float64(n)
    var L = List[Float64](length=n * n, fill=0)
    for i in range(n):
        for j in range(i + 1):
            var s = Float64(A[i * n + j]) + (lam if i == j else 0.0)
            for k in range(j):
                s -= L[i * n + k] * L[j * n + k]
            if i == j:
                if s <= 0:
                    raise Error("ridge: matrix not positive definite at " + String(i) + " (corpus too short for this hidden size?)")
                L[i * n + i] = sqrt(s)
            else:
                L[i * n + j] = s / L[j * n + j]
    var R = List[Float64](length=mm * n, fill=0)  # RHS j = column j of B: B[o][j] over o
    for j in range(mm):
        for o in range(n):
            R[j * n + o] = Float64(B[o * mm + j])
    var job: List[Chol] = [Chol(L.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), R.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), n)]
    parallelize(solve_task, Int(job.unsafe_ptr()), mm)
    _ = len(job) + len(L) + len(R)
    var M = List[Float32](length=mm * n, fill=0)  # W[o][j] = R[j][o]; M[i][o] = W[o][i] = R[i][o]
    for i in range(mm):
        for o in range(n):
            M[i * n + o] = Float32(R[i * n + o])
    return M^


def run() raises:
    var a = List[String]()
    var root = String(".work/neurons")
    var dump = String()
    var layers_arg = String()
    var windows = 24
    var holdout = 0
    var av = argv()
    var i = 1
    while i < len(av):
        var t = String(av[i])
        if t == "--out" and i + 1 < len(av):
            root = String(av[i + 1])
            i += 2
        elif t == "--dump" and i + 1 < len(av):
            dump = String(av[i + 1])
            i += 2
        elif t == "--layers" and i + 1 < len(av):
            layers_arg = String(av[i + 1])
            i += 2
        elif t == "--holdout" and i + 1 < len(av):
            holdout = Int(String(av[i + 1]))
            i += 2
        elif t == "--windows" and i + 1 < len(av):
            windows = Int(String(av[i + 1]))
            i += 2
        else:
            a.append(t)
            i += 1
    if len(a) < 2:
        raise Error("usage: rlens MODEL CORPUS [--layers a,b] [--windows N] [--holdout N] [--out DIR] [--dump DIR]")
    var g = Gguf(a[0])
    var arch = kv_str(g, "general.architecture")
    var nl = kv_int(g, arch + ".block_count")
    var last = nl - 1
    var layers = List[Int]()
    if layers_arg.byte_length() > 0:
        for p in layers_arg.split(","):
            layers.append(Int(String(p)))
    else:
        for l in range(nl - 1):
            layers.append(l)
    var rl = layers.copy()
    rl.append(last)
    var text = read_text(a[1])
    var step = 700  # characters per window, about 150 tokens
    var seq = List[Int]()  # windows of SEG tokens each, one batched forward
    comptime SEG = 128
    var w = 0
    var at = 0
    while w < windows and at + step <= text.byte_length():
        var ids = llama_ids(a[0], String(text[byte=at : at + step]), ".work/rlens-text.txt")
        at += step
        if len(ids) >= SEG:
            for j in range(SEG):
                seq.append(ids[j])
            w += 1
    var n_pos = len(seq)
    var r = forward(g, seq, List[Int](), rl, SEG)
    ref rr = r[1]
    var h = len(rr[0]) // n_pos
    var xs = List[List[Float32]]()
    for k in range(len(layers)):
        xs.append(rr[k].copy())
    var y = rr[len(layers)].copy()
    if n_pos < 2 * h:
        raise Error("only " + String(n_pos) + " positions for hidden " + String(h) + ": give a longer corpus or more windows (need >= 2 x hidden)")
    var n_fit = n_pos - holdout * SEG
    print("positions", n_pos, "fitted on", n_fit, "hidden", h, "layers", len(layers))
    var yt = List[Float32](length=h * n_fit, fill=0)
    for p in range(n_fit):
        for j in range(h):
            yt[j * n_fit + p] = y[p * h + j]
    var b = List[UInt8]()
    for c in String("MNRL").as_bytes():
        b.append(c)
    put_u32(b, 1)
    put_u32(b, h)
    put_u32(b, len(layers))
    for k in range(len(layers)):
        var xt = List[Float32](length=h * n_fit, fill=0)
        for p in range(n_fit):
            for j in range(h):
                xt[j * n_fit + p] = xs[k][p * h + j]
        var A = matmul(xt, xt, h, n_fit, h)  # Gram: y[t][o] = sum_n X[n][o] X[n][t]
        var B = matmul(xt, yt, h, n_fit, h)  # B[t][o] = sum_n Y[n][t] X[n][o]; transposed below to B[o][t]
        var Bo = List[Float32](length=h * h, fill=0)
        for o in range(h):
            for t in range(h):
                Bo[o * h + t] = B[t * h + o]
        var M = ridge(A, Bo, h, 1e-3)
        put_u32(b, layers[k])
        for v in M:
            put_f32(b, v)
        if dump.byte_length() > 0 and k == 0:
            makedirs(dump, exist_ok=True)
            var xb = List[UInt8]()
            for v in xs[0]:
                put_f32(xb, v)
            write_file(dump + "/x.f32", xb)
            var yb = List[UInt8]()
            for v in y:
                put_f32(yb, v)
            write_file(dump + "/y.f32", yb)
            var mb = List[UInt8]()
            put_u32(mb, h)
            put_u32(mb, n_pos)
            put_u32(mb, layers[0])
            put_u32(mb, n_fit)
            for v in M:
                put_f32(mb, v)
            write_file(dump + "/m.f32", mb)
    var parts = a[0].split("/")
    var base = String(parts[len(parts) - 1])
    var dot = base.rfind(".")
    var stem = (String(base[byte=0:dot]) if dot > 0 else base).lower()
    var dir = root + "/" + stem
    makedirs(dir, exist_ok=True)
    write_file(dir + "/rlens.bin", b)
    print("wrote", dir + "/rlens.bin")


def main():
    try:
        run()
    except e:
        print("FAIL rlens:", e)
        exit(1)
