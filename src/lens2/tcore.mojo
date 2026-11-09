"""Tuned lens core (docs/prereg-tlensb.md): translator u = r + s * (A x + b), x = r / s, s = rms(r); student logits z = W_S rmsnorm(u) * w.
loss_grad returns KL(q || softmax(z)) averaged over the batch and fills the gradients of A (h x h) and b (h). Shapes: R, X (B x h), s (B), q (B x S, rows sum to 1),
WS (S x h), WT (h x S, the transpose), w (h). Everything f32."""
from std.math import sqrt, exp, log
from std.sys import get_defined_int
from ggufx.par import parallelize
from mneuron.fwd import matmul

comptime MUT = get_defined_int["TCORE_MUT", 0]()  # 1: the rmsnorm backward drops its mean term (the oracle must catch it)


@fieldwise_init
struct AD(Copyable):
    var a: Pointer[Float32, MutUntrackedOrigin]
    var g: Pointer[Float32, ImmUntrackedOrigin]
    var m: Pointer[Float32, MutUntrackedOrigin]
    var v: Pointer[Float32, MutUntrackedOrigin]
    var n: Int
    var lr: Float32
    var c1: Float32
    var c2: Float32


def ad_task(ctx: Int, i: Int):
    var c = Pointer[AD, ImmUntrackedOrigin](unsafe_from_address=ctx)
    var lo = i * 65536
    var hi = min(lo + 65536, c[].n)
    for k in range(lo, hi):
        var g = c[].g[unsafe_offset=k]
        var m = 0.9 * c[].m[unsafe_offset=k] + 0.1 * g
        var v = 0.999 * c[].v[unsafe_offset=k] + 0.001 * g * g
        c[].m[unsafe_offset=k] = m
        c[].v[unsafe_offset=k] = v
        c[].a[unsafe_offset=k] = c[].a[unsafe_offset=k] - c[].lr * (m * c[].c1) / (sqrt(v * c[].c2) + 1e-8)


def adam(mut a: List[Float32], g: List[Float32], mut m: List[Float32], mut v: List[Float32], lr: Float32, step: Int):
    """One Adam step (b1 0.9, b2 0.999) over all of `a`, parallel over 64k-element blocks; step counts from 1."""
    var c1 = Float32(1.0 / (1.0 - 0.9 ** Float64(step)))
    var c2 = Float32(1.0 / (1.0 - 0.999 ** Float64(step)))
    var job: List[AD] = [AD(a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), g.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
                            m.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), v.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(a), lr, c1, c2)]
    parallelize(ad_task, Int(job.unsafe_ptr()), (len(a) + 65535) // 65536)
    _ = len(job) + len(a) + len(g) + len(m) + len(v)


def translate(A: List[Float32], b: List[Float32], R: List[Float32], X: List[Float32], s: List[Float32], B: Int, h: Int, use_a: Bool) -> List[Float32]:
    """u = r + s (A x + b)."""
    var U = R.copy()
    if use_a:
        var AX = matmul(A, X, h, h, B)
        for t in range(B):
            for j in range(h):
                U[t * h + j] += s[t] * (AX[t * h + j] + b[j])
    else:
        for t in range(B):
            for j in range(h):
                U[t * h + j] += s[t] * b[j]
    return U^


def normed(U: List[Float32], w: List[Float32], B: Int, h: Int, mut rho: List[Float32]) -> List[Float32]:
    """rmsnorm(u) * w, the rho of every row kept for the backward pass."""
    var N = List[Float32](length=B * h, fill=0)
    for t in range(B):
        var ss: Float64 = 0
        for j in range(h):
            ss += Float64(U[t * h + j]) * Float64(U[t * h + j])
        var r = Float32(1.0 / sqrt(ss / Float64(h) + 1e-6))
        rho[t] = r
        for j in range(h):
            N[t * h + j] = U[t * h + j] * r * w[j]
    return N^


def loss_grad(A: List[Float32], b: List[Float32], R: List[Float32], X: List[Float32], s: List[Float32], q: List[Float32], WS: List[Float32], WT: List[Float32],
              w: List[Float32], B: Int, h: Int, S: Int, use_a: Bool, mut gA: List[Float32], mut gb: List[Float32], grad: Bool) -> Float64:
    var U = translate(A, b, R, X, s, B, h, use_a)
    var rho = List[Float32](length=B, fill=0)
    var N = normed(U, w, B, h, rho)
    var Z = matmul(WS, N, S, h, B)
    var loss: Float64 = 0
    for t in range(B):
        var mx = Z[t * S]
        for i in range(S):
            mx = max(mx, Z[t * S + i])
        var zs: Float64 = 0
        for i in range(S):
            zs += exp(Float64(Z[t * S + i] - mx))
        var lse = Float64(mx) + log(zs)
        for i in range(S):
            var qi = Float64(q[t * S + i])
            var lp = Float64(Z[t * S + i]) - lse
            if qi > 0:
                loss += qi * (log(qi) - lp)
            Z[t * S + i] = Float32((exp(lp) - qi) / Float64(B))  # dz in place
    loss /= Float64(B)
    if not grad:
        return loss
    var dN = matmul(WT, Z, h, S, B)
    var dAXT = List[Float32](length=h * B, fill=0)
    var XT = List[Float32](length=h * B, fill=0)
    for j in range(h):
        gb[j] = 0
    for t in range(B):
        var c: Float64 = 0
        for k in range(h):
            c += Float64(w[k]) * Float64(dN[t * h + k]) * Float64(U[t * h + k])
        var r = Float64(rho[t])
        var cr = Float32(r * r * r * c / Float64(h))
        for j in range(h):
            var du = Float32(r) * w[j] * dN[t * h + j] - (U[t * h + j] * cr if MUT == 0 else Float32(0))
            var d = s[t] * du
            gb[j] += d
            dAXT[j * B + t] = d
            XT[j * B + t] = X[t * h + j]
    if use_a:
        var y = matmul(dAXT, XT, h, B, h)  # y[j * h + o] = gA[o][j]
        for o in range(h):
            for j in range(h):
                gA[o * h + j] = y[j * h + o]
    return loss
