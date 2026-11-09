"""Gradient oracle (docs/prereg-tlensb.md bar 4): analytic gradients of A and b equal central finite differences on tiny dims (h 16, S 24, batch 5).
Forward is f32, so the step is 1e-2 and the error is relative to the larger of |analytic|, |numeric| and 1e-3 of the largest gradient entry."""
from std.math import sqrt
from std.sys import exit
from lens2.tcore import loss_grad


def rnd(mut st: UInt64) -> Float32:
    st ^= st << 13
    st ^= st >> 7
    st ^= st << 17
    return Float32(Int(st >> 11) % 20001) / 10000.0 - 1.0


def main():
    var h = 16
    var S = 24
    var B = 5
    var st: UInt64 = 0x9E3779B97F4A7C15
    var R = List[Float32](length=B * h, fill=0)
    var X = List[Float32](length=B * h, fill=0)
    var s = List[Float32](length=B, fill=0)
    for t in range(B):
        var ss: Float32 = 0
        for j in range(h):
            R[t * h + j] = rnd(st) * 3.0
            ss += R[t * h + j] * R[t * h + j]
        s[t] = sqrt(ss / Float32(h))
        for j in range(h):
            X[t * h + j] = R[t * h + j] / s[t]
    var WS = List[Float32](length=S * h, fill=0)
    var WT = List[Float32](length=h * S, fill=0)
    for i in range(S):
        for j in range(h):
            WS[i * h + j] = rnd(st) * 0.5
            WT[j * S + i] = WS[i * h + j]
    var w = List[Float32](length=h, fill=0)
    for j in range(h):
        w[j] = 0.5 + 0.5 * (rnd(st) + 1.0)
    var q = List[Float32](length=B * S, fill=0)
    for t in range(B):
        var tot: Float32 = 0
        for i in range(S):
            q[t * S + i] = (rnd(st) + 1.0) * (rnd(st) + 1.0)
            tot += q[t * S + i]
        for i in range(S):
            q[t * S + i] /= tot
    var A = List[Float32](length=h * h, fill=0)
    var b = List[Float32](length=h, fill=0)
    for i in range(h * h):
        A[i] = rnd(st) * 0.05
    for j in range(h):
        b[j] = rnd(st) * 0.05
    var gA = List[Float32](length=h * h, fill=0)
    var gb = List[Float32](length=h, fill=0)
    _ = loss_grad(A, b, R, X, s, q, WS, WT, w, B, h, S, True, gA, gb, True)
    var eps: Float64 = 3e-3
    var dummyA = List[Float32](length=h * h, fill=0)
    var dummyb = List[Float32](length=h, fill=0)
    var worst: Float64 = 0
    # directional derivatives (central, f64 differences of the loss): along the gradient itself, along its A part, its b part, and 3 random directions.
    for k in range(6):
        var dA = List[Float32](length=h * h, fill=0)
        var db = List[Float32](length=h, fill=0)
        if k == 0:
            dA = gA.copy()
            db = gb.copy()
        elif k == 1:
            dA = gA.copy()
        elif k == 2:
            db = gb.copy()
        else:
            for i in range(h * h):
                dA[i] = rnd(st)
            for j in range(h):
                db[j] = rnd(st)
        var ana: Float64 = 0
        var nd: Float64 = 0
        var ng: Float64 = 0
        for i in range(h * h):
            ana += Float64(gA[i]) * Float64(dA[i])
            nd += Float64(dA[i]) ** 2
            ng += Float64(gA[i]) ** 2
        for j in range(h):
            ana += Float64(gb[j]) * Float64(db[j])
            nd += Float64(db[j]) ** 2
            ng += Float64(gb[j]) ** 2
        var Ap = A.copy()
        var bp = b.copy()
        var Am = A.copy()
        var bm = b.copy()
        for i in range(h * h):
            Ap[i] += Float32(eps) * dA[i]
            Am[i] -= Float32(eps) * dA[i]
        for j in range(h):
            bp[j] += Float32(eps) * db[j]
            bm[j] -= Float32(eps) * db[j]
        var lp = loss_grad(Ap, bp, R, X, s, q, WS, WT, w, B, h, S, True, dummyA, dummyb, False)
        var lm = loss_grad(Am, bm, R, X, s, q, WS, WT, w, B, h, S, True, dummyA, dummyb, False)
        var num = (lp - lm) / (2.0 * eps)
        var rel = abs(ana - num) / (sqrt(nd) * sqrt(ng))
        print("  direction", k, "analytic", ana, "numeric", num, "error / (|d| |g|)", rel)
        worst = max(worst, rel)
    print("tcore_check: worst error relative to |d| |g|:", worst)
    if worst > 1e-3:
        print("FAIL tcore_check: relative error above 1e-3")
        exit(1)
    print("PASS tcore_check")
