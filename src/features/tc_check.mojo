"""Gate: 3 GPU training steps of the transcoder against an f64 CPU reference (z, a, all four gradients, all parameters after each
Adam step), max |err| <= 1e-4 * max|ref| per tensor. Row 0 of x is zero (no positive z: the < K branch). Prints PASS or FAIL <what>."""
from std.sys import exit
from std.random import seed, random_float64
from std.math import sqrt
from max.gpu.host import DeviceContext
from tc import *

comptime T = 64
comptime DI = 24
comptime DO = 20
comptime NF = 96
comptime K = 8


def rnd(n: Int, s: Float64) -> List[Float32]:
    var h = List[Float32](length=n, fill=0)
    for i in range(n):
        h[i] = Float32(random_float64(-s, s))
    return h^


def unit_rows(mut w: List[Float32], rows: Int, d: Int):
    for r in range(rows):
        var s: Float64 = 0
        for j in range(d):
            s += Float64(w[r * d + j]) * Float64(w[r * d + j])
        for j in range(d):
            w[r * d + j] = Float32(Float64(w[r * d + j]) / sqrt(s))


def kth_bits(z: List[Float64], r: Int, k: Int) -> Int32:
    """Bit pattern of the K-th largest positive z of row r in f32 (1 if fewer than K positives)."""
    var bits = List[Int32]()
    for j in range(NF):
        var b = bitcast[DType.int32, 1](Float32(z[r * NF + j]))
        if b >= 1:
            bits.append(b)
    if len(bits) <= k:
        return 1
    sort(bits)
    return bits[len(bits) - k]


def cmp(name: String, got: List[Float32], want: List[Float64], mut ok: Bool):
    var mx: Float64 = 0
    var er: Float64 = 0
    for i in range(len(want)):
        mx = max(mx, abs(want[i]))
        er = max(er, abs(Float64(got[i]) - want[i]))
    if er > 1e-4 * max(mx, 1e-12):
        print("FAIL " + name + ": max err " + String(er) + " vs max|ref| " + String(mx))
        ok = False


def main() raises:
    seed(7)
    var ctx = DeviceContext()
    var vtot = Float32(3.0)
    var x = rnd(T * DI, 1.0)
    for j in range(DI):
        x[j] = 0
    var y = rnd(T * DO, 1.0)
    var we = rnd(DI * NF, 0.3)
    var wd = rnd(NF * DO, 1.0)
    unit_rows(wd, NF, DO)
    var tc = Tc(ctx, DI, DO, NF, K, T, vtot)
    tc.x.put(ctx, x)
    tc.y.put(ctx, y)
    tc.we.put(ctx, we)
    tc.wd.put(ctx, wd)
    # f64 reference state
    var pwe = List[Float64]()
    var pbe = List[Float64](length=NF, fill=0)
    var pwd = List[Float64]()
    var pbd = List[Float64](length=DO, fill=0)
    for v in we:
        pwe.append(Float64(v))
    for v in wd:
        pwd.append(Float64(v))
    var mm = List[List[Float64]]()
    var vv = List[List[Float64]]()
    for n in [DI * NF, NF, NF * DO, DO]:
        mm.append(List[Float64](length=n, fill=0))
        vv.append(List[Float64](length=n, fill=0))
    var ok = True
    for step in range(1, 4):
        var tag = "step " + String(step) + " "
        tc.grads(ctx)
        var z = List[Float64](length=T * NF, fill=0)
        var a = List[Float64](length=T * NF, fill=0)
        for r in range(T):
            for f in range(NF):
                var s = pbe[f]
                for i in range(DI):
                    s += Float64(x[r * DI + i]) * pwe[i * NF + f]
                z[r * NF + f] = Float64(Float32(s))
        for r in range(T):
            var t = kth_bits(z, r, K)
            for f in range(NF):
                var b = bitcast[DType.int32, 1](Float32(z[r * NF + f]))
                a[r * NF + f] = z[r * NF + f] if b >= t and b >= 1 else 0.0
        var dy = List[Float64](length=T * DO, fill=0)
        for r in range(T):
            for o in range(DO):
                var s = pbd[o]
                for f in range(NF):
                    s += a[r * NF + f] * pwd[f * DO + o]
                dy[r * DO + o] = (s - Float64(y[r * DO + o])) * 2.0 / (Float64(T) * Float64(vtot))
        var gwd = List[Float64](length=NF * DO, fill=0)
        var gbd = List[Float64](length=DO, fill=0)
        var gwe = List[Float64](length=DI * NF, fill=0)
        var gbe = List[Float64](length=NF, fill=0)
        for r in range(T):
            for o in range(DO):
                gbd[o] += dy[r * DO + o]
                for f in range(NF):
                    gwd[f * DO + o] += a[r * NF + f] * dy[r * DO + o]
            for f in range(NF):
                if a[r * NF + f] > 0:
                    var da: Float64 = 0
                    for o in range(DO):
                        da += dy[r * DO + o] * pwd[f * DO + o]
                    gbe[f] += da
                    for i in range(DI):
                        gwe[i * NF + f] += Float64(x[r * DI + i]) * da
        cmp(tag + "a", tc.a.get(ctx), a, ok)
        cmp(tag + "gwd", tc.gwd.get(ctx), gwd, ok)
        cmp(tag + "gbd", tc.gbd.get(ctx), gbd, ok)
        cmp(tag + "gwe", tc.gwe.get(ctx), gwe, ok)
        cmp(tag + "gbe", tc.gbe.get(ctx), gbe, ok)
        tc.update(ctx, 1e-2)
        var c1 = 1.0 / (1.0 - 0.9 ** Float64(step))
        var c2 = 1.0 / (1.0 - 0.999 ** Float64(step))
        var gs = List[List[Float64]]()
        gs.append(gwe.copy())
        gs.append(gbe.copy())
        gs.append(gwd.copy())
        gs.append(gbd.copy())
        for q in range(4):
            ref g = gs[q]
            for i in range(len(g)):
                mm[q][i] = 0.9 * mm[q][i] + 0.1 * g[i]
                vv[q][i] = 0.999 * vv[q][i] + 0.001 * g[i] * g[i]
                var d = 1e-2 * (mm[q][i] * c1) / (sqrt(vv[q][i] * c2) + 1e-8)
                if q == 0:
                    pwe[i] -= d
                elif q == 1:
                    pbe[i] -= d
                elif q == 2:
                    pwd[i] -= d
                else:
                    pbd[i] -= d
        for f in range(NF):
            var s: Float64 = 0
            for o in range(DO):
                s += pwd[f * DO + o] * pwd[f * DO + o]
            for o in range(DO):
                pwd[f * DO + o] /= sqrt(s)
        cmp(tag + "we", tc.we.get(ctx), pwe, ok)
        cmp(tag + "be", tc.be.get(ctx), pbe, ok)
        cmp(tag + "wd", tc.wd.get(ctx), pwd, ok)
        cmp(tag + "bd", tc.bd.get(ctx), pbd, ok)
    # fp16 WMMA GEMM path against the fp32 path: same parameters, gradient cosine >= 0.995 and loss within 1 %
    comptime T2 = 256
    comptime D2 = 128
    comptime F2 = 512
    var x2 = rnd(T2 * D2, 1.0)
    var y2 = rnd(T2 * D2, 1.0)
    var we2 = rnd(D2 * F2, 0.3)
    var wd2 = rnd(F2 * D2, 1.0)
    unit_rows(wd2, F2, D2)
    var gs2 = List[List[Float32]]()
    var ls2 = List[Float64]()
    for q in range(2):
        var t2 = Tc(ctx, D2, D2, F2, 16, T2, vtot, q == 1)
        t2.x.put(ctx, x2)
        t2.y.put(ctx, y2)
        t2.we.put(ctx, we2)
        t2.wd.put(ctx, wd2)
        t2.grads(ctx)
        ls2.append(t2.sse(ctx))
        gs2.append(t2.gwe.get(ctx))
        gs2.append(t2.gwd.get(ctx))
    for q in range(2):
        var dot: Float64 = 0
        var na: Float64 = 0
        var nb: Float64 = 0
        for i in range(len(gs2[q])):
            dot += Float64(gs2[q][i]) * Float64(gs2[2 + q][i])
            na += Float64(gs2[q][i]) ** 2
            nb += Float64(gs2[2 + q][i]) ** 2
        var cs = dot / sqrt(na * nb)
        if cs < 0.995:
            print("FAIL wmma gradient " + String(q) + ": cosine " + String(cs))
            ok = False
    if abs(ls2[1] - ls2[0]) > 0.01 * ls2[0]:
        print("FAIL wmma loss: " + String(ls2[1]) + " vs " + String(ls2[0]))
        ok = False
    if not ok:
        exit(1)
    print("PASS tc step parity (3 steps, z/a/grads/params vs f64; WMMA path cosine >= 0.995)")
