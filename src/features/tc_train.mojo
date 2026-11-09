"""tc_train DATA LAYER OUT NF K B STEPS LR [--seed S] [--bar F] [--fp32] [--eval N] [--warm W] [--init-var T] [--init-ridge FILE] [--cos] [--b2 B] [--abort X] [--ntrain N] [--hold F] [--eval2 DIR2] [--stats FILE]: train a TopK transcoder on a capture v1 (DATA/L<LAYER>.f16 records [x|y|..],
req.i32, meta.json; docs/format-capture.md). The last 10 % of the records, cut at a request boundary, are held out; ymean and vtot
(the FVU denominator, per-token total variance of y) come from one pass over the training records. Batches of B consecutive rows in seeded random order, lr warmup 50 steps then linear decay to 10 %.
Writes OUT/tc.bin (f32: We din x NF and be with the input centering and scale folded in, so it reads raw x; Wd NF x dout, bd), OUT/feat.f32 (held-out per feature: fire count, activation sum), OUT/run.json.
Prints `step train_fvu` per tenth, then `HELDOUT fvu dead mean_active`. --bar F: held-out FVU above F logs an rlog anomaly."""
from std.sys import argv, exit
from std.os import makedirs
from std.time import perf_counter_ns
from std.random import seed, random_float64
from std.math import sqrt, cos
from std.memory import bitcast
from std.os.path import exists
from json import parse
from max.gpu.host import DeviceContext
from ops import F, THREADS, grid
from tc import *
from mneuron.ggio import write_file, put_f32
from mneuron.store import read_text
from mneuron.rlog import Rlog


def feed(ctx: DeviceContext, tc: Tc, st: DeviceBuffer[DType.float16], path: String, stride: Int, row0: Int, xm: Buf, sx: Float32) raises:
    """Records row0.. of L<l>.f16 (f16, [x|y|..] per token) into tc.x / tc.y."""
    var bytes: List[UInt8]
    with open(path, "r") as f:
        _ = f.seek(row0 * stride * 2)
        bytes = f.read_bytes(tc.rows * stride * 2)
    ctx.enqueue_copy(dst_buf=st, src_ptr=bytes.unsafe_ptr().unsafe_bitcast[Float16]())
    ctx.synchronize()
    _ = len(bytes)
    tc.load_xy(ctx, st, stride, xm, sx)


def eval_set(ctx: DeviceContext, tc: Tc, st: DeviceBuffer[DType.float16], path: String, stride: Int, row0: Int, nb: Int, xm: Buf, sx: Float32, vtot: Float32) raises -> Float64:
    """Held-out FVU of nb batches starting at record row0 of another capture file, with this run's normalisation."""
    var e: Float64 = 0
    for b in range(nb):
        feed(ctx, tc, st, path, stride, row0 + b * tc.rows, xm, sx)
        tc.forward(ctx)
        e += tc.sse(ctx)
    return e / (Float64(nb * tc.rows) * Float64(vtot))


def i32s(path: String) raises -> List[Int]:
    var b = open(path, "r").read_bytes()
    var r = List[Int](length=len(b) // 4, fill=0)
    for k in range(len(r)):
        r[k] = Int(b.unsafe_ptr().unsafe_bitcast[Int32]()[k])
    return r^


def main() raises:
    var a = List[String]()
    var sd = 1
    var bar = 1e30
    var fast = True
    var eval_every = 0
    var warm = 50.0
    var init_var = 0.0
    var init_ridge = String()
    var cosine = False
    var beta2 = 0.999
    var abort_above = 0.0
    var ntrain = 0
    var hold = 0.10
    var stats_path = String()
    var eval2 = String()
    var av = argv()
    var i = 1
    while i < len(av):
        var t = String(av[i])
        if t == "--seed" and i + 1 < len(av):
            sd = Int(String(av[i + 1]))
            i += 2
        elif t == "--eval" and i + 1 < len(av):
            eval_every = Int(String(av[i + 1]))
            i += 2
        elif t == "--stats" and i + 1 < len(av):
            stats_path = String(av[i + 1])
            i += 2
        elif t == "--hold" and i + 1 < len(av):
            hold = Float64(String(av[i + 1]))
            i += 2
        elif t == "--eval2" and i + 1 < len(av):
            eval2 = String(av[i + 1])
            i += 2
        elif t == "--ntrain" and i + 1 < len(av):
            ntrain = Int(String(av[i + 1]))
            i += 2
        elif t == "--cos":
            cosine = True
            i += 1
        elif t == "--b2" and i + 1 < len(av):
            beta2 = Float64(String(av[i + 1]))
            i += 2
        elif t == "--abort" and i + 1 < len(av):
            abort_above = Float64(String(av[i + 1]))
            i += 2
        elif t == "--init-ridge" and i + 1 < len(av):
            init_ridge = String(av[i + 1])
            i += 2
        elif t == "--init-var" and i + 1 < len(av):
            init_var = Float64(String(av[i + 1]))
            i += 2
        elif t == "--warm" and i + 1 < len(av):
            warm = Float64(String(av[i + 1]))
            i += 2
        elif t == "--fp32":
            fast = False
            i += 1
        elif t == "--bar" and i + 1 < len(av):
            bar = Float64(String(av[i + 1]))
            i += 2
        else:
            a.append(t)
            i += 1
    if len(a) < 8:
        print("usage: tc_train DATA LAYER OUT NF K B STEPS LR [--seed S] [--bar F]")
        exit(1)
    var data = a[0]
    var out = a[2]
    var nf = Int(a[3])
    var k = Int(a[4])
    var B = Int(a[5])
    var steps = Int(a[6])
    var lr = Float64(a[7])
    var meta = parse(read_text(data + "/meta.json"))
    var d = meta.get("h").int()
    var stride = meta.text("fields").byte_length() * d
    var dx = data + "/L" + a[1] + ".f16"
    var req = i32s(data + "/req.i32")
    var n = len(req)
    var hs = Int(Float64(n) * (1.0 - hold))
    while hs < n and req[hs] == req[hs - 1]:
        hs += 1
    var ntr = hs // B
    var ntrn = min(ntr, ntrain // B) if ntrain > 0 else ntr  # data-scaling runs: only the first ntrain records are trained on; ymean, vtot and the input scale stay those of the full train split so held-out FVU is comparable
    var nho = (n - hs) // B
    if ntr < 1 or nho < 1:
        raise Error("too few tokens (" + String(n) + ") for batch " + String(B) + " with a 10 % holdout")
    var dx2 = String()
    var hs2 = 0
    var nho2 = 0
    var stride2 = stride
    if eval2.byte_length() > 0:  # a second capture's held-out split (the last 10 % of its records, cut at a request boundary), scored with this run's normalisation
        var req2 = i32s(eval2 + "/req.i32")
        var n2 = len(req2)
        hs2 = n2 * 9 // 10
        while hs2 < n2 and req2[hs2] == req2[hs2 - 1]:
            hs2 += 1
        nho2 = (n2 - hs2) // B
        dx2 = eval2 + "/L" + a[1] + ".f16"
        stride2 = parse(read_text(eval2 + "/meta.json")).text("fields").byte_length() * d
        if nho2 < 1:
            raise Error("--eval2: no held-out batch in " + eval2)
    var rl = Rlog("tc_train", String(" ").join(a), meta.text("model") if meta.text("model").startswith("/") else "")
    makedirs(out, exist_ok=True)
    seed(sd)
    var ctx = DeviceContext()
    # pass over the training records: mean and total variance of y
    var sy = List[Float64](length=d, fill=0)
    var sy2 = List[Float64](length=d, fill=0)
    var sx1 = List[Float64](length=d, fill=0)
    var sxx: Float64 = 0
    var r0 = 0
    var cached = False
    if stats_path.byte_length() > 0 and exists(stats_path):  # the sums of the pass below, kept so each run of a sweep skips the read of the whole train split
        var cb = open(stats_path, "r").read_bytes()
        if len(cb) == (3 * d + 1) * 8:
            var cp = cb.unsafe_ptr().unsafe_bitcast[Float64]()
            for j in range(d):
                sy[j] = cp[j]
                sy2[j] = cp[d + j]
                sx1[j] = cp[2 * d + j]
            sxx = cp[3 * d]
            cached = True
            r0 = ntr * B
            print("statistics from the cache", stats_path)
        _ = len(cb)
    while r0 < ntr * B:
        var cn = min(2048, ntr * B - r0)
        var raw: List[UInt8]
        with open(dx, "r") as f:
            _ = f.seek(r0 * stride * 2)
            raw = f.read_bytes(cn * stride * 2)
        var hp = raw.unsafe_ptr().unsafe_bitcast[Float16]()
        for t in range(cn):
            for j in range(d):
                var xv = Float64(hp[t * stride + j])
                sx1[j] += xv
                sxx += xv * xv
            for j in range(d):
                var v = Float64(hp[t * stride + d + j])
                sy[j] += v
                sy2[j] += v * v
        _ = len(raw)
        r0 += cn
    if stats_path.byte_length() > 0 and not cached:
        var cb = List[UInt8]()
        for j in range(3 * d + 1):
            var v = sy[j] if j < d else (sy2[j - d] if j < 2 * d else (sx1[j - 2 * d] if j < 3 * d else sxx))
            var u = bitcast[DType.uint64](v)
            for q in range(8):
                cb.append(UInt8(Int((u >> UInt64(8 * q)) & 255)))
        write_file(stats_path, cb)
    var vt: Float64 = 0
    var ym = List[Float32](length=d, fill=0)
    for j in range(d):
        var mu = sy[j] / Float64(ntr * B)
        vt += sy2[j] / Float64(ntr * B) - mu * mu
        ym[j] = Float32(mu)
    var vtot = Float32(vt)
    var xm = List[Float32](length=d, fill=0)  # inputs become (x - mean) * sx with E|.|^2 = d; both folded into We and be on save
    var mm: Float64 = 0
    for j in range(d):
        var mu = sx1[j] / Float64(ntr * B)
        xm[j] = Float32(mu)
        mm += mu * mu
    var sx = Float32(sqrt(Float64(d) / (sxx / Float64(ntr * B) - mm)))
    var tc = Tc(ctx, d, d, nf, k, B, vtot, fast)
    tc.b2 = beta2
    var wd = List[Float32](length=nf * d, fill=0)
    for f in range(nf):
        var s: Float64 = 0
        for j in range(d):
            var v = random_float64(-1, 1)
            wd[f * d + j] = Float32(v)
            s += v * v
        for j in range(d):
            wd[f * d + j] = Float32(Float64(wd[f * d + j]) / sqrt(s))
    var we = List[Float32](length=d * nf, fill=0)
    for f in range(nf):
        for j in range(d):
            we[j * nf + f] = wd[f * d + j]
    tc.we.put(ctx, we)
    tc.wd.put(ctx, wd)
    tc.bd.put(ctx, ym)
    if init_ridge.byte_length() > 0:  # warm start: feature 2c / 2c+1 = relu(+/- (x_c - mean) |col_c|), decoder +/- unit column c of the ridge matrix (K = inf reproduces the ridge)
        if nf != 2 * d:
            raise Error("--init-ridge needs NF = 2 * hidden = " + String(2 * d))
        var rbm = open(init_ridge, "r").read_bytes()
        var rp = rbm.unsafe_ptr().unsafe_bitcast[Float32]()
        for q in range(len(we)):
            we[q] = 0
        for q in range(len(wd)):
            wd[q] = 0
        for c in range(d):
            var nrm: Float64 = 0
            for i in range(d):
                nrm += Float64(rp[i * d + c]) ** 2
            nrm = sqrt(nrm)
            for q in range(2):
                var sg = Float32(1.0 if q == 0 else -1.0)
                we[c * nf + 2 * c + q] = sg * Float32(nrm) / sx
                for i in range(d):
                    wd[(2 * c + q) * d + i] = sg * rp[i * d + c] / Float32(nrm)
        _ = len(rbm)
        tc.we.put(ctx, we)
        tc.wd.put(ctx, wd)
        print("init from the ridge matrix", init_ridge)
    var xmb = Buf(ctx, d, False)
    xmb.put(ctx, xm)
    var st = ctx.enqueue_create_buffer[DType.float16](B * stride)
    if init_var > 0:  # rescale We so that the reconstruction part of the first batch has variance init_var * vtot (z, a and y_hat - bd scale linearly with We)
        feed(ctx, tc, st, dx, stride, 0, xmb, sx)
        tc.forward(ctx)
        var yh0 = tc.yh.get(ctx)
        var m0: Float64 = 0
        for r in range(B):
            for j in range(d):
                m0 += (Float64(yh0[r * d + j]) - Float64(ym[j])) ** 2 / Float64(B)
        var alpha = Float32(sqrt(init_var * Float64(vtot) / m0))
        for q in range(len(we)):
            we[q] = we[q] * alpha
        tc.we.put(ctx, we)
        print("init rescale of We:", alpha)
    var order = List[Int]()
    for j in range(ntrn):
        order.append(j)
    for j in range(ntrn - 1, 0, -1):
        var r = Int(random_float64(0, Float64(j + 1)))
        r = min(r, j)
        var tmp = order[j]
        order[j] = order[r]
        order[r] = tmp
    var t0 = perf_counter_ns()
    var log = String()
    var every = max(1, steps // 10)
    var acc: Float64 = 0
    var accn = 0
    for s in range(steps):
        feed(ctx, tc, st, dx, stride, order[s % ntrn] * B, xmb, sx)
        tc.grads(ctx)
        var decay = (0.1 + 0.45 * (1.0 + cos(3.141592653589793 * Float64(s) / Float64(steps)))) if cosine else (1.0 - 0.9 * Float64(s) / Float64(steps))
        var rate = lr * min(1.0, Float64(s + 1) / warm) * decay
        if s % every == every - 1 or s == 0:
            acc = tc.sse(ctx) / (Float64(B) * Float64(vtot))
            print(s + 1, acc)
        tc.update(ctx, Float32(rate))
        if eval_every > 0 and s % eval_every == eval_every - 1:  # held-out trace: fvu and the share of features silent on the held-out rows
            var c2 = Buf(ctx, nf)
            var m2 = Buf(ctx, nf)
            var e2: Float64 = 0
            for b in range(nho):
                feed(ctx, tc, st, dx, stride, hs + b * B, xmb, sx)
                tc.forward(ctx)
                e2 += tc.sse(ctx)
                ctx.enqueue_function[stat_k](tc.a.p(), c2.p(), m2.p(), Int32(B), Int32(nf), grid_dim=grid(nf), block_dim=THREADS)
            var h2 = c2.get(ctx)
            var d2 = 0
            for f in range(nf):
                if h2[f] == 0:
                    d2 += 1
            var fv = e2 / (Float64(nho * B) * Float64(vtot))
            var f2s = String()
            if nho2 > 0:
                f2s = " eval2 " + String(eval_set(ctx, tc, st, dx2, stride2, hs2, nho2, xmb, sx, vtot))
            print("EVAL", s + 1, fv, "dead", Float64(d2) / Float64(nf), f2s)
            if abort_above > 0 and s >= 1499 and fv > abort_above:
                print("ABORT: held-out FVU", fv, "above", abort_above, "at step", s + 1, "(diverged); stopping the run")
                steps = s + 1
                break
    ctx.synchronize()
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    # held-out pass
    var sse: Float64 = 0
    var cnt = Buf(ctx, nf)
    var sm = Buf(ctx, nf)
    for b in range(nho):
        feed(ctx, tc, st, dx, stride, hs + b * B, xmb, sx)
        tc.forward(ctx)
        sse += tc.sse(ctx)
        ctx.enqueue_function[stat_k](tc.a.p(), cnt.p(), sm.p(), Int32(B), Int32(nf), grid_dim=grid(nf), block_dim=THREADS)
    var fvu = sse / (Float64(nho * B) * Float64(vtot))
    var hc = cnt.get(ctx)
    var dead = 0
    var fires: Float64 = 0
    for f in range(nf):
        if hc[f] == 0:
            dead += 1
        fires += Float64(hc[f])
    var mean_active = fires / Float64(nho * B)
    var bytes = List[UInt8]()
    var ps = List[List[Float32]]()
    var wes = tc.we.get(ctx)
    var bes = tc.be.get(ctx)
    for q in range(len(wes)):
        wes[q] = wes[q] * sx
    for j in range(d):
        for f in range(nf):
            bes[f] -= xm[j] * wes[j * nf + f]
    ps.append(wes^)
    ps.append(bes^)
    ps.append(tc.wd.get(ctx))
    ps.append(tc.bd.get(ctx))
    for q in range(4):
        for v in ps[q]:
            put_f32(bytes, v)
    write_file(out + "/tc.bin", bytes)
    var fb = List[UInt8]()
    for v in hc:
        put_f32(fb, v)
    for v in sm.get(ctx):
        put_f32(fb, v)
    write_file(out + "/feat.f32", fb)
    var res = "fvu=" + String(fvu) + " dead=" + String(Float64(dead) / Float64(nf)) + " mean_active=" + String(mean_active) + " xscale=" + String(sx) + " tokens=" + String(ntrn * B) + " heldout_tokens=" + String(nho * B) + " vtot=" + String(vtot) + " steps=" + String(steps) + " step_ms=" + String(ms / Float64(steps)) + " nf=" + String(nf) + " k=" + String(k) + " B=" + String(B) + " lr=" + String(lr) + " seed=" + String(sd)
    var rj = List[UInt8]()
    for c in ('{"d":' + String(d) + ',"nf":' + String(nf) + ',"k":' + String(k) + ',"heldout_fvu":' + String(fvu) + ',"dead_frac":' + String(Float64(dead) / Float64(nf)) + ',"mean_active":' + String(mean_active) + ',"step_ms":' + String(ms / Float64(steps)) + "}\n").as_bytes():
        rj.append(c)
    write_file(out + "/run.json", rj)
    var vals = List[Float64]()
    vals.append(fvu)
    rl.done(res, out + "/run.json", vals, 0.0, 1.5)
    if fvu > bar:
        rl.event("anomaly", "prereg bar missed: held-out fvu " + String(fvu) + " > " + String(bar) + " | " + res, out + "/run.json")
    if nho2 > 0:
        print("HELDOUT2", eval_set(ctx, tc, st, dx2, stride2, hs2, nho2, xmb, sx, vtot), "rows", nho2 * B, "(second capture, this run's normalisation)")
    print("HELDOUT", fvu, "dead", Float64(dead) / Float64(nf), "mean_active", mean_active, "step_ms", ms / Float64(steps))
    _ = (tc.we.n, Int(st.unsafe_ptr()))
