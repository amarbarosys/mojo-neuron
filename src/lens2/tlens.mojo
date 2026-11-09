"""Tuned lens fit and eval (docs/prereg-tlensb.md). `nx tlens {gguf|model, cap, out, layers?="7,11,15,19,23,27", ntrain?=8000, nval?=800, nheld?=256, lr?}`:
per layer fits the translator u = r + s (A x + b) against the teacher (final residual, layer 31) on a restricted vocabulary S, picks the best epoch by validation KL,
then measures full-vocabulary KL(teacher || student) and top-1 agreement on held-out positions for logit lens, b-only and tuned. Writes <out>/L<l>.bin (A then b, f32),
<out>/result.json and <out>/log.txt. `-D TLENS_MUT=1` fits against shuffled teacher rows (the gate demands it to fail)."""
from std.math import sqrt, exp, log, isfinite
from std.sys import get_defined_int
from std.time import perf_counter_ns
from std.os import makedirs
from json import Json
from ggufx.model import Gguf
from mneuron.ggio import tensor, load_rows, write_atomic
from mneuron.fwd import matmul
from mneuron.rlog import Rlog
from mneuron.store import model_info
from probe.lie import i32s, cap_rows
from probe.concept import insert
from lens2.tcore import loss_grad, adam, translate, normed
from lens2.lensdiff import lens_multi, pairs_json
from mneuron.ggio import vocab
from probe.lie import last_recs

comptime MUT = get_defined_int["TLENS_MUT", 0]()


def xs(mut st: UInt64) -> UInt64:
    st ^= st << 13
    st ^= st >> 7
    st ^= st << 17
    return st


def shuffle(mut v: List[Int], mut st: UInt64):
    var i = len(v) - 1
    while i > 0:
        var j = Int(xs(st) % UInt64(i + 1))
        var t = v[i]
        v[i] = v[j]
        v[j] = t
        i -= 1


struct Fit(Movable):
    var A: List[Float32]
    var b: List[Float32]
    var train: Float64
    var val: Float64
    var epoch: Int

    def __init__(out self, h: Int):
        self.A = List[Float32](length=h * h, fill=0)
        self.b = List[Float32](length=h, fill=0)
        self.train = 0.0
        self.val = 0.0
        self.epoch = 0


def scale_rows(R: List[Float32], n: Int, h: Int, mut X: List[Float32], mut s: List[Float32]):
    for t in range(n):
        var ss: Float64 = 0
        for j in range(h):
            ss += Float64(R[t * h + j]) ** 2
        var r = Float32(sqrt(ss / Float64(h)))
        s[t] = r
        for j in range(h):
            X[t * h + j] = R[t * h + j] / r


def gather(R: List[Float32], X: List[Float32], s: List[Float32], Q: List[Float32], idx: List[Int], lo: Int, hi: Int, h: Int, S: Int,
           mut Rb: List[Float32], mut Xb: List[Float32], mut sb: List[Float32], mut Qb: List[Float32]):
    for k in range(lo, hi):
        var t = idx[k]
        var o = k - lo
        for j in range(h):
            Rb[o * h + j] = R[t * h + j]
            Xb[o * h + j] = X[t * h + j]
        sb[o] = s[t]
        for i in range(S):
            Qb[o * S + i] = Q[t * S + i]


def kl_over(A: List[Float32], b: List[Float32], R: List[Float32], X: List[Float32], s: List[Float32], Q: List[Float32], idx: List[Int], lo: Int, hi: Int,
            WS: List[Float32], WT: List[Float32], w: List[Float32], h: Int, S: Int, use_a: Bool) -> Float64:
    var tot: Float64 = 0
    var n = 0
    var k = lo
    var gA = List[Float32]()
    var gb = List[Float32](length=h, fill=0)
    while k < hi:
        var e = min(k + 64, hi)
        var B = e - k
        var Rb = List[Float32](length=B * h, fill=0)
        var Xb = List[Float32](length=B * h, fill=0)
        var sb = List[Float32](length=B, fill=0)
        var Qb = List[Float32](length=B * S, fill=0)
        gather(R, X, s, Q, idx, k, e, h, S, Rb, Xb, sb, Qb)
        tot += loss_grad(A, b, Rb, Xb, sb, Qb, WS, WT, w, B, h, S, use_a, gA, gb, False) * Float64(B)
        n += B
        k = e
    return tot / Float64(n)


def fit_layer(R: List[Float32], X: List[Float32], s: List[Float32], Q: List[Float32], tr: List[Int], nval: Int, WS: List[Float32], WT: List[Float32],
              w: List[Float32], h: Int, S: Int, use_a: Bool, lr_a: Float32, lr_b: Float32, epochs: Int, mut st: UInt64) raises -> Fit:
    """Train on tr[0 : len - nval], validate on the last nval of tr; the kept translator is the best of epoch 0 (logit lens) .. epochs by validation KL."""
    var f = Fit(h)
    var ntr = len(tr) - nval
    var order = List[Int]()
    for i in range(ntr):
        order.append(i)
    var A = List[Float32](length=h * h, fill=0)
    var b = List[Float32](length=h, fill=0)
    var mA = List[Float32](length=h * h, fill=0)
    var vA = List[Float32](length=h * h, fill=0)
    var mb = List[Float32](length=h, fill=0)
    var vb = List[Float32](length=h, fill=0)
    var gA = List[Float32](length=h * h, fill=0)
    var gb = List[Float32](length=h, fill=0)
    f.val = kl_over(A, b, R, X, s, Q, tr, ntr, len(tr), WS, WT, w, h, S, use_a)
    f.train = kl_over(A, b, R, X, s, Q, tr, 0, min(ntr, 512), WS, WT, w, h, S, use_a)
    var step = 0
    for ep in range(1, epochs + 1):
        shuffle(order, st)
        var shuf = List[Int]()
        for i in range(ntr):
            shuf.append(tr[order[i]])
        var run: Float64 = 0
        var nb = 0
        var k = 0
        while k + 64 <= ntr:
            var Rb = List[Float32](length=64 * h, fill=0)
            var Xb = List[Float32](length=64 * h, fill=0)
            var sb = List[Float32](length=64, fill=0)
            var Qb = List[Float32](length=64 * S, fill=0)
            gather(R, X, s, Q, shuf, k, k + 64, h, S, Rb, Xb, sb, Qb)
            var l = loss_grad(A, b, Rb, Xb, sb, Qb, WS, WT, w, 64, h, S, use_a, gA, gb, True)
            if not isfinite(l):
                raise Error("tlens: non-finite loss at epoch " + String(ep) + " batch " + String(nb))
            run += l
            nb += 1
            step += 1
            if use_a:
                adam(A, gA, mA, vA, lr_a, step)
            adam(b, gb, mb, vb, lr_b, step)
            k += 64
        var v = kl_over(A, b, R, X, s, Q, tr, ntr, len(tr), WS, WT, w, h, S, use_a)
        if v < f.val:
            f.val = v
            f.train = run / Float64(max(nb, 1))
            f.epoch = ep
            f.A = A.copy()
            f.b = b.copy()
    return f^


def load_all(g: Gguf, name: String, rowid: Int, n: Int) raises -> List[Float32]:
    return load_rows(g, tensor(g, name), rowid, n)


def read_map(path: String, h: Int, mut A: List[Float32], mut b: List[Float32]) raises:
    var raw = open(path, "r").read_bytes()
    if len(raw) != (h * h + h) * 4:
        raise Error("tlens: " + path + " is not f32[h*h + h]")
    var p = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(h * h):
        A[i] = p[unsafe_offset=i]
    for j in range(h):
        b[j] = p[unsafe_offset=h * h + j]


def lens_op(a: Json) raises -> Json:
    """op "lens" {gguf, cap, map, layer, req, back?=0, k?=10}: top k tokens of the logit lens and of the tuned lens (map = a fit's out dir) at the position `back`
    tokens before the last of request `req` (1-based)."""
    var g = Gguf(a.get("gguf").str())
    var h = g.tensors[tensor(g, "output_norm.weight")].dims[0]
    var cap = a.get("cap").str()
    var L = a.get("layer").int()
    var k = a.get("k").int() if a.has("k") else 10
    var recs = last_recs(cap)
    var req = a.get("req").int()
    if req < 1 or req > len(recs):
        raise Error("tlens: request " + String(req) + " outside 1.." + String(len(recs)))
    var back = a.get("back").int() if a.has("back") else 0
    var sel = List[Int]()
    sel.append(recs[req - 1] - back)
    var R = cap_rows(cap, L, sel, h)
    var X = List[Float32](length=h, fill=0)
    var sc = List[Float32](length=1, fill=0)
    scale_rows(R, 1, h, X, sc)
    var A = List[Float32](length=h * h, fill=0)
    var b = List[Float32](length=h, fill=0)
    read_map(a.get("map").str() + "/L" + String(L) + ".bin", h, A, b)
    var U = translate(A, b, R, X, sc, 1, h, True)
    var V = List[Float32](length=2 * h, fill=0)
    for j in range(h):
        V[j] = R[j]
        V[h + j] = U[j]
    var Ls = lens_multi(g, V, 2, h, k)
    var tok = vocab(g)
    var o = Json.object()
    var lj = Json.object()
    lj.set("top", pairs_json(tok, Ls.top[0], Ls.tsc[0]))
    var tj = Json.object()
    tj.set("top", pairs_json(tok, Ls.top[1], Ls.tsc[1]))
    o.set("logit_lens", lj)
    o.set("tuned_lens", tj)
    o.set("layer", Json.of(L))
    return o^


def tlens_call(a: Json) raises -> Json:
    if a.text("op") == "lens":
        return lens_op(a)
    var path = a.text("gguf")
    if path.byte_length() == 0:
        if not a.has("model"):
            raise Error("tlens: give 'gguf' (a path) or 'model' (a store name)")
        path = model_info(a.get("model").str()).text("gguf")
    for key in ["cap", "out"]:
        if not a.has(key):
            raise Error("tlens: missing argument '" + key + "'")
    var cap = a.get("cap").str()
    var out = a.get("out").str()
    makedirs(out, exist_ok=True)
    var layers = List[Int]()
    for p in (a.get("layers").str() if a.has("layers") else String("7,11,15,19,23,27")).split(","):
        layers.append(Int(String(p)))
    var ntrain = a.get("ntrain").int() if a.has("ntrain") else 8000
    var nval = a.get("nval").int() if a.has("nval") else 800
    var nheld = a.get("nheld").int() if a.has("nheld") else 256
    var epochs = a.get("epochs").int() if a.has("epochs") else 2
    var t0 = perf_counter_ns()
    var logtxt = String()
    var g = Gguf(path)
    var ni = tensor(g, "output_norm.weight")
    var h = g.tensors[ni].dims[0]
    var ui = tensor(g, "output.weight")
    if ui < 0:
        ui = tensor(g, "token_embd.weight")
    var V = g.tensors[ui].dims[1]
    var w = load_rows(g, ni, 0, 1)
    var req = i32s(cap + "/req.i32")
    var tok = i32s(cap + "/tokens.i32")
    var n = len(req)
    var cand_tr = List[Int]()
    var cand_he = List[Int]()
    var counts = List[Int](length=V, fill=0)
    for i in range(1, n):
        if req[i] != req[i - 1]:
            continue
        var has_next = i + 1 < n and req[i + 1] == req[i]
        if req[i] % 5 == 0:
            cand_he.append(i)
        else:
            cand_tr.append(i)
            if has_next:
                counts[tok[i + 1]] += 1
    var st: UInt64 = 0x9E3779B97F4A7C15
    shuffle(cand_tr, st)
    shuffle(cand_he, st)
    if len(cand_tr) < ntrain + nval or len(cand_he) < nheld:
        raise Error("tlens: capture has " + String(len(cand_tr)) + " train and " + String(len(cand_he)) + " held-out positions, asked " + String(ntrain + nval) + " and " + String(nheld))
    var recs = List[Int]()  # train, validation, then held-out records
    for i in range(ntrain + nval):
        recs.append(cand_tr[i])
    for i in range(nheld):
        recs.append(cand_he[i])
    var NT = ntrain + nval
    var T = NT + nheld
    # restricted vocabulary: the 4096 most frequent next tokens of the train requests
    var sid = List[Int]()
    var ssc = List[Float32]()
    for id in range(V):
        if counts[id] > 0:
            insert(sid, ssc, id, Float32(counts[id]), 4096, True)
    var S = len(sid)
    var WS = List[Float32](length=S * h, fill=0)
    var WT = List[Float32](length=h * S, fill=0)
    for i in range(S):
        var row = load_rows(g, ui, sid[i], 1)
        for j in range(h):
            WS[i * h + j] = row[j]
            WT[j * S + i] = row[j]
    logtxt += "S " + String(S) + " tokens, " + String(NT) + " train+val and " + String(nheld) + " held-out positions\n"
    # teacher: final residual (layer 31) through the same head
    var nl = Int(a.get("final").int()) if a.has("final") else 31
    var R31 = cap_rows(cap, nl, recs, h)
    var rho = List[Float32](length=T, fill=0)
    var NTch = normed(R31, w, T, h, rho)
    var Q = List[Float32](length=NT * S, fill=0)
    var t = 0
    while t < NT:
        var e = min(t + 512, NT)
        var Nb = List[Float32](length=(e - t) * h, fill=0)
        for k in range((e - t) * h):
            Nb[k] = NTch[t * h + k]
        var Z = matmul(WS, Nb, S, h, e - t)
        for r in range(e - t):
            var mx = Z[r * S]
            for i in range(S):
                mx = max(mx, Z[r * S + i])
            var zs: Float64 = 0
            for i in range(S):
                zs += exp(Float64(Z[r * S + i] - mx))
            for i in range(S):
                Q[(t + r) * S + i] = Float32(exp(Float64(Z[r * S + i] - mx)) / zs)
        t = e
    if MUT == 1:  # fit against the teacher of the next position
        var Q2 = Q.copy()
        for r in range(NT):
            var src = (r + 1) % NT
            for i in range(S):
                Q2[r * S + i] = Q[src * S + i]
        Q = Q2^
    var tr = List[Int]()
    for i in range(NT):
        tr.append(i)
    # held-out teacher over the full vocabulary, stored: pass 1 over W_U in 4096-row chunks
    var NH = List[Float32](length=nheld * h, fill=0)
    for k in range(nheld * h):
        NH[k] = NTch[NT * h + k]
    var TZ = List[Float32](length=nheld * V, fill=0)
    var lseq = List[Float64](length=nheld, fill=0)
    var r0 = 0
    var tm = List[Float32](length=nheld, fill=-1e30)
    var tzs = List[Float64](length=nheld, fill=0)
    while r0 < V:
        var m = min(4096, V - r0)
        var Z = matmul(load_rows(g, ui, r0, m), NH, m, h, nheld)
        for r in range(nheld):
            var cm = Z[r * m]
            for i in range(m):
                cm = max(cm, Z[r * m + i])
                TZ[r * V + r0 + i] = Z[r * m + i]
            if cm > tm[r]:
                tzs[r] = tzs[r] * exp(Float64(tm[r] - cm))
                tm[r] = cm
            for i in range(m):
                tzs[r] += exp(Float64(Z[r * m + i] - tm[r]))
        r0 += m
    var eq = List[Float64](length=nheld, fill=0)  # E_q[z_q]
    var top1q = List[Int](length=nheld, fill=0)
    var acc_text = 0
    var n_text = 0
    var cover: Float64 = 0
    var in_s = List[Bool](length=V, fill=False)
    for i in range(S):
        in_s[sid[i]] = True
    for r in range(nheld):
        lseq[r] = Float64(tm[r]) + log(tzs[r])
        var best = 0
        for i in range(V):
            var qi = exp(Float64(TZ[r * V + i]) - lseq[r])
            eq[r] += qi * Float64(TZ[r * V + i])
            if in_s[i]:
                cover += qi
            if TZ[r * V + i] > TZ[r * V + best]:
                best = i
        top1q[r] = best
        var rec = recs[NT + r]
        if rec + 1 < n and req[rec + 1] == req[rec]:
            n_text += 1
            if tok[rec + 1] == best:
                acc_text += 1
    cover /= Float64(nheld)
    logtxt += "teacher top-1 equals the next text token on " + String(acc_text) + " of " + String(n_text) + " held-out positions; S covers " + String(cover) + " of teacher mass\n"
    # per layer: fit lr by validation at layer 19 (if lr not given), then every layer
    var lr_a: Float32 = 0
    var chosen = a.has("lr")
    if chosen:
        lr_a = Float32(a.get("lr").num())
    var sel_layer = layers[0]
    for li in range(len(layers)):
        if layers[li] == 19:
            sel_layer = 19
    var students = List[List[Float32]]()  # normalised head inputs, held-out, per student
    var names = List[String]()
    students.append(NH.copy())  # layer 31 logit lens: same hidden state, same code
    names.append("31:lens")
    var info = List[String]()
    for li in range(len(layers)):
        var L = layers[li]
        var Rl = cap_rows(cap, L, recs, h)
        var Xl = List[Float32](length=T * h, fill=0)
        var sl = List[Float32](length=T, fill=0)
        scale_rows(Rl, T, h, Xl, sl)
        if (not chosen) and L == sel_layer:
            var best_val = 1e30
            for lr in [Float32(3e-6), Float32(1e-5), Float32(3e-5)]:
                var s2 = st
                var ft = fit_layer(Rl, Xl, sl, Q, tr, nval, WS, WT, w, h, S, True, lr, 0.01, epochs, s2)
                logtxt += "L" + String(L) + " lr " + String(lr) + ": best epoch " + String(ft.epoch) + " val KL " + String(ft.val) + " (logit-lens-only epoch 0 kept when 0)\n"
                if ft.val < best_val:
                    best_val = ft.val
                    lr_a = lr
            chosen = True
        if not chosen:
            lr_a = 1e-5  # layers fitted before 19 in a custom list: the grid middle, said in the log
        var s3 = st
        var ft = fit_layer(Rl, Xl, sl, Q, tr, nval, WS, WT, w, h, S, True, lr_a, 0.01, epochs, s3)
        var s4 = st
        var fb = fit_layer(Rl, Xl, sl, Q, tr, nval, WS, WT, w, h, S, False, 0, 0.01, epochs, s4)
        logtxt += "L" + String(L) + " tuned epoch " + String(ft.epoch) + " train " + String(ft.train) + " val " + String(ft.val) + " | b-only epoch " + String(fb.epoch) + " val " + String(fb.val) + " (" + String(Float64(perf_counter_ns() - t0) / 1e9) + " s)\n"
        write_atomic(out + "/log.txt", List[UInt8](logtxt.as_bytes()))
        var raw = List[UInt8]()
        var pa = ft.A.unsafe_ptr().bitcast[UInt8]()
        for i in range(h * h * 4):
            raw.append(pa[i])
        var pb = ft.b.unsafe_ptr().bitcast[UInt8]()
        for i in range(h * 4):
            raw.append(pb[i])
        write_atomic(out + "/L" + String(L) + ".bin", raw)
        # held-out head inputs
        var Rh = List[Float32](length=nheld * h, fill=0)
        var Xh = List[Float32](length=nheld * h, fill=0)
        var sh = List[Float32](length=nheld, fill=0)
        for k in range(nheld * h):
            Rh[k] = Rl[NT * h + k]
            Xh[k] = Xl[NT * h + k]
        for k in range(nheld):
            sh[k] = sl[NT + k]
        var rr = List[Float32](length=nheld, fill=0)
        students.append(normed(Rh, w, nheld, h, rr))
        names.append(String(L) + ":lens")
        students.append(normed(translate(fb.A, fb.b, Rh, Xh, sh, nheld, h, False), w, nheld, h, rr))
        names.append(String(L) + ":bonly")
        students.append(normed(translate(ft.A, ft.b, Rh, Xh, sh, nheld, h, True), w, nheld, h, rr))
        names.append(String(L) + ":tuned")
        info.append(String(L) + "," + String(ft.train) + "," + String(ft.val) + "," + String(fb.val) + "," + String(ft.epoch) + "," + String(fb.epoch))
    # pass 2 over W_U: all students
    var NS = len(students)
    var sm = List[Float32](length=NS * nheld, fill=-1e30)
    var szs = List[Float64](length=NS * nheld, fill=0)
    var sq = List[Float64](length=NS * nheld, fill=0)  # sum q_i z_p_i
    var sarg = List[Int](length=NS * nheld, fill=0)
    var sbest = List[Float32](length=NS * nheld, fill=-1e30)
    var SN = List[Float32](length=NS * nheld * h, fill=0)
    for si in range(NS):
        for k in range(nheld * h):
            SN[si * nheld * h + k] = students[si][k]
    r0 = 0
    while r0 < V:
        var m = min(4096, V - r0)
        var rows = load_rows(g, ui, r0, m)
        var Z = matmul(rows, SN, m, h, NS * nheld)  # (NS * nheld) x m
        var qc = List[Float64](length=nheld * m, fill=0)
        for r in range(nheld):
            for i in range(m):
                qc[r * m + i] = exp(Float64(TZ[r * V + r0 + i]) - lseq[r])
        for si in range(NS):
            for r in range(nheld):
                var o = (si * nheld + r) * m
                var cm = Z[o]
                for i in range(m):
                    cm = max(cm, Z[o + i])
                var ix = si * nheld + r
                if cm > sm[ix]:
                    szs[ix] = szs[ix] * exp(Float64(sm[ix] - cm))
                    sm[ix] = cm
                var acc: Float64 = 0
                var zz: Float64 = 0
                for i in range(m):
                    var z = Z[o + i]
                    zz += exp(Float64(z - sm[ix]))
                    acc += qc[r * m + i] * Float64(z)
                    if z > sbest[ix]:
                        sbest[ix] = z
                        sarg[ix] = r0 + i
                szs[ix] += zz
                sq[ix] += acc
        r0 += m
    var kls = List[Float64](length=NS, fill=0)
    var agree = List[Float64](length=NS, fill=0)
    for si in range(NS):
        for r in range(nheld):
            var ix = si * nheld + r
            var lsep = Float64(sm[ix]) + log(szs[ix])
            kls[si] += (eq[r] - lseq[r] - sq[ix] + lsep) / Float64(nheld)
            if sarg[ix] == top1q[r]:
                agree[si] += 1.0 / Float64(nheld)
    var res = Json.object()
    res.set("teacher_top1_text_acc", Json.of(Float64(acc_text) / Float64(max(n_text, 1))))
    res.set("s_size", Json.of(S))
    res.set("s_coverage", Json.of(cover))
    res.set("identity_kl_l31", Json.of(kls[0]))
    res.set("lr_a", Json.of(Float64(lr_a)))
    res.set("mut", Json.of(MUT))
    var lj = Json.array()
    var rl = Rlog("tlens", cap, path)
    for li in range(len(layers)):
        var e = Json.object()
        var p = info[li].split(",")
        e.set("layer", Json.of(layers[li]))
        e.set("kl_lens", Json.of(kls[1 + 3 * li]))
        e.set("kl_bonly", Json.of(kls[2 + 3 * li]))
        e.set("kl_tuned", Json.of(kls[3 + 3 * li]))
        e.set("top1_lens", Json.of(agree[1 + 3 * li]))
        e.set("top1_bonly", Json.of(agree[2 + 3 * li]))
        e.set("top1_tuned", Json.of(agree[3 + 3 * li]))
        e.set("train_kl_tuned", Json.of(atof(String(p[1]))))
        e.set("val_kl_tuned", Json.of(atof(String(p[2]))))
        e.set("val_kl_bonly", Json.of(atof(String(p[3]))))
        e.set("epoch_tuned", Json.of(Int(String(p[4]))))
        lj.push(e)
        var vals = List[Float64]()
        vals.append(kls[1 + 3 * li])
        vals.append(kls[3 + 3 * li])
        rl.done("tlens L" + String(layers[li]) + " held-out KL lens " + String(kls[1 + 3 * li]) + " b-only " + String(kls[2 + 3 * li]) + " tuned " + String(kls[3 + 3 * li]),
                out + "/L" + String(layers[li]) + ".bin", vals, 0.0, 1e6)
    res.set("layers", lj)
    res.set("wall_s", Json.of(Float64(perf_counter_ns() - t0) / 1e9))
    var js = res.dumps(False)
    write_atomic(out + "/result.json", List[UInt8](js.as_bytes()))
    return res^
