"""Lie_steer prep|eval: the steering gate of docs/prereg-f24.md (bars 3 and 4).
  prep CAP LABELS PROBES OUT   picks Ls (best validation layer that has a captured layer above) and Lm = the next captured layer, writes
                               OUT/{plus,minus,rand0..2}.f32 (f32[h], alpha = 2 * class-mean projection gap at Ls), OUT/req.jsonl (150 test
                               statements x 6 conditions, per-request steer_file) and OUT/prep.json.
  eval CAP LABELS PROBES STEERCAP OUT   reads the steered capture, scores at Lm with the Lm probe, checks the bars, ends with an Rlog line."""
from std.math import sqrt, log, cos, isfinite
from std.sys import argv, exit
from json import Json, parse
from mneuron.rlog import Rlog
from mneuron.store import read_text
from probe.lie import last_recs, cap_rows, i32s, load_probe, Probe, part, pick

comptime NSEL = 150
comptime CONDS = 6  # 0 base, 1 plus, 2 minus, 3..5 random directions
comptime ALPHA_GAPS: Float64 = 2.0


def put_f32s(path: String, v: List[Float32]) raises:
    var raw = List[UInt8](length=len(v) * 4, fill=0)
    var src = v.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(len(v) * 4):
        raw[i] = src[unsafe_offset=i]
    with open(path, "w") as f:
        f.write_bytes(raw)


def unit_of(w: List[Float32]) -> List[Float32]:
    var s: Float64 = 0
    for j in range(len(w)):
        s += Float64(w[j]) ** 2
    var o = List[Float32](length=len(w), fill=0)
    for j in range(len(w)):
        o[j] = w[j] / Float32(sqrt(s))
    return o^


def gauss(mut s: UInt64) -> Float32:
    s = s * 6364136223846793005 + 1442695040888963407
    var u1 = (Float64(Int(s >> 11)) + 1.0) / 9007199254740993.0
    s = s * 6364136223846793005 + 1442695040888963407
    var u2 = Float64(Int(s >> 11)) / 9007199254740992.0
    return Float32(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2))


def dotrow(w: List[Float32], x: List[Float32], row: Int, h: Int) -> Float64:
    var s: Float64 = 0
    for j in range(h):
        s += Float64(w[j]) * Float64(x[row * h + j])
    return s


struct Ctx(Movable):
    var h: Int
    var layers: List[Int]
    var y: List[Int]
    var pairs: List[Int]
    var recs: List[Int]

    def __init__(out self, cap: String, labels: String) raises:
        var meta = parse(read_text(cap + "/meta.json"))
        self.h = meta.get("h").int()
        self.layers = List[Int]()
        for l in meta.get("layers").items():
            self.layers.append(l.int())
        self.y = List[Int]()
        self.pairs = List[Int]()
        for line in read_text(labels).split("\n"):
            if line.byte_length() == 0:
                continue
            var f = line.split("\t")
            self.y.append(Int(f[0]))
            self.pairs.append(Int(f[1]))
        self.recs = last_recs(cap)
        if len(self.recs) != len(self.y):
            raise Error("lie_steer: " + String(len(self.recs)) + " requests in the capture, " + String(len(self.y)) + " labels")

    def parts(self) -> List[Int]:
        var p = List[Int](length=len(self.y), fill=0)
        for i in range(len(self.y)):
            p[i] = part(self.pairs[i])
        return p^


def prep(cap: String, labels: String, probes: String, odir: String) raises:
    var c = Ctx(cap, labels)
    var h = c.h
    var fit = parse(read_text(probes + "/fit.json"))
    var li = -1
    var best = -1.0
    var per = fit.get("layers").items()
    for i in range(len(c.layers) - 1):  # a layer above must exist
        var v = per[i].get("val").num()
        if v > best:
            best = v
            li = i
    var Ls = c.layers[li]
    var Lm = c.layers[li + 1]
    var w = unit_of(load_probe(probes + "/L" + String(Ls) + ".probe").w)
    var x = cap_rows(cap, Ls, c.recs, h)
    var parts = c.parts()
    var tr = pick(parts, 0)
    var sp: Float64 = 0
    var sn: Float64 = 0
    var np_ = 0
    var nn = 0
    for i in tr:
        var z = dotrow(w, x, i, h)
        if c.y[i] == 1:
            sp += z
            np_ += 1
        else:
            sn += z
            nn += 1
    var gap = sp / Float64(np_) - sn / Float64(nn)
    if gap <= 0:
        raise Error("lie_steer: class gap " + String(gap) + " at L" + String(Ls) + " is not positive")
    var alpha = ALPHA_GAPS * gap
    var vecs = List[List[Float32]]()
    var plus = List[Float32](length=h, fill=0)
    var minus = List[Float32](length=h, fill=0)
    for j in range(h):
        plus[j] = w[j] * Float32(alpha)
        minus[j] = -plus[j]
    vecs.append(plus^)
    vecs.append(minus^)
    for k in range(3):
        var s = UInt64(k + 1) * 0x9E3779B97F4A7C15
        var r = List[Float32](length=h, fill=0)
        for j in range(h):
            r[j] = gauss(s)
        var u = unit_of(r)
        for j in range(h):
            u[j] *= Float32(alpha)
        vecs.append(u^)
    var names = List[String]()
    names.append("plus")
    names.append("minus")
    names.append("rand0")
    names.append("rand1")
    names.append("rand2")
    for i in range(5):
        put_f32s(odir + "/" + names[i] + ".f32", vecs[i])
    var te = pick(parts, 2)
    var tok = i32s(cap + "/tokens.i32")
    var req = String()
    var sel = Json.array()
    for cond in range(CONDS):
        for j in range(NSEL):
            var r = te[j]
            var a0 = c.recs[r - 1] + 1 if r > 0 else 0
            var ids = String()
            for t in range(a0, c.recs[r] + 1):
                ids += ("," if t > a0 else "") + String(tok[t])
            req += "{\"id\":" + String(cond * NSEL + j + 1) + ",\"prompt\":[" + ids + "],\"n\":1"
            if cond > 0:
                req += ",\"steer_file\":\"" + odir + "/" + names[cond - 1] + ".f32\",\"steer_layer\":" + String(Ls)
            req += "}\n"
            if cond == 0:
                sel.push(Json.of(r))
    with open(odir + "/req.jsonl", "w") as f:
        f.write(req)
    var o = Json.object()
    o.set("Ls", Json.of(Ls))
    o.set("Lm", Json.of(Lm))
    o.set("alpha", Json.of(alpha))
    o.set("gap", Json.of(gap))
    o.set("sel", sel)
    with open(odir + "/prep.json", "w") as f:
        f.write(o.dumps(False))
    print("prep: Ls", Ls, "Lm", Lm, "gap", gap, "alpha", alpha, "->", odir + "/req.jsonl")


def evaluate(cap: String, labels: String, probes: String, scap: String, odir: String) raises:
    var c = Ctx(cap, labels)
    var h = c.h
    var pj = parse(read_text(odir + "/prep.json"))
    var Ls = pj.get("Ls").int()
    var Lm = pj.get("Lm").int()
    var alpha = pj.get("alpha").num()
    var sel = List[Int]()
    for s in pj.get("sel").items():
        sel.append(s.int())
    var srecs = last_recs(scap)
    if len(srecs) != CONDS * NSEL:
        raise Error("lie_steer: steered capture holds " + String(len(srecs)) + " requests, want " + String(CONDS * NSEL))
    var pm = load_probe(probes + "/L" + String(Lm) + ".probe")
    var ws = unit_of(load_probe(probes + "/L" + String(Ls) + ".probe").w)
    var xm = cap_rows(scap, Lm, srecs, h)
    var xs = cap_rows(scap, Ls, srecs, h)
    var mrecs = List[Int]()
    for j in range(NSEL):
        mrecs.append(c.recs[sel[j]])
    var xmain = cap_rows(cap, Lm, mrecs, h)
    var rl = Rlog("lie_steer", scap, "")
    var finite = True
    var score = List[Float64](length=CONDS * NSEL, fill=0)
    for i in range(CONDS * NSEL):
        score[i] = Float64(pm.score(xm, i, h))
        if not isfinite(score[i]):
            finite = False
    var gt: Float64 = 0
    var gf: Float64 = 0
    var nt = 0
    var nf = 0
    for j in range(NSEL):
        if c.y[sel[j]] == 1:
            gt += score[j]
            nt += 1
        else:
            gf += score[j]
            nf += 1
    var G = gt / Float64(max(nt, 1)) - gf / Float64(max(nf, 1))
    var mean = List[Float64](length=CONDS, fill=0)
    var agree = List[Float64](length=CONDS, fill=0)
    for cond in range(1, CONDS):
        for j in range(NSEL):
            var d = score[cond * NSEL + j] - score[j]
            mean[cond] += d / Float64(NSEL)
            var up = d > 0
            if (cond == 2 and not up) or (cond != 2 and up):
                agree[cond] += 1.0 / Float64(NSEL)
    # bar 4: the unsteered base in the steering process vs the main capture
    var bad4 = 0
    for j in range(NSEL):
        var mx: Float32 = 0
        var df: Float32 = 0
        for k in range(h):
            var a = xm[j * h + k]
            var b = xmain[j * h + k]
            mx = max(mx, abs(b))
            df = max(df, abs(a - b))
        if df > 1e-2 * mx:
            bad4 += 1
    # applied: the steer shows at Ls as +alpha along ws
    var applied: Float64 = 0
    for j in range(NSEL):
        applied += (dotrow(ws, xs, NSEL + j, h) - dotrow(ws, xs, j, h)) / Float64(NSEL)
    var b3a = G > 0 and mean[1] >= 0.25 * G and mean[2] <= -0.25 * G
    var b3b = agree[1] >= 0.8 and agree[2] >= 0.8
    var b3c = True
    for k in range(3, CONDS):
        if abs(mean[k]) > 0.5 * mean[1]:
            b3c = False
    var b3e = abs(applied - alpha) <= 0.05 * alpha
    var b4 = bad4 == 0
    var res = Json.object()
    res.set("G", Json.of(G))
    res.set("alpha", Json.of(alpha))
    res.set("applied_at_Ls", Json.of(applied))
    res.set("mean_shift_plus", Json.of(mean[1]))
    res.set("mean_shift_minus", Json.of(mean[2]))
    res.set("sign_agree_plus", Json.of(agree[1]))
    res.set("sign_agree_minus", Json.of(agree[2]))
    res.set("mean_shift_rand", Json.of(abs(mean[3]) + abs(mean[4]) + abs(mean[5])))
    res.set("base_rows_off", Json.of(bad4))
    res.set("bar3_shift", Json.of(b3a))
    res.set("bar3_sign", Json.of(b3b))
    res.set("bar3_control", Json.of(b3c))
    res.set("bar3_finite", Json.of(finite))
    res.set("bar3_applied", Json.of(b3e))
    res.set("bar4", Json.of(b4))
    with open(odir + "/steer.json", "w") as f:
        f.write(res.dumps(False))
    var msg = "Ls " + String(Ls) + " Lm " + String(Lm) + " G " + String(G) + " shift+ " + String(mean[1]) + " shift- " + String(mean[2]) + " agree " + String(agree[1]) + "/" + String(agree[2]) + " rand " + String(mean[3]) + "," + String(mean[4]) + "," + String(mean[5]) + " applied " + String(applied) + "/" + String(alpha) + " base off " + String(bad4)
    var ok = b3a and b3b and b3c and finite and b3e and b4
    print(msg)
    print("bars: shift", b3a, "sign", b3b, "control", b3c, "finite", finite, "applied", b3e, "base", b4)
    if ok:
        var v = List[Float64]()
        v.append(G)
        v.append(mean[1])
        v.append(mean[2])
        rl.done(msg, "", v, -1e4, 1e4)
    else:
        rl.event("anomaly", "prereg F24 steering bar missed: " + msg)
        exit(2)


def main() raises:
    var a = argv()
    if len(a) >= 6 and String(a[1]) == "prep":
        prep(String(a[2]), String(a[3]), String(a[4]), String(a[5]))
    elif len(a) >= 7 and String(a[1]) == "eval":
        evaluate(String(a[2]), String(a[3]), String(a[4]), String(a[5]), String(a[6]))
    else:
        print("usage: lie_steer prep CAP LABELS PROBES OUT | eval CAP LABELS PROBES STEERCAP OUT")
        exit(1)
