"""CAST request-level condition gate (docs/prereg-cast.md). `nx cast {op: fit|gate|score}`:
fit   {cap, out, layers?, train_pairs?=41, nctl?=20}: capture = French prompts 1..P then their English twins P+1..2P (P = pairs, last-token residual, field r). Condition
      direction d_l = unit(mean(French) - mean(English)) on train pairs, score = cos(r - mu, d), (layer, tau) by 5-fold CV over train pairs (folds by pair),
      test pairs scored once; also a shuffled-label control and the threshold width. Writes <out>/fit.json, d.f32, mu.f32.
gate  {fit, cap, req, persona, out}: the persona steer vector (f32 file) when the score of request `req` (1-based) fires, exactly zeros of the same size when not.
score {fit, cap, reqs: "a-b"}: how many requests of a capture fire (out-of-domain false-fire rate).
`-D CAST_MUT=1` inverts the comparison."""
from std.math import sqrt
from std.sys import get_defined_int
from std.os import makedirs
from json import Json, parse
from mneuron.ggio import write_atomic
from mneuron.rlog import Rlog
from mneuron.store import read_text
from probe.lie import last_recs, cap_rows

comptime MUT = get_defined_int["CAST_MUT", 0]()


def fires(score: Float64, tau: Float64) -> Bool:
    return (score < tau) if MUT == 1 else (score >= tau)


def xs(mut st: UInt64) -> UInt64:
    st ^= st << 13
    st ^= st >> 7
    st ^= st << 17
    return st


def fit_dir(R: List[Float32], prompts: List[Int], lab: List[Int], h: Int, mut mu: List[Float32], mut d: List[Float32]):
    """mu = mean residual of `prompts`; d = unit(mean of label 1 - mean of label 0). lab is indexed by prompt."""
    var n1 = 0
    var n0 = 0
    var m1 = List[Float64](length=h, fill=0)
    var m0 = List[Float64](length=h, fill=0)
    for p in prompts:
        for j in range(h):
            if lab[p] == 1:
                m1[j] += Float64(R[p * h + j])
            else:
                m0[j] += Float64(R[p * h + j])
        if lab[p] == 1:
            n1 += 1
        else:
            n0 += 1
    var nrm: Float64 = 0
    for j in range(h):
        mu[j] = Float32((m1[j] + m0[j]) / Float64(n1 + n0))
        var v = m1[j] / Float64(max(n1, 1)) - m0[j] / Float64(max(n0, 1))
        m1[j] = v
        nrm += v * v
    nrm = sqrt(nrm)
    for j in range(h):
        d[j] = Float32(m1[j] / max(nrm, 1e-30))


def cosine(R: List[Float32], p: Int, h: Int, mu: List[Float32], d: List[Float32]) -> Float64:
    var dot: Float64 = 0
    var nr: Float64 = 0
    for j in range(h):
        var x = Float64(R[p * h + j]) - Float64(mu[j])
        dot += x * Float64(d[j])
        nr += x * x
    return dot / max(sqrt(nr), 1e-30)


def best_tau(sc: List[Float64], y: List[Int]) -> Tuple[Float64, Float64]:
    """(tau, balanced accuracy) on a grid of 41 values between the lowest and highest score; ties keep the lower tau."""
    var lo = sc[0]
    var hi = sc[0]
    var n1 = 0
    for i in range(len(sc)):
        lo = min(lo, sc[i])
        hi = max(hi, sc[i])
        n1 += y[i]
    var n0 = len(sc) - n1
    var bt = lo
    var bb: Float64 = -1
    for k in range(41):
        var tau = lo + (hi - lo) * Float64(k) / 40.0
        var tp = 0
        var tn = 0
        for i in range(len(sc)):
            var f = fires(sc[i], tau)
            if y[i] == 1 and f:
                tp += 1
            if y[i] == 0 and not f:
                tn += 1
        var ba = 0.5 * (Float64(tp) / Float64(max(n1, 1)) + Float64(tn) / Float64(max(n0, 1)))
        if ba > bb:
            bb = ba
            bt = tau
    return (bt, bb)


struct Sel(Movable):
    var layer: Int
    var tau: Float64
    var cv: Float64
    var cvs: List[Float64]  # CV balanced accuracy per layer
    var taus: List[Float64]
    var lo: Float64  # lowest and highest out-of-fold score at the chosen layer
    var hi: Float64

    def __init__(out self):
        self.layer = 0
        self.tau = 0.0
        self.cv = -1.0
        self.cvs = List[Float64]()
        self.taus = List[Float64]()
        self.lo = 0.0
        self.hi = 0.0


def select(RL: List[List[Float32]], layers: List[Int], h: Int, trp: List[Int], lab: List[Int], npairs: Int) -> Sel:
    """Layer and tau by 5-fold CV balanced accuracy over the train prompts (folds by pair), ties -> lower layer."""
    var s = Sel()
    for li in range(len(layers)):
        var oof = List[Float64]()
        var yy = List[Int]()
        var order = List[Int]()
        for f in range(5):
            var fp = List[Int]()
            var hp = List[Int]()
            for p in trp:
                var pair = p if p < npairs else p - npairs
                if pair % 5 == f:
                    hp.append(p)
                else:
                    fp.append(p)
            var mu = List[Float32](length=h, fill=0)
            var d = List[Float32](length=h, fill=0)
            fit_dir(RL[li], fp, lab, h, mu, d)
            for p in hp:
                oof.append(cosine(RL[li], p, h, mu, d))
                yy.append(lab[p])
        var bt = best_tau(oof, yy)
        s.cvs.append(bt[1])
        s.taus.append(bt[0])
        if bt[1] > s.cv:
            s.cv = bt[1]
            s.tau = bt[0]
            s.layer = li
            var lo = oof[0]
            var hi = oof[0]
            for v in oof:
                lo = min(lo, v)
                hi = max(hi, v)
            s.lo = lo
            s.hi = hi
    return s^


def test_scores(R: List[Float32], h: Int, trp: List[Int], lab: List[Int], tep: List[Int], mut mu: List[Float32], mut d: List[Float32]) -> List[Float64]:
    fit_dir(R, trp, lab, h, mu, d)
    var o = List[Float64]()
    for p in tep:
        o.append(cosine(R, p, h, mu, d))
    return o^


def acc_at(sc: List[Float64], y: List[Int], tau: Float64) -> Float64:
    var ok = 0
    for i in range(len(sc)):
        if fires(sc[i], tau) == (y[i] == 1):
            ok += 1
    return Float64(ok) / Float64(len(sc))


def auc(sc: List[Float64], y: List[Int]) -> Float64:
    var w: Float64 = 0
    var n = 0
    for i in range(len(sc)):
        for j in range(len(sc)):
            if y[i] == 1 and y[j] == 0:
                n += 1
                w += 1.0 if sc[i] > sc[j] else (0.5 if sc[i] == sc[j] else 0.0)
    return w / Float64(max(n, 1))


def f32file(path: String) raises -> List[Float32]:
    var raw = open(path, "r").read_bytes()
    var v = List[Float32](length=len(raw) // 4, fill=0)
    var p = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(len(v)):
        v[i] = p[unsafe_offset=i]
    return v^


def wf32(path: String, v: List[Float32]) raises:
    var raw = List[UInt8]()
    var p = v.unsafe_ptr().bitcast[UInt8]()
    for i in range(len(v) * 4):
        raw.append(p[i])
    write_atomic(path, raw)


def fit_call(a: Json) raises -> Json:
    var cap = a.get("cap").str()
    var out = a.get("out").str()
    var meta = parse(read_text(cap + "/meta.json"))
    var h = meta.get("h").int()
    var layers = List[Int]()
    for x in meta.get("layers").items():
        layers.append(x.int())
    if a.has("layers"):
        layers = List[Int]()
        for p in a.get("layers").str().split(","):
            layers.append(Int(String(p)))
    var recs = last_recs(cap)
    var P = len(recs) // 2
    var ntr = a.get("train_pairs").int() if a.has("train_pairs") else 41
    var nctl = a.get("nctl").int() if a.has("nctl") else 20
    if len(recs) != 2 * P or ntr >= P:
        raise Error("cast: capture has " + String(len(recs)) + " requests; want French 1..P then English P+1..2P and train_pairs < P")
    makedirs(out, exist_ok=True)
    var rl = Rlog("cast", cap + " layers " + String(len(layers)), "")
    var RL = List[List[Float32]]()
    for L in layers:
        RL.append(cap_rows(cap, L, recs, h))
    var lab = List[Int](length=2 * P, fill=0)
    var trp = List[Int]()
    var tep = List[Int]()
    var ytr = List[Int]()
    var yte = List[Int]()
    for i in range(P):
        lab[i] = 1
    for i in range(P):
        for c in range(2):
            var p = i + c * P
            if i < ntr:
                trp.append(p)
            else:
                tep.append(p)
                yte.append(lab[p])
    var s = select(RL, layers, h, trp, lab, P)
    var mu = List[Float32](length=h, fill=0)
    var d = List[Float32](length=h, fill=0)
    var sc = test_scores(RL[s.layer], h, trp, lab, tep, mu, d)
    var acc = acc_at(sc, yte, s.tau)
    var j = Json.object()
    j.set("layer", Json.of(layers[s.layer]))
    j.set("tau", Json.of(s.tau))
    j.set("cv_balanced_accuracy", Json.of(s.cv))
    j.set("test_accuracy", Json.of(acc))
    j.set("test_auc", Json.of(auc(sc, yte)))
    var pl = Json.array()
    for li in range(len(layers)):  # per layer: CV and test accuracy at that layer's own best tau
        var m2 = List[Float32](length=h, fill=0)
        var d2 = List[Float32](length=h, fill=0)
        var sc2 = test_scores(RL[li], h, trp, lab, tep, m2, d2)
        var e = Json.object()
        e.set("layer", Json.of(layers[li]))
        e.set("cv", Json.of(s.cvs[li]))
        e.set("test", Json.of(acc_at(sc2, yte, s.taus[li])))
        pl.push(e)
    j.set("per_layer", pl)
    var wj = Json.array()  # threshold width: tau + delta * range, delta in -0.25 .. 0.25 (range = out-of-fold score range at the chosen layer)
    var wmin: Float64 = 1.0
    for k in range(-5, 6):
        var tt = s.tau + Float64(k) * 0.05 * (s.hi - s.lo)
        var e = Json.object()
        e.set("delta", Json.of(Float64(k) * 0.05))
        var ac = acc_at(sc, yte, tt)
        e.set("accuracy", Json.of(ac))
        wmin = min(wmin, ac)
        wj.push(e)
    j.set("width", wj)
    j.set("width_min_accuracy", Json.of(wmin))
    var cj = Json.array()
    var csum: Float64 = 0
    for sd in range(nctl):  # shuffled labels over the train prompts, whole pipeline including selection
        var st = UInt64(sd + 1) * 0x9E3779B97F4A7C15
        var sl = lab.copy()
        var idx = trp.copy()
        var i = len(idx) - 1
        while i > 0:
            var k = Int(xs(st) % UInt64(i + 1))
            var t = sl[idx[i]]
            sl[idx[i]] = sl[idx[k]]
            sl[idx[k]] = t
            i -= 1
        var sel2 = select(RL, layers, h, trp, sl, P)
        var m3 = List[Float32](length=h, fill=0)
        var d3 = List[Float32](length=h, fill=0)
        var sc3 = test_scores(RL[sel2.layer], h, trp, sl, tep, m3, d3)
        var ac = acc_at(sc3, yte, sel2.tau)
        csum += ac
        cj.push(Json.of(ac))
    j.set("control_accuracies", cj)
    j.set("control_mean", Json.of(csum / Float64(max(nctl, 1))))
    wf32(out + "/d.f32", d)
    wf32(out + "/mu.f32", mu)
    j.set("cap", Json.of(cap))
    j.set("n_train", Json.of(len(trp)))
    j.set("n_test", Json.of(len(tep)))
    write_atomic(out + "/fit.json", List[UInt8](j.dumps(False).as_bytes()))
    var vals = List[Float64]()
    vals.append(acc)
    vals.append(s.cv)
    rl.done("cast fit layer " + String(layers[s.layer]) + " tau " + String(s.tau) + " test accuracy " + String(acc) + " control " + String(csum / Float64(max(nctl, 1))), out + "/fit.json", vals, 0.0, 1.0)
    return j^


def score_req(fitdir: String, cap: String, req: Int) raises -> Tuple[Float64, Float64]:
    var fj = parse(read_text(fitdir + "/fit.json"))
    var meta = parse(read_text(cap + "/meta.json"))
    var h = meta.get("h").int()
    var recs = last_recs(cap)
    if req < 1 or req > len(recs):
        raise Error("cast: request " + String(req) + " outside 1.." + String(len(recs)))
    var sel = List[Int]()
    sel.append(recs[req - 1])
    var R = cap_rows(cap, fj.get("layer").int(), sel, h)
    var mu = f32file(fitdir + "/mu.f32")
    var d = f32file(fitdir + "/d.f32")
    return (cosine(R, 0, h, mu, d), fj.get("tau").num())


def gate_call(a: Json) raises -> Json:
    var req = a.get("req").int()
    var sc = score_req(a.get("fit").str(), a.get("cap").str(), req)
    var persona = f32file(a.get("persona").str())
    var fire = fires(sc[0], sc[1])
    var v = persona.copy() if fire else List[Float32](length=len(persona), fill=0)
    wf32(a.get("out").str(), v)
    var o = Json.object()
    o.set("score", Json.of(sc[0]))
    o.set("tau", Json.of(sc[1]))
    o.set("fired", Json.of(fire))
    var rl = Rlog("cast", "gate req " + String(req), "")
    var vals = List[Float64]()
    vals.append(sc[0])
    rl.done("cast gate req " + String(req) + " score " + String(sc[0]) + " tau " + String(sc[1]) + (" fired" if fire else " off"), a.get("out").str(), vals, -1.0, 1.0)
    return o^


def score_call(a: Json) raises -> Json:
    var p = a.get("reqs").str().split("-")
    var lo = Int(String(p[0]))
    var hi = Int(String(p[1])) if len(p) > 1 else lo
    var n = 0
    for r in range(lo, hi + 1):
        var sc = score_req(a.get("fit").str(), a.get("cap").str(), r)
        if fires(sc[0], sc[1]):
            n += 1
    var o = Json.object()
    o.set("requests", Json.of(hi - lo + 1))
    o.set("fired", Json.of(n))
    o.set("fire_rate", Json.of(Float64(n) / Float64(hi - lo + 1)))
    return o^


def cast_call(a: Json) raises -> Json:
    var op = a.text("op")
    if op == "fit":
        for key in ["cap", "out"]:
            if not a.has(key):
                raise Error("cast: fit needs '" + key + "'")
        return fit_call(a)
    if op == "gate":
        for key in ["fit", "cap", "req", "persona", "out"]:
            if not a.has(key):
                raise Error("cast: gate needs '" + key + "'")
        return gate_call(a)
    if op == "score":
        for key in ["fit", "cap", "reqs"]:
            if not a.has(key):
                raise Error("cast: score needs '" + key + "'")
        return score_call(a)
    raise Error("cast: op must be fit, gate or score")
