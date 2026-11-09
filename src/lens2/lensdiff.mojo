"""lens-diff: contrastive lens of matched prompts (docs/prereg-lensdiff.md). For matched requests a_i, b_i of a capture dir take the last-token residual
(field r) at `layer`, d_i = h(a_i) - h(b_i), m = mean(d_i), lens of m = W_U (m * output_norm). Reports norm, consistency (share of pairs with cos(d_i, m) > 0)
and `null` lenses (centred sign flips s_i * (d_i - m), what the residual variation alone gives). `-D LENSDIFF_MUT=1` sums instead of subtracting."""
from std.math import sqrt
from std.sys import get_defined_int
from json import Json
from ggufx.model import Gguf
from mneuron.ggio import tensor, load_rows, vocab
from mneuron.fwd import matmul
from mneuron.rlog import Rlog
from mneuron.store import model_info
from probe.lie import last_recs, cap_rows
from probe.concept import insert

comptime MUT = get_defined_int["LENSDIFF_MUT", 0]()


struct Lens(Movable):
    var top: List[List[Int]]
    var tsc: List[List[Float32]]
    var bot: List[List[Int]]
    var bsc: List[List[Float32]]

    def __init__(out self, nv: Int):
        self.top = List[List[Int]]()
        self.tsc = List[List[Float32]]()
        self.bot = List[List[Int]]()
        self.bsc = List[List[Float32]]()
        for _ in range(nv):
            self.top.append(List[Int]())
            self.tsc.append(List[Float32]())
            self.bot.append(List[Int]())
            self.bsc.append(List[Float32]())


def lens_multi(g: Gguf, V: List[Float32], nv: Int, h: Int, k: Int) raises -> Lens:
    """Top and bottom k vocabulary ids of W_U (V[t] * output_norm) for each of the nv vectors, one pass over W_U."""
    var ni = tensor(g, "output_norm.weight")
    var ui = tensor(g, "output.weight")
    if ui < 0:
        ui = tensor(g, "token_embd.weight")
    var nw = load_rows(g, ni, 0, 1)
    var G = List[Float32](length=nv * h, fill=0)
    for t in range(nv):
        for j in range(h):
            G[t * h + j] = V[t * h + j] * nw[j]
    var n = g.tensors[ui].dims[1]
    var o = Lens(nv)
    var r0 = 0
    while r0 < n:
        var m = min(4096, n - r0)
        var z = matmul(load_rows(g, ui, r0, m), G, m, h, nv)
        for t in range(nv):
            for i in range(m):
                insert(o.top[t], o.tsc[t], r0 + i, z[t * m + i], k, True)
                insert(o.bot[t], o.bsc[t], r0 + i, z[t * m + i], k, False)
        r0 += m
    return o^


def two(x: Float64, y: Float64) -> List[Float64]:
    var o = List[Float64]()
    o.append(x)
    o.append(y)
    return o^


def span(s: String, n: Int, what: String) raises -> List[Int]:
    """1-based inclusive "a-b" (or "a") -> 0-based indexes."""
    var p = s.split("-")
    var lo = Int(String(p[0]))
    var hi = Int(String(p[1])) if len(p) > 1 else lo
    if lo < 1 or hi > n or lo > hi:
        raise Error("lens-diff: " + what + " " + s + " outside 1.." + String(n))
    var o = List[Int]()
    for i in range(lo - 1, hi):
        o.append(i)
    return o^


def pairs_json(tok: List[String], ids: List[Int], sc: List[Float32]) -> Json:
    var j = Json.array()
    for i in range(len(ids)):
        var e = Json.array()
        e.push(Json.of(tok[ids[i]]))
        e.push(Json.of(Float64(sc[i])))
        j.push(e)
    return j^


def lensdiff_call(a: Json) raises -> Json:
    """{gguf|model, cap, layer, a, b, k?=10, null?=20, out?}: lens of mean(h(a_i) - h(b_i)); a, b are 1-based request ranges of equal length."""
    var path = a.text("gguf")
    if path.byte_length() == 0:
        if not a.has("model"):
            raise Error("lens-diff: give 'gguf' (a path) or 'model' (a store name)")
        path = model_info(a.get("model").str()).text("gguf")
    for key in ["cap", "layer", "a", "b"]:
        if not a.has(key):
            raise Error("lens-diff: missing argument '" + key + "'")
    var cap = a.get("cap").str()
    var layer = a.get("layer").int()
    var k = a.get("k").int() if a.has("k") else 10
    var nnull = a.get("null").int() if a.has("null") else 20
    var g = Gguf(path)
    var h = g.tensors[tensor(g, "output_norm.weight")].dims[0]
    var rl = Rlog("lens-diff", cap + " L" + String(layer) + " " + a.get("a").str() + " vs " + a.get("b").str(), path)
    var recs = last_recs(cap)
    var ia = span(a.get("a").str(), len(recs), "a")
    var ib = span(a.get("b").str(), len(recs), "b")
    if len(ia) != len(ib):
        raise Error("lens-diff: a has " + String(len(ia)) + " requests, b has " + String(len(ib)) + " (pairs must match)")
    var n = len(ia)
    var sa = List[Int]()
    var sb = List[Int]()
    for i in range(n):
        sa.append(recs[ia[i]])
        sb.append(recs[ib[i]])
    var ha = cap_rows(cap, layer, sa, h)
    var hb = cap_rows(cap, layer, sb, h)
    var d = List[Float32](length=n * h, fill=0)
    var m = List[Float32](length=h, fill=0)
    for i in range(n):
        for j in range(h):
            d[i * h + j] = (ha[i * h + j] + hb[i * h + j]) if MUT == 1 else (ha[i * h + j] - hb[i * h + j])
            m[j] += d[i * h + j] / Float32(n)
    var nrm: Float64 = 0
    for j in range(h):
        nrm += Float64(m[j]) ** 2
    nrm = sqrt(nrm)
    var o = Json.object()
    o.set("layer", Json.of(layer))
    o.set("pairs", Json.of(n))
    o.set("norm", Json.of(nrm))
    var good = 0
    for i in range(n):
        var dot: Float64 = 0
        for j in range(h):
            dot += Float64(d[i * h + j]) * Float64(m[j])
        if dot > 0:
            good += 1
    var cons = Float64(good) / Float64(n)
    o.set("consistency", Json.of(cons))
    if nrm == 0:
        o.set("zero", Json.of(True))
        rl.done("lens-diff zero difference at L" + String(layer), "", two(0.0, cons), 0.0, 1e9)
        return o^
    var V = List[Float32](length=(1 + nnull) * h, fill=0)
    for j in range(h):
        V[j] = m[j]
    for s in range(nnull):  # centred sign flips: what the variation around the mean difference gives
        var st = UInt64(s + 1) * 0x9E3779B97F4A7C15
        for i in range(n):
            st ^= st << 13
            st ^= st >> 7
            st ^= st << 17
            var sg = Float32(1.0) if (st & 1) == 1 else Float32(-1.0)
            for j in range(h):
                V[(1 + s) * h + j] += sg * (d[i * h + j] - m[j]) / Float32(n)
    var L = lens_multi(g, V, 1 + nnull, h, k)
    var tok = vocab(g)
    var lj = Json.object()
    lj.set("top", pairs_json(tok, L.top[0], L.tsc[0]))
    lj.set("bottom", pairs_json(tok, L.bot[0], L.bsc[0]))
    o.set("lens", lj)
    var nj = Json.array()
    for s in range(nnull):
        var e = Json.array()
        for i in range(len(L.top[1 + s])):
            e.push(Json.of(tok[L.top[1 + s][i]]))
        nj.push(e)
    o.set("null_top", nj)
    rl.done("lens-diff L" + String(layer) + " " + String(n) + " pairs, norm " + String(nrm) + ", consistency " + String(cons), "", two(nrm, cons), 0.0, 1e9)
    return o^
