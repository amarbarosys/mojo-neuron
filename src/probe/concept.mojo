"""F3 concept arithmetic: `A - B + C` -> one steering vector (f32[n_layer * h], the BARO_STEER layout) and its logit lens.
Terms (each is unit-normalised, then weighted, summed per layer, times `scale`; `A - A` is exactly zero):
  L:N              decoder column N of layer L (ffn_down.weight)       2*L:N  weighted
  tok:TEXT@L       unembedding row of the token whose display text is TEXT (`_` stands for a space: tok:_king@16)
  vec:FILE@L       a raw f32[h] direction from a file
  mean(CAP,L,a-b)  mean last-token residual (field r, layer L) of requests a..b (1-based) of a capture dir
A term without @L uses the call's `layer`. Lens = W_U (v * output_norm) per touched layer, top and bottom k tokens."""
from std.math import sqrt
from json import Json
from ggufx.model import Gguf
from mneuron.ggio import tensor, load_rows, vocab
from mneuron.fwd import matmul
from probe.lie import last_recs, cap_rows


struct Term(Copyable, Movable):
    var coef: Float64
    var kind: String  # neuron, tok, vec, mean
    var spec: String
    var layer: Int

    def __init__(out self, coef: Float64, kind: String, spec: String, layer: Int):
        self.coef = coef
        self.kind = kind
        self.spec = spec
        self.layer = layer


def parse_term(s: String, sign: Float64, default_layer: Int) raises -> Term:
    var coef = sign
    var body = s
    var star = s.find("*")
    if star > 0:
        var pre = String(s[byte=0:star])
        if pre.find("(") < 0 and pre.find(":") < 0 and pre.find("@") < 0:  # a leading number, not a path or mean(...) containing '*'
            coef = sign * atof(pre)
            body = String(s[byte=star + 1 :])
    var layer = default_layer
    if body.startswith("mean(") and body.endswith(")"):
        var p = String(body[byte=5 : body.byte_length() - 1]).split(",")
        if len(p) != 3:
            raise Error("concept: bad term '" + body + "', want mean(CAP,L,a-b)")
        return Term(coef, "mean", String(p[0]) + "," + String(p[2]), Int(p[1]))
    var at = body.rfind("@")
    if at > 0:
        layer = Int(String(body[byte=at + 1 :]))
        var head = String(body[byte=0:at])
        body = head^
    if body.startswith("tok:"):
        return Term(coef, "tok", String(body[byte=4 :]).replace("_", " "), layer)
    if body.startswith("vec:"):
        return Term(coef, "vec", String(body[byte=4 :]), layer)
    var p = body.split(":")
    if len(p) == 2:
        return Term(coef, "neuron", String(p[1]), Int(p[0]))
    raise Error("concept: cannot read term '" + s + "' (L:N, tok:TEXT@L, vec:FILE@L, mean(CAP,L,a-b))")


def parse_expr(expr: String, default_layer: Int) raises -> List[Term]:
    var out = List[Term]()
    var sign: Float64 = 1.0
    var want_term = True
    for tk in expr.split(" "):
        var t = String(tk)
        if t.byte_length() == 0:
            continue
        if want_term and (t == "+" or t == "-") and len(out) == 0:
            sign = 1.0 if t == "+" else -1.0
        elif not want_term and (t == "+" or t == "-"):
            sign = 1.0 if t == "+" else -1.0
            want_term = True
        elif want_term:
            out.append(parse_term(t, sign, default_layer))
            want_term = False
        else:
            raise Error("concept: expected + or - before '" + t + "' (operators need spaces around them)")
    if len(out) == 0 or want_term:
        raise Error("concept: empty or dangling expression '" + expr + "'")
    return out^


def unit(v: List[Float32], what: String) raises -> List[Float32]:
    var s: Float64 = 0
    for j in range(len(v)):
        s += Float64(v[j]) ** 2
    if s <= 0:
        raise Error("concept: term " + what + " is the zero vector")
    var inv = Float32(1.0 / sqrt(s))
    var o = List[Float32](length=len(v), fill=0)
    for j in range(len(v)):
        o[j] = v[j] * inv
    return o^


def term_dir(g: Gguf, t: Term, h: Int, tokens: List[String]) raises -> List[Float32]:
    if t.kind == "neuron":
        var di = tensor(g, "blk." + String(t.layer) + ".ffn_down.weight")
        if di < 0:
            raise Error("concept: layer " + String(t.layer) + " has no ffn_down")
        var ffn = g.tensors[di].dims[0]
        var n = Int(t.spec)
        if n < 0 or n >= ffn:
            raise Error("concept: neuron " + t.spec + " out of range 0.." + String(ffn - 1))
        var D = load_rows(g, di, 0, h)
        var c = List[Float32](length=h, fill=0)
        for j in range(h):
            c[j] = D[j * ffn + n]
        return unit(c, "L" + String(t.layer) + ":" + t.spec)
    if t.kind == "tok":
        var ui = tensor(g, "output.weight")
        if ui < 0:
            ui = tensor(g, "token_embd.weight")
        for i in range(len(tokens)):
            if tokens[i] == t.spec:
                return unit(load_rows(g, ui, i, 1), "tok:" + t.spec)
        raise Error("concept: no token with display text '" + t.spec + "'")
    if t.kind == "vec":
        var raw = open(t.spec, "r").read_bytes()
        if len(raw) != h * 4:
            raise Error("concept: " + t.spec + " must be f32[" + String(h) + "]")
        var v = List[Float32](length=h, fill=0)
        var p = raw.unsafe_ptr().unsafe_bitcast[Float32]()
        for j in range(h):
            v[j] = p[unsafe_offset=j]
        return unit(v, "vec:" + t.spec)
    # mean(CAP, L, a-b)
    var cp = t.spec.split(",")
    var range_ = String(cp[1]).split("-")
    var a = Int(range_[0])
    var b = Int(range_[1]) if len(range_) > 1 else a
    var recs = last_recs(String(cp[0]))
    if a < 1 or b > len(recs) or a > b:
        raise Error("concept: requests " + String(cp[1]) + " outside 1.." + String(len(recs)))
    var sel = List[Int]()
    for i in range(a - 1, b):
        sel.append(recs[i])
    var rows = cap_rows(String(cp[0]), t.layer, sel, h)
    var m = List[Float32](length=h, fill=0)
    for k in range(len(sel)):
        for j in range(h):
            m[j] += rows[k * h + j] / Float32(len(sel))
    return unit(m, "mean(" + t.spec + ")")


def build(g: Gguf, terms: List[Term], scale: Float64, h: Int, n_layer: Int) raises -> List[Float32]:
    """f32[n_layer * h]: row L = scale * sum of the (unit * coef) terms at layer L."""
    var vec = List[Float32](length=n_layer * h, fill=0)
    var tokens = vocab(g)
    for t in terms:
        if t.layer < 0 or t.layer >= n_layer:
            raise Error("concept: layer " + String(t.layer) + " outside 0.." + String(n_layer - 1))
        var d = term_dir(g, t, h, tokens)
        for j in range(h):
            vec[t.layer * h + j] += Float32(t.coef * scale) * d[j]
    return vec^


def lens_of(g: Gguf, v: List[Float32], h: Int, k: Int) raises -> Json:
    """Top and bottom k tokens of W_U (v * output_norm), best first."""
    var ni = tensor(g, "output_norm.weight")
    var ui = tensor(g, "output.weight")
    if ui < 0:
        ui = tensor(g, "token_embd.weight")
    var nw = load_rows(g, ni, 0, 1)
    var gv = List[Float32](length=h, fill=0)
    for j in range(h):
        gv[j] = v[j] * nw[j]
    var V = g.tensors[ui].dims[1]
    var top = List[Int]()  # ids sorted by score, best first
    var bot = List[Int]()
    var ts = List[Float32]()
    var bs = List[Float32]()
    var r0 = 0
    while r0 < V:
        var m = min(4096, V - r0)
        var z = matmul(load_rows(g, ui, r0, m), gv, m, h, 1)
        for i in range(m):
            insert(top, ts, r0 + i, z[i], k, True)
            insert(bot, bs, r0 + i, z[i], k, False)
        r0 += m
    var tok = vocab(g)
    var o = Json.object()
    var tj = Json.array()
    var bj = Json.array()
    for i in range(len(top)):
        var e = Json.array()
        e.push(Json.of(tok[top[i]]))
        e.push(Json.of(Float64(ts[i])))
        tj.push(e)
    for i in range(len(bot)):
        var e = Json.array()
        e.push(Json.of(tok[bot[i]]))
        e.push(Json.of(Float64(bs[i])))
        bj.push(e)
    o.set("top", tj)
    o.set("bottom", bj)
    return o^


def insert(mut ids: List[Int], mut sc: List[Float32], id: Int, z: Float32, k: Int, high: Bool):
    var pos = len(ids)
    while pos > 0 and ((sc[pos - 1] < z) if high else (sc[pos - 1] > z)):
        pos -= 1
    if pos >= k:
        return
    ids.insert(pos, id)
    sc.insert(pos, z)
    if len(ids) > k:
        _ = ids.pop()
        _ = sc.pop()
