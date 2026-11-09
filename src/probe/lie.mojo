"""F24 lie probe core: read last-token residual rows from a capture (docs/format-capture.md), fit a logistic probe, score rows.
A probe is f32[h + 1] on disk: w (raw residual space, so the score of a raw row is w.r + b) then b. Score = logit of P(true)."""
from std.math import exp, log, sqrt
from json import parse
from mneuron.fwd import matmul
from mneuron.store import read_text


def i32s(path: String) raises -> List[Int]:
    var b = open(path, "r").read_bytes()
    var n = len(b) // 4
    var out = List[Int](length=n, fill=0)
    var p = b.unsafe_ptr().unsafe_bitcast[Int32]()
    for i in range(n):
        out[i] = Int(p[unsafe_offset=i])
    return out^


def last_recs(capdir: String) raises -> List[Int]:
    """Record index of the last token of every request, in order of first appearance (req.i32 groups are contiguous)."""
    var req = i32s(capdir + "/req.i32")
    var out = List[Int]()
    for i in range(len(req)):
        if i + 1 == len(req) or req[i + 1] != req[i]:
            out.append(i)
    return out^


def f16_rows(path: String, h: Int, nf: Int, field: Int, recs: List[Int]) raises -> List[Float32]:
    """Row `field` of record r for every r in recs, as f32 (n x h)."""
    var b = open(path, "r").read_bytes()
    if len(b) % (nf * h * 2) != 0:
        raise Error(path + ": size is not a whole number of records")
    var p = b.unsafe_ptr().unsafe_bitcast[Float16]()
    var out = List[Float32](length=len(recs) * h, fill=0)
    for k in range(len(recs)):
        var base = (recs[k] * nf + field) * h
        if (recs[k] + 1) * nf * h * 2 > len(b):
            raise Error(path + ": record " + String(recs[k]) + " past the end")
        for j in range(h):
            out[k * h + j] = p[unsafe_offset=base + j].cast[DType.float32]()
    return out^


def cap_rows(cap: String, layer: Int, recs: List[Int], h: Int) raises -> List[Float32]:
    """Residual (field r) rows of `layer` at records `recs`, the capture's own meta.json giving nf and the field index."""
    var meta = parse(read_text(cap + "/meta.json"))
    if meta.get("h").int() != h:
        raise Error(cap + ": capture h " + String(meta.get("h").int()) + " != model h " + String(h))
    var field = meta.text("fields").find("r")
    if field < 0:
        raise Error(cap + ": no r field in the capture")
    return f16_rows(cap + "/L" + String(layer) + ".f16", h, meta.text("fields").byte_length(), field, recs)


def part(pair: Int) -> Int:
    """0 train, 1 validation, 2 test (by pair id, deterministic)."""
    var m = (pair * 7919 + 13) % 10
    return 0 if m < 6 else (1 if m < 8 else 2)


def pick(parts: List[Int], which: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(len(parts)):
        if parts[i] == which:
            out.append(i)
    return out^


struct Probe(Copyable, Movable):
    var w: List[Float32]
    var b: Float32

    def __init__(out self, h: Int):
        self.w = List[Float32](length=h, fill=0)
        self.b = 0

    def score(self, x: List[Float32], row: Int, h: Int) -> Float32:
        var s = self.b
        for j in range(h):
            s += self.w[j] * x[row * h + j]
        return s


def sigmoid(z: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-z))


def fit(x: List[Float32], y: List[Int], rows: List[Int], h: Int, lam: Float32, lr: Float32, iters: Int) -> Probe:
    """Standardise the rows of `rows`, full-batch gradient descent on the mean logistic loss + lam/2 |w|^2, return the probe in raw space.
    y holds 0/1 per row of x (indexed by the row id in `rows`)."""
    var n = len(rows)
    var mu = List[Float32](length=h, fill=0)
    var sd = List[Float32](length=h, fill=0)
    for k in range(n):
        for j in range(h):
            mu[j] += x[rows[k] * h + j]
    for j in range(h):
        mu[j] /= Float32(n)
    for k in range(n):
        for j in range(h):
            var d = x[rows[k] * h + j] - mu[j]
            sd[j] += d * d
    for j in range(h):
        sd[j] = sqrt(sd[j] / Float32(n)) + 1e-6
    var xs = List[Float32](length=n * h, fill=0)  # n x h standardised
    var xt = List[Float32](length=h * n, fill=0)  # h x n, for the gradient
    for k in range(n):
        for j in range(h):
            var v = (x[rows[k] * h + j] - mu[j]) / sd[j]
            xs[k * h + j] = v
            xt[j * n + k] = v
    var w = List[Float32](length=h, fill=0)
    var b: Float32 = 0
    var r = List[Float32](length=n, fill=0)
    for _ in range(iters):
        var z = matmul(xs, w, n, h, 1)  # z[k] = xs[k] . w
        for k in range(n):
            r[k] = (sigmoid(z[k] + b) - Float32(y[rows[k]])) / Float32(n)
        var g = matmul(xt, r, h, n, 1)
        var gb: Float32 = 0
        for k in range(n):
            gb += r[k]
        for j in range(h):
            w[j] -= lr * (g[j] + lam * w[j])
        b -= lr * gb
    var p = Probe(h)
    var shift: Float32 = 0
    for j in range(h):
        p.w[j] = w[j] / sd[j]
        shift += w[j] * mu[j] / sd[j]
    p.b = b - shift
    return p^


def accuracy(p: Probe, x: List[Float32], y: List[Int], rows: List[Int], h: Int) -> Float64:
    var ok = 0
    for k in range(len(rows)):
        var s = p.score(x, rows[k], h)
        if (s > 0) == (y[rows[k]] == 1):
            ok += 1
    return Float64(ok) / Float64(max(len(rows), 1))


def save_probe(path: String, p: Probe) raises:
    var h = len(p.w)
    var buf = List[Float32](length=h + 1, fill=0)
    for j in range(h):
        buf[j] = p.w[j]
    buf[h] = p.b
    var raw = List[UInt8](length=(h + 1) * 4, fill=0)
    var src = buf.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range((h + 1) * 4):
        raw[i] = src[unsafe_offset=i]
    with open(path, "w") as f:
        f.write_bytes(raw)


def load_probe(path: String) raises -> Probe:
    var b = open(path, "r").read_bytes()
    var h = len(b) // 4 - 1
    if h < 1 or len(b) % 4 != 0:
        raise Error(path + ": not a probe file (f32 w[h] then b)")
    var p = Probe(h)
    var s = b.unsafe_ptr().unsafe_bitcast[Float32]()
    for j in range(h):
        p.w[j] = s[unsafe_offset=j]
    p.b = s[unsafe_offset=h]
    return p^
