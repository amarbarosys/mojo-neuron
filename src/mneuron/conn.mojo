"""Connections by weight cosine, direct path only (no norms, no attention in between).

Downstream of (L, n): cos(unit(W_down[:, n] of L), W_gate/W_up row m of layer L') for L' in L+1..L+span.
Upstream of (L, n): cos(W_gate/W_up row n of L, unit(W_down[:, m] of layer L')) for L' in L-span..L-1.
A large |cos| means the neuron writes where the other reads (or the reverse). Dense MLP layers only.
"""
from std.math import sqrt
from ggufx.model import Gguf
from mneuron.ggio import tensor, load_rows


@fieldwise_init
struct Conn(ImplicitlyCopyable):
    var layer: Int
    var neuron: Int
    var role: String  # "gate" or "up" (the reading side)
    var cos: Float64


def keep(mut best: List[Conn], c: Conn, k: Int):
    """Insert c into best, sorted by |cos| descending, at most k entries."""
    var a = c.cos if c.cos >= 0 else -c.cos
    if len(best) == k:
        var l = best[k - 1].cos
        if a <= (l if l >= 0 else -l):
            return
        _ = best.pop()
    var p = len(best)
    best.append(c.copy())
    while p > 0:
        var q = best[p - 1].cos
        if (q if q >= 0 else -q) >= a:
            break
        best.swap_elements(p - 1, p)
        p -= 1


def down_col(g: Gguf, L: Int, n: Int) raises -> List[Float32]:
    """Column n of blk.L.ffn_down (length hidden), raw, not normalised."""
    var ti = tensor(g, "blk." + String(L) + ".ffn_down.weight")
    if ti < 0:
        raise Error("layer " + String(L) + " has no dense ffn_down")
    var ffn = g.tensors[ti].dims[0]
    var h = g.tensors[ti].dims[1]
    var D = load_rows(g, ti, 0, h)
    var out = List[Float32](length=h, fill=0)
    for j in range(h):
        out[j] = D[j * ffn + n]
    return out^


def unit(v: List[Float32]) -> List[Float32]:
    var s: Float64 = 0
    for x in v:
        s += Float64(x) * Float64(x)
    var inv = Float32(1.0 / sqrt(s)) if s > 0 else Float32(0)
    var o = List[Float32](length=len(v), fill=0)
    for j in range(len(v)):
        o[j] = v[j] * inv
    return o^


def downstream(g: Gguf, L: Int, n: Int, span: Int, k: Int) raises -> List[Conn]:
    var v = unit(down_col(g, L, n))
    var h = len(v)
    var best = List[Conn]()
    for L2 in range(L + 1, L + span + 1):
        for role in ["gate", "up"]:
            var ti = tensor(g, "blk." + String(L2) + ".ffn_" + role + ".weight")
            if ti < 0:
                continue
            var ffn = g.tensors[ti].dims[1]
            var R = load_rows(g, ti, 0, ffn)
            for m in range(ffn):
                var d: Float64 = 0
                var nn: Float64 = 0
                for j in range(h):
                    var x = Float64(R[m * h + j])
                    d += x * Float64(v[j])
                    nn += x * x
                if nn > 0:
                    keep(best, Conn(L2, m, String(role), d / sqrt(nn)), k)
    return best^


def upstream(g: Gguf, L: Int, n: Int, span: Int, k: Int) raises -> List[Conn]:
    var best = List[Conn]()
    for role in ["gate", "up"]:
        var ti = tensor(g, "blk." + String(L) + ".ffn_" + role + ".weight")
        if ti < 0:
            continue
        var h = g.tensors[ti].dims[0]
        var row = unit(load_rows(g, ti, n, 1))
        for L2 in range(max(0, L - span), L):
            var di = tensor(g, "blk." + String(L2) + ".ffn_down.weight")
            if di < 0:
                continue
            var ffn = g.tensors[di].dims[0]
            var D = load_rows(g, di, 0, h)
            var dot = List[Float64](length=ffn, fill=0)
            var nrm = List[Float64](length=ffn, fill=0)
            for j in range(h):
                var rj = Float64(row[j])
                for m in range(ffn):
                    var x = Float64(D[j * ffn + m])
                    dot[m] += x * rj
                    nrm[m] += x * x
            for m in range(ffn):
                if nrm[m] > 0:
                    keep(best, Conn(L2, m, String(role), dot[m] / sqrt(nrm[m])), k)
    return best^
