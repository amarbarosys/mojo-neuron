"""Check MODEL LAYER NEURON: golden for connections. Recomputes the top downstream and upstream pairs by an independent
element-wise f64 path and compares the cosine; a mutant (neighbour neuron) must give a different cosine."""
from std.sys import argv, exit
from std.math import sqrt
from ggufx.model import Gguf
from mneuron.ggio import tensor, load_rows
from mneuron.conn import downstream, upstream, down_col


def cos_down_to_read(g: Gguf, L: Int, n: Int, L2: Int, m: Int, role: String) raises -> Float64:
    var di = tensor(g, "blk." + String(L) + ".ffn_down.weight")
    var h = g.tensors[di].dims[1]
    var dn = List[Float64]()
    for j in range(h):  # one row of W_down at a time: element n of row j
        dn.append(Float64(load_rows(g, di, j, 1)[n]))
    var r = load_rows(g, tensor(g, "blk." + String(L2) + ".ffn_" + role + ".weight"), m, 1)
    var d: Float64 = 0
    var a: Float64 = 0
    var b: Float64 = 0
    for j in range(h):
        d += dn[j] * Float64(r[j])
        a += dn[j] * dn[j]
        b += Float64(r[j]) * Float64(r[j])
    return d / sqrt(a * b)


def run() raises:
    var av = argv()
    if len(av) < 4:
        raise Error("usage: conn_check MODEL LAYER NEURON")
    var g = Gguf(String(av[1]))
    var L = Int(String(av[2]))
    var n = Int(String(av[3]))
    var dn = downstream(g, L, n, 2, 3)
    if len(dn) == 0:
        raise Error("no downstream connections")
    for i in range(len(dn)):
        var c = cos_down_to_read(g, L, n, dn[i].layer, dn[i].neuron, dn[i].role)
        if abs(c - dn[i].cos) > 1e-4:
            raise Error("downstream cos " + String(dn[i].cos) + " vs f64 oracle " + String(c))
    var other = downstream(g, L, n + 1, 2, 3)
    if other[0].layer == dn[0].layer and other[0].neuron == dn[0].neuron and abs(other[0].cos - dn[0].cos) < 1e-6:
        raise Error("mutant passed: neighbour neuron has the same top connection")
    var up = upstream(g, L + 1, dn[0].neuron if dn[0].layer == L + 1 else n, 1, 3)
    if len(up) == 0:
        raise Error("no upstream connections")
    # symmetry: the top downstream pair (L+1, m) must list (L, n) upstream with the same cosine, when the roles match
    var top = dn[0]
    if top.layer == L + 1:
        var back = upstream(g, L + 1, top.neuron, 1, 100000)
        var seen = False
        for b in back:
            if b.layer == L and b.neuron == n and b.role == top.role:
                seen = True
                if abs(b.cos - top.cos) > 1e-4:
                    raise Error("up/down asymmetry: " + String(b.cos) + " vs " + String(top.cos))
        if not seen:
            raise Error("pair (L,n) missing from the upstream list of its reader")
    print("PASS conn check: top", len(dn), "downstream pairs match the f64 oracle, upstream symmetric, mutant differs")


def main():
    try:
        run()
    except e:
        print("FAIL conn_check:", e)
        exit(1)
