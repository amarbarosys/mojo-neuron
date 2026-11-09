"""Prerank MODEL.gguf CAPDIR LAYER FACTS.tsv OUT (plan A2c, docs/prereg-a2c-group.md): cheap linear direct effect of every FFN neuron of one layer on logit(target) - logit(foil).

FACTS.tsv lines `context<TAB>target word<TAB>foil word`; CAPDIR = an A1 capture (field x) of the contexts as requests of n = 1 in file order. Per fact: x = the FFN input of the last prompt
position, a_n = silu(gate_n . x) * (up_n . x), w = gamma * (U[target] - U[foil]) (the output-norm weighted unembedding difference, first token of " word"), c_n = w . down_n, de_n = a_n * c_n.
Writes OUT.de.f32 (facts x ffn, f32) and OUT.rank.tsv (neuron, mean de over facts; descending = the neurons that push toward the target). Weights come from MODEL (the engine
pack is a Q4_0 of the same model: this is a proxy for the ranking, not the engine's numbers). Prints seconds. Ends with rlog.
"""
from std.sys import argv
from std.math import exp, isfinite
from std.memory import bitcast
from std.time import perf_counter_ns
from ggufx.model import Gguf, pread_into, open_fd
from json import parse
from tokenizer import Tokenizer
from mneuron.ggio import tensor, load_rows, abs_path
from mneuron.store import read_text
from mneuron.rlog import Rlog


def main() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    if len(a) < 6:
        raise Error("usage: prerank MODEL.gguf CAPDIR LAYER FACTS.tsv OUT")
    var t0 = perf_counter_ns()
    var cap = abs_path(a[2])
    var layer = Int(a[3])
    var out = abs_path(a[5])
    var meta = parse(read_text(cap + "/meta.json"))
    var h = meta.get("h").int()
    var fields = meta.get("fields").str()
    var nf = fields.byte_length()
    var xi = -1
    for i in range(nf):
        if fields.as_bytes()[i] == UInt8(ord("x")):
            xi = i
    if xi < 0:
        raise Error("capture has no x field")
    var g = Gguf(a[1])
    var tok = Tokenizer(a[1])
    var ls = String(layer)
    var gi = tensor(g, "blk." + ls + ".ffn_gate.weight")
    var ui = tensor(g, "blk." + ls + ".ffn_up.weight")
    var di = tensor(g, "blk." + ls + ".ffn_down.weight")
    var oi = tensor(g, "output.weight")
    if oi < 0:
        oi = tensor(g, "token_embd.weight")
    var ni = tensor(g, "output_norm.weight")
    if gi < 0 or ui < 0 or di < 0 or oi < 0 or ni < 0:
        raise Error("missing tensor for layer " + ls)
    var ffn = g.tensors[gi].dims[1]
    var G = load_rows(g, gi, 0, ffn)  # ffn rows of h
    var U = load_rows(g, ui, 0, ffn)
    var D = load_rows(g, di, 0, h)  # h rows of ffn: D[j * ffn + n]
    var gamma = load_rows(g, ni, 0, 1)
    var reqs = List[Int32](length=1, fill=0)
    var rb = open(cap + "/req.i32", "r").read_bytes()
    var nr = len(rb) // 4
    var ends = List[Int]()
    for i in range(nr):
        var cur = Int(rb[4 * i]) | (Int(rb[4 * i + 1]) << 8) | (Int(rb[4 * i + 2]) << 16) | (Int(rb[4 * i + 3]) << 24)
        var nxt = -1
        if i + 1 < nr:
            nxt = Int(rb[4 * i + 4]) | (Int(rb[4 * i + 5]) << 8) | (Int(rb[4 * i + 6]) << 16) | (Int(rb[4 * i + 7]) << 24)
        if i + 1 == nr or nxt != cur:
            ends.append(i)
    var lines = List[String]()
    for l in read_text(a[4]).split("\n"):
        if l.byte_length() > 0:
            lines.append(String(l))
    var nfacts = len(lines)
    if len(ends) != nfacts:
        raise Error("capture has " + String(len(ends)) + " requests, FACTS has " + String(nfacts))
    var fd = open_fd(cap + "/L" + ls + ".f16", 0)
    var raw = List[UInt8](length=h * 2, fill=0)
    var x = List[Float32](length=h, fill=0)
    var de = List[Float32](length=nfacts * ffn, fill=0)
    var act = List[Float32](length=ffn, fill=0)
    var cn = List[Float32](length=ffn, fill=0)
    var w = List[Float32](length=h, fill=0)
    for f in range(nfacts):
        var p = lines[f].split("\t")
        if len(p) != 3:
            raise Error("FACTS line needs context, target, foil: " + lines[f])
        var tt = tok.encode(String(" ") + String(p[1]), add_special=False)[0]
        var ft = tok.encode(String(" ") + String(p[2]), add_special=False)[0]
        pread_into(fd, raw, 0, h * 2, (ends[f] * nf + xi) * h * 2)
        for j in range(h):
            x[j] = Float32(bitcast[DType.float16](UInt16(Int(raw[2 * j]) | (Int(raw[2 * j + 1]) << 8))))
        var ur = load_rows(g, oi, tt, 1)
        var uw = load_rows(g, oi, ft, 1)
        for j in range(h):
            w[j] = gamma[j] * (ur[j] - uw[j])
        for n in range(ffn):
            var gs: Float32 = 0
            var us: Float32 = 0
            var base = n * h
            for j in range(h):
                gs += G[base + j] * x[j]
                us += U[base + j] * x[j]
            act[n] = gs / (1 + Float32(exp(Float64(-gs)))) * us
        for n in range(ffn):
            cn[n] = 0
        for j in range(h):
            var wj = w[j]
            var base = j * ffn
            for n in range(ffn):
                cn[n] += D[base + n] * wj
        for n in range(ffn):
            de[f * ffn + n] = act[n] * cn[n]
    var mean = List[Float64](length=ffn, fill=0)
    for n in range(ffn):
        var s: Float64 = 0
        for f in range(nfacts):
            s += Float64(de[f * ffn + n])
        mean[n] = s / Float64(nfacts)
    var order = List[Int]()
    for n in range(ffn):
        order.append(n)
    # sort by mean de descending: simple merge-free approach via index sort with a heap-less shell sort
    var gap = ffn // 2
    while gap > 0:
        for i in range(gap, ffn):
            var v = order[i]
            var j = i
            while j >= gap and mean[order[j - gap]] < mean[v]:
                order[j] = order[j - gap]
                j -= gap
            order[j] = v
        gap //= 2
    var ob = List[UInt8]()
    for i in range(nfacts * ffn):
        var bits = bitcast[DType.uint32](de[i])
        for k in range(4):
            ob.append(UInt8((bits >> UInt32(8 * k)) & 255))
    with open(out + ".de.f32", "w") as f:
        f.write_bytes(ob)
    var tsv = String()
    for n in order:
        tsv += String(n) + "\t" + String(mean[n]) + "\n"
    with open(out + ".rank.tsv", "w") as f:
        f.write(tsv)
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    var res = "layer=" + ls + " facts=" + String(nfacts) + " top=" + String(order[0]) + " de_top=" + String(mean[order[0]]) + " seconds=" + String(secs)
    print("prerank:", res)
    var rl = Rlog("prerank", res, a[1])
    var vals = List[Float64]()
    vals.append(mean[order[0]])
    vals.append(mean[order[ffn - 1]])
    rl.done(res, out + ".rank.tsv", vals, -1e9, 1e9)
