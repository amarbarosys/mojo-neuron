"""Implants MODEL.gguf CAPDIR LAYER PAIRS.tsv OUT.bin [tau=0.95] [vnorm=10] [scale=1] [margin=2] (plan G2, format docs/format-implants.md).

One implant per PAIRS line `rec<TAB>right<TAB>wrong<TAB>gap`: rec = record of CAPDIR/L<layer>.f16 (A1 capture) holding the mistake's last prompt position, right and wrong are
token ids, gap = logit(wrong) - logit(right) the engine measured there. key = that record's FFN input row `x`. value = unit(gamma * (U[right] - U[wrong])) * vnorm, the output-norm
weighted unembedding difference (the lens direction). gate = scale * (gap + margin) / ((1 - tau) * vnorm * |gamma * (U[right] - U[wrong])|): the first-order push (the key matches
itself at cosine 1, so the relu gives 1 - tau) that makes the right token win by `margin`; `scale` absorbs what the later layers and the final-norm scale do, and the harness fits it.
No gradient step. Prints one summary line and calls rlog.
"""
from std.sys import argv
from std.math import sqrt, isfinite
from std.memory import bitcast
from ggufx.model import Gguf, pread_into, open_fd
from json import parse
from mneuron.ggio import tensor, load_rows, abs_path
from mneuron.store import read_text
from mneuron.rlog import Rlog


def opt(a: List[String], key: String, dflt: String) -> String:
    for x in a:
        if x.startswith(key + "="):
            return String(x[byte=key.byte_length() + 1:])
    return dflt


def put32(mut b: List[UInt8], v: UInt32):
    for i in range(4):
        b.append(UInt8((v >> UInt32(8 * i)) & 255))


def main() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    if len(a) < 6:
        raise Error("usage: implants MODEL.gguf CAPDIR LAYER PAIRS.tsv OUT.bin [tau=] [vnorm=] [scale=] [margin=]")
    var cap = abs_path(a[2])
    var layer = Int(a[3])
    var tau = Float32(Float64(opt(a, "tau", "0.95")))
    var vnorm = Float32(Float64(opt(a, "vnorm", "10")))
    var scale = Float32(Float64(opt(a, "scale", "1")))
    var margin = Float32(Float64(opt(a, "margin", "2")))
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
    var ui = tensor(g, "output.weight")
    if ui < 0:
        ui = tensor(g, "token_embd.weight")
    var ni = tensor(g, "output_norm.weight")
    if ui < 0 or ni < 0 or g.tensors[ui].dims[0] != h:
        raise Error("need output(.|token_embd).weight with ne0 == hidden and output_norm.weight")
    var gamma = load_rows(g, ni, 0, 1)
    var fd = open_fd(cap + "/L" + String(layer) + ".f16", 0)
    var raw = List[UInt8](length=h * 2, fill=0)
    var keys = List[Float32]()
    var vals = List[Float32]()
    var gates = List[Float32]()
    var gaps = List[Float64]()
    for line in read_text(a[4]).split("\n"):
        if line.byte_length() == 0:
            continue
        var p = line.split("\t")
        if len(p) != 4:
            raise Error("PAIRS line needs rec, right, wrong, gap: " + String(line))
        var rec = Int(p[0])
        pread_into(fd, raw, 0, h * 2, (rec * nf + xi) * h * 2)
        for j in range(h):
            keys.append(Float32(bitcast[DType.float16](UInt16(Int(raw[2 * j]) | (Int(raw[2 * j + 1]) << 8)))))
        var ur = load_rows(g, ui, Int(p[1]), 1)
        var uw = load_rows(g, ui, Int(p[2]), 1)
        var n2: Float32 = 0
        var d = List[Float32](length=h, fill=0)
        for j in range(h):
            d[j] = gamma[j] * (ur[j] - uw[j])
            n2 += d[j] * d[j]
        var dn = sqrt(n2)
        if dn <= 0:
            raise Error("right and wrong token have the same unembedding row: " + String(line))
        for j in range(h):
            vals.append(d[j] / dn * vnorm)
        var gap = Float32(Float64(p[3]))
        gates.append(scale * (gap + margin) / ((1 - tau) * vnorm * dn))
        gaps.append(Float64(gap))
    var k = len(gates)
    var out = List[UInt8]()
    for c in "NIMP0001".as_bytes():
        out.append(c)
    put32(out, UInt32(h))
    put32(out, 1)
    put32(out, UInt32(layer))
    put32(out, UInt32(k))
    put32(out, bitcast[DType.uint32](tau))
    for i in range(k * h):
        put32(out, bitcast[DType.uint32](keys[i]))
    for i in range(k):
        put32(out, bitcast[DType.uint32](gates[i]))
    for i in range(k * h):
        put32(out, bitcast[DType.uint32](vals[i]))
    with open(abs_path(a[5]), "w") as f:
        f.write_bytes(out)
    var gmax: Float32 = 0
    var chk = List[Float64]()
    for i in range(k):
        gmax = max(gmax, gates[i])
        chk.append(Float64(gates[i]))
    var res = "layer=" + String(layer) + " k=" + String(k) + " tau=" + String(tau) + " scale=" + String(scale) + " gate_max=" + String(gmax)
    print("implants:", res, "->", a[5])
    var rl = Rlog("implants", res, a[1])
    rl.done(res, abs_path(a[5]), chk, 0.0, 1e9)
