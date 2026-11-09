"""Surgery IN.gguf OUT.gguf layer=L neuron=N op=zero|scale|replace|add [k=F] [vec=FILE.f32] (plan A3).

op=zero (default via=up) zeroes the neuron's whole ffn_up row: its own blocks, all-zero bytes dequantize to 0 in every ggml type, so the
activation silu(gate) * up is exactly 0 with nothing else touched (the engine's gain 0). via=down zeroes the down column instead, but K quants
cannot hold an exact 0 inside a block (new_max shows the residual). Other ops edit the down-projection column of one FFN neuron (blk.L.ffn_down.weight, ne0 = ffn, one value per output row j at j*ffn + N) and
writes OUT: a reflink copy of IN with only the touched blocks rewritten in place (byte-exact elsewhere). zero: column = 0; scale: column * k;
replace: column = vec (f32[hidden]); add: column + vec. K/Q types re-quantize the 256-value block each element sits in, so the other
values of that block move by the quantization step: the tool prints the largest such move (requant) next to the column change.
"""
from std.sys import argv
from std.math import isfinite
from std.memory import bitcast
from std.ffi import external_call, c_ssize_t
from ggufx.model import Gguf, open_fd, pread_into, refuse_same_file
from ggufx.dequant import quant_block, deq_rows
from ggufx.quants import quantize_row, FP, BP, T_F32, T_F16, T_BF16
from subprocess import run
from mneuron.ggio import tensor, abs_path
from mneuron.rlog import Rlog


def opt(a: List[String], key: String, dflt: String) -> String:
    for x in a:
        if x.startswith(key + "="):
            return String(x[byte=key.byte_length() + 1:])
    return dflt


def main() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    if len(a) < 3:
        raise Error("usage: surgery IN.gguf OUT.gguf layer= neuron= op=zero|scale|replace|add [k=] [vec=]")
    var inp = a[1]
    var outp = abs_path(a[2])
    var layer = Int(opt(a, "layer", "0"))
    var n = Int(opt(a, "neuron", "0"))
    var op = opt(a, "op", "zero")
    var k = Float32(Float64(opt(a, "k", "1")))
    if op != "zero" and op != "scale" and op != "replace" and op != "add":
        raise Error("op must be zero, scale, replace or add")
    refuse_same_file(inp, outp)
    var g = Gguf(inp)
    var ti = tensor(g, "blk." + String(layer) + ".ffn_down.weight")
    if ti < 0:
        raise Error("no blk." + String(layer) + ".ffn_down.weight")
    var t = g.tensors[ti].ttype
    var ffn = g.tensors[ti].dims[0]
    var h = g.tensors[ti].dims[1]
    if n < 0 or n >= ffn:
        raise Error("neuron out of range 0.." + String(ffn - 1))
    var vec = List[Float32]()
    if op == "replace" or op == "add":
        var vb = open(opt(a, "vec", ""), "r").read_bytes()
        if len(vb) != h * 4:
            raise Error("vec must be f32[" + String(h) + "]")
        vec = List[Float32](length=h, fill=0)
        for i in range(h * 4):
            vec.unsafe_ptr().unsafe_bitcast[UInt8]()[i] = vb[i]
    if op == "zero" and opt(a, "via", "up") == "up":
        var ui = tensor(g, "blk." + String(layer) + ".ffn_up.weight")
        if ui < 0 or g.tensors[ui].dims[1] != ffn:
            raise Error("no blk." + String(layer) + ".ffn_up.weight with " + String(ffn) + " rows (use via=down)")
        var uinfo = quant_block(g.tensors[ui].ttype)
        var urb = g.tensors[ui].dims[0] // max(uinfo[0], 1) * uinfo[1]
        if uinfo[0] == 0:
            raise Error("no dequantizer for the ffn_up type")
        var cpu: List[String] = ["cp", "--reflink=auto", inp, outp]
        if run(cpu).returncode != 0:
            raise Error("cp failed: " + outp)
        var zfd = open_fd(outp, 2)
        var zb = List[UInt8](length=urb, fill=0)
        var zw = external_call["pwrite", c_ssize_t](zfd, zb.unsafe_ptr(), urb, Int64(g.data_offset + g.tensors[ui].offset + n * urb))
        _ = external_call["close", Int32](zfd)
        if Int(zw) != urb:
            raise Error("pwrite failed")
        var zres = "layer=" + String(layer) + " neuron=" + String(n) + " op=zero via=up type=" + String(g.tensors[ui].ttype) + " bytes=" + String(urb)
        print("surgery:", zres, "->", outp)
        var zl = Rlog("surgery", zres, inp)
        var zv = List[Float64]()
        zv.append(Float64(urb))
        zl.done(zres, outp, zv, 1.0, 1e12)
        return
    var float_t = t == T_F32 or t == T_F16 or t == T_BF16
    var info = quant_block(t)
    var be = 1 if float_t else info[0]
    var bb = (4 if t == T_F32 else 2) if float_t else info[1]
    if be == 0:
        raise Error("no dequantizer for ggml type " + String(t))
    var rb = ffn // be * bb
    var base = g.data_offset + g.tensors[ti].offset
    var cpv: List[String] = ["cp", "--reflink=auto", inp, outp]
    if run(cpv).returncode != 0:
        raise Error("cp failed: " + outp)
    var ofd = open_fd(outp, 2)
    var raw = List[UInt8](length=bb, fill=0)
    var xs = List[Float32](length=be, fill=0)
    var sb = List[UInt8](length=256, fill=0)
    var rp = raw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var xp = xs.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var sp = sb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var ex = n % be
    var requant: Float32 = 0
    var dcol: Float64 = 0
    var old_max: Float32 = 0
    var new_max: Float32 = 0
    var vals = List[Float64]()
    for j in range(h):
        var at = base + j * rb + (n // be) * bb
        pread_into(g.fd, raw, 0, bb, at)
        var old = List[Float32](length=be, fill=0)
        if t == T_F32:
            old[0] = bitcast[DType.float32](UInt32(Int(raw[0]) | (Int(raw[1]) << 8) | (Int(raw[2]) << 16) | (Int(raw[3]) << 24)))
        elif t == T_F16:
            old[0] = Float32(bitcast[DType.float16](UInt16(Int(raw[0]) | (Int(raw[1]) << 8))))
        elif t == T_BF16:
            old[0] = bitcast[DType.float32](UInt32((Int(raw[0]) << 16) | (Int(raw[1]) << 24)))
        else:
            deq_rows(t, rp.unsafe_origin_cast[MutUntrackedOrigin](), be, 1, xp.unsafe_origin_cast[MutUntrackedOrigin]())
            for i in range(be):
                old[i] = xs[i]
        var v = old[ex]
        var nv = Float32(0) if op == "zero" else (v * k if op == "scale" else (vec[j] if op == "replace" else v + vec[j]))
        if not isfinite(nv):
            raise Error("non-finite new value at output row " + String(j))
        for i in range(be):
            xs[i] = old[i]
        xs[ex] = nv
        if t == T_F32:
            var bits = Int(bitcast[DType.uint32](nv))
            for i in range(4):
                raw[i] = UInt8((bits >> (8 * i)) & 255)
        elif t == T_F16:
            var bits = Int(bitcast[DType.uint16](Float16(nv)))
            raw[0] = UInt8(bits & 255)
            raw[1] = UInt8(bits >> 8)
        elif t == T_BF16:
            var bits = Int(bitcast[DType.uint32](nv))
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
            raw[0] = UInt8(bits & 255)
            raw[1] = UInt8(bits >> 8)
        else:
            quantize_row(t, xp, rp, be, sp)
            var chk = List[Float32](length=be, fill=0)
            deq_rows(t, rp.unsafe_origin_cast[MutUntrackedOrigin](), be, 1, chk.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
            for i in range(be):
                if i != ex:
                    requant = max(requant, abs(chk[i] - old[i]))
                else:
                    nv = chk[i]
            _ = len(chk)
        dcol += Float64(abs(nv - v))
        old_max = max(old_max, abs(v))
        new_max = max(new_max, abs(nv))  # nv is what was written, read back through the dequantizer
        if j < 8:
            vals.append(Float64(nv))
        var w = external_call["pwrite", c_ssize_t](ofd, raw.unsafe_ptr(), bb, Int64(at))
        if Int(w) != bb:
            raise Error("pwrite failed at row " + String(j))
    _ = len(raw) + len(xs) + len(sb)
    _ = external_call["close", Int32](ofd)
    var res = "layer=" + String(layer) + " neuron=" + String(n) + " op=" + op + " type=" + String(t) + " col_abs_change=" + String(dcol) + " old_max=" + String(old_max) + " new_max=" + String(new_max) + " requant_max=" + String(requant)
    print("surgery:", res, "->", outp)
    var rl = Rlog("surgery", res, inp)
    var ok = List[Float64]()
    ok.append(dcol)
    ok.append(Float64(requant))
    rl.done(res, outp, ok, 0.0, 1e12)
