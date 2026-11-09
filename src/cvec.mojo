"""Cvec MODEL RECIPE OUT [RHO.f32]: turn `L:N:dose,...` into steering vectors.

Writes OUT.f32 (f32[n_layer * hidden], row L is added after layer L: the BARO_STEER file) and
OUT.cvec.gguf (llama.cpp `--control-vector`, tensors `direction.L` for the non-zero layers L >= 1).
Unit vector: dose * W_down[:, N] (raw), or dose * rho[L] * unit(W_down[:, N]) when RHO.f32
(f32[n_layer], docs/format.md model.json rho) is given (rel).
"""
from std.sys import argv, exit
from std.math import sqrt
from std.memory import bitcast
from ggufx.model import Gguf, Out, pad_to
from ggufx.values import uint_at, peek
from lens import tensor, load_rows

comptime MAGIC = 0x46554747
comptime ALIGN = 32


def dose_of(s: String) raises -> Float64:
    return Float64(String(s[byte=1:]) if s.startswith("+") else s)


def put_kv_str(mut o: Out, key: String, val: String) raises:
    o.put_str(key)
    o.put(8, 4)
    o.put_str(val)


def arch_of(g: Gguf) raises -> String:
    for k in range(len(g.kvs)):
        if g.kvs[k].key == "general.architecture":
            var n = Int(uint_at(g, g.kvs[k].start, 8))
            return String(unsafe_from_utf8=Span(peek(g, g.kvs[k].start + 8, n)))
    return "unknown"


def kv_int(g: Gguf, key: String) raises -> Int:
    """Integer KV (u32 or u64 payload), -1 when absent."""
    for k in range(len(g.kvs)):
        if g.kvs[k].key == key:
            return Int(uint_at(g, g.kvs[k].start, 8 if g.kvs[k].vtype == 10 else 4))
    return -1


def model_layers(g: Gguf) raises -> Int:
    """Layers of the residual stream: blocks with an ffn_down, minus multi-token-prediction heads (block_count - nextn_predict_layers)."""
    var n = 0
    while tensor(g, "blk." + String(n) + ".ffn_down.weight") >= 0:
        n += 1
    if n == 0:
        raise Error("no blk.*.ffn_down.weight")
    var arch = arch_of(g)
    var bc = kv_int(g, arch + ".block_count")
    var nx = kv_int(g, arch + ".nextn_predict_layers")
    if bc > 0 and bc - max(nx, 0) < n:
        n = bc - max(nx, 0)
    return n


def write_cvec(model: String, recipe: String, path: String, rho: List[Float32]) raises -> Tuple[Int, Float64]:
    """Write PATH.f32 and PATH.cvec.gguf for `recipe`; rho empty = raw units. Returns (steered layers, max |component|)."""
    var g = Gguf(model)
    var n_layer = model_layers(g)
    var h = g.tensors[tensor(g, "blk.0.ffn_down.weight")].dims[1]
    if len(rho) > 0 and len(rho) != n_layer:
        raise Error("rho must hold " + String(n_layer) + " values")
    var vec = List[Float32](length=n_layer * h, fill=0)
    var Ls = List[Int]()
    var Ns = List[Int]()
    var Ds = List[Float64]()
    for u in recipe.split(","):
        var p = u.split(":")
        if len(p) != 3:
            raise Error("bad unit '" + String(u) + "', want L:N:dose")
        Ls.append(Int(p[0]))
        Ns.append(Int(p[1]))
        Ds.append(dose_of(String(p[2])))
    var done = List[Bool](length=len(Ls), fill=False)
    # one dequantisation per layer (never more than one layer in memory): units of a layer are served together
    for first in range(len(Ls)):
        if done[first]:
            continue
        var L = Ls[first]
        var di = tensor(g, "blk." + String(L) + ".ffn_down.weight")
        if L < 0 or L >= n_layer or di < 0:
            raise Error("layer " + String(L) + " out of range 0.." + String(n_layer - 1))
        var ffn = g.tensors[di].dims[0]
        var D = load_rows(g, di, 0, h)
        for q in range(first, len(Ls)):
            if Ls[q] != L:
                continue
            done[q] = True
            var n = Ns[q]
            if n < 0 or n >= ffn:
                raise Error("neuron " + String(n) + " out of range 0.." + String(ffn - 1))
            var s: Float64 = 0
            for j in range(h):
                s += Float64(D[j * ffn + n]) ** 2
            var k = Ds[q]
            if len(rho) > 0:
                k = Ds[q] * Float64(rho[L]) / sqrt(s)
            for j in range(h):
                vec[L * h + j] += Float32(k * Float64(D[j * ffn + n]))
    var nz = List[Int]()
    for L in range(1, n_layer):
        for j in range(h):
            if vec[L * h + j] != 0:
                nz.append(L)
                break
    var raw = Out(path + ".f32")
    for i in range(n_layer * h):
        raw.put(Int(bitcast[DType.uint32](vec[i])), 4)
    raw.close()
    var o = Out(path + ".cvec.gguf")
    o.put(MAGIC, 4)
    o.put(3, 4)
    o.put(len(nz), 8)
    o.put(3, 8)
    put_kv_str(o, "general.architecture", "controlvector")
    put_kv_str(o, "controlvector.model_hint", arch_of(g))
    o.put_str("controlvector.layer_count")
    o.put(4, 4)
    o.put(n_layer, 4)
    var at = 0
    for L in nz:
        o.put_str("direction." + String(L))
        o.put(1, 4)
        o.put(h, 8)
        o.put(0, 4)
        o.put(at, 8)
        at += pad_to(h * 4, ALIGN)
    o.pad(ALIGN)
    for L in nz:
        for j in range(h):
            o.put(Int(bitcast[DType.uint32](vec[L * h + j])), 4)
        o.pad(ALIGN)
    o.close()
    var mx = Float64(0)
    for i in range(n_layer * h):
        mx = max(mx, abs(Float64(vec[i])))  # NaN does not win max: caught below by the caller's finite check
        if vec[i] != vec[i]:
            return (len(nz), Float64(vec[i]))
    return (len(nz), mx)


def run() raises:
    var a = argv()
    if len(a) < 4:
        raise Error("usage: cvec MODEL RECIPE OUT [RHO.f32]")
    var rho = List[Float32]()
    if len(a) > 4:
        var b = open(String(a[4]), "r").read_bytes()
        rho = List[Float32](length=len(b) // 4, fill=0)
        for i in range(len(rho) * 4):
            rho.unsafe_ptr().unsafe_bitcast[UInt8]()[i] = b[i]
    var n = write_cvec(String(a[1]), String(a[2]), String(a[3]), rho)[0]
    print("cvec:", n, "layers ->", String(a[3]) + ".cvec.gguf", String(a[3]) + ".f32")


def main():
    try:
        run()
    except e:
        print("FAIL cvec:", e)
        exit(1)
