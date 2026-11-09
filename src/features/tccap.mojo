"""tccap MODEL.gguf CORPUS LAYER OUT_DIR [--windows N] [--seg S] [--chunk W]: CPU capture (dense llama/qwen2/granite) of one block's MLP input
(post ffn_norm) and MLP output, windows of S tokens (default 128) from one llama-tokenize pass, in the capture format v1
(docs/format-capture.md): OUT_DIR/L<layer>.f16 records [x|y] (fields "xy"), tokens.i32, req.i32 (window number), meta.json."""
from std.sys import argv, exit
from std.os import makedirs
from std.memory import bitcast
from ggufx.model import Gguf
from mneuron.ggio import kv_int, kv_str, put_u32, write_file, llama_ids
from mneuron.fwd import forward
from mneuron.store import read_text
from mneuron.rlog import Rlog


def put_h(mut b: List[UInt8], v: Float32):
    var u = bitcast[DType.uint16](v.cast[DType.float16]())
    b.append(UInt8(Int(u) & 255))
    b.append(UInt8(Int(u) >> 8))


def main() raises:
    var a = List[String]()
    var windows = 800
    var seg = 128
    var chunk = 32
    var av = argv()
    var i = 1
    while i < len(av):
        var t = String(av[i])
        if t == "--windows" and i + 1 < len(av):
            windows = Int(String(av[i + 1]))
            i += 2
        elif t == "--seg" and i + 1 < len(av):
            seg = Int(String(av[i + 1]))
            i += 2
        elif t == "--chunk" and i + 1 < len(av):
            chunk = Int(String(av[i + 1]))
            i += 2
        else:
            a.append(t)
            i += 1
    if len(a) < 4:
        print("usage: tccap MODEL.gguf CORPUS LAYER OUT_DIR [--windows N] [--seg S] [--chunk W]")
        exit(1)
    var layer = Int(a[2])
    var out = a[3]
    var rl = Rlog("tccap", String(" ").join(a), a[0])
    makedirs(out, exist_ok=True)
    var g = Gguf(a[0])
    var ids = llama_ids(a[0], read_text(a[1]), out + "/corpus-tmp.txt")
    var nw = min(windows, len(ids) // seg)
    var xb = List[UInt8]()
    var yb = List[UInt8]()
    var h = 0
    var w0 = 0
    while w0 < nw:
        var w1 = min(w0 + chunk, nw)
        var seq = List[Int]()
        for k in range(w0 * seg, w1 * seg):
            seq.append(ids[k])
        var r = forward(g, seq, List[Int](), List[Int](), seg, layer)
        ref m = r[2]
        h = len(m[0]) // len(seq)
        for t in range(len(seq)):
            for j in range(h):
                put_h(xb, m[0][t * h + j])
                put_h(yb, m[1][t * h + j])
        w0 = w1
    var n = nw * seg
    var rec = List[UInt8]()  # records [x | y] per token, the capture format v1 (docs/format-capture.md), fields "xy"
    for t in range(n):
        for q in range(2):
            for k in range(h * 2):
                rec.append((xb if q == 0 else yb)[t * h * 2 + k])
    write_file(out + "/L" + String(layer) + ".f16", rec)
    var tk = List[UInt8]()
    var rq = List[UInt8]()
    for t in range(n):
        put_u32(tk, ids[t])
        put_u32(rq, t // seg + 1)
    write_file(out + "/tokens.i32", tk)
    write_file(out + "/req.i32", rq)
    var meta = '{"model":"' + a[0] + '","dtype":"f16","h":' + String(h) + ',"n_layers":' + String(kv_int(g, kv_str(g, "general.architecture") + ".block_count")) + ',"fields":"xy","layers":[' + String(layer) + "]}\n"
    var mb = List[UInt8]()
    for c in meta.as_bytes():
        mb.append(c)
    write_file(out + "/meta.json", mb)
    var vals = List[Float64]()
    vals.append(Float64(n))
    rl.done("tokens=" + String(n) + " h=" + String(h) + " layer=" + String(layer), out + "/meta.json", vals, 1.0, 1e12)
    print("tokens", n, "h", h)
