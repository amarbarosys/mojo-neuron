"""Reverse MODEL.gguf OUTDIR prompts=FILE sys_a=TEXT sys_b=TEXT [top=100 swap=0 chat=1 baro= spark= pack= runner=]

Persona contrast: the same prompts run once under system prompt A and once under B in one resident spark; the residual
after every layer at the last prompt token is dumped (BARO_DUMP_LAYERS), d_L = mean_A - mean_B. Every neuron of every
layer is then ranked by cosine(W_down[:, n], d_L). Writes OUTDIR/reverse.jsonl (top `top` by cosine, best first, then
the `top` most negative) and OUTDIR/contrast.f32 (f32[n_layer * hidden], d_L: usable as a direction).
swap=1 contrasts B - A (the mutant of the gate: the ranking must flip).
"""
from std.sys import argv, exit
from std.os import getenv
from std.math import sqrt
from std.memory import bitcast
from ggufx.model import Gguf, Out
from tokenizer import Tokenizer
from minja import render_chat
from lens import tensor, load_rows
from cvec import model_layers
from sweep import opt, jesc, read_text, run_spark


def run_all() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    if len(a) < 3:
        raise Error("usage: reverse MODEL.gguf OUTDIR prompts=FILE sys_a=TEXT sys_b=TEXT [top=100 swap=0 chat=1]")
    var model = a[1]
    var odir = a[2] if a[2].startswith("/") else getenv("PWD") + "/" + a[2]
    var top = Int(opt(a, "top", "100"))
    var swap = opt(a, "swap", "0") == "1"
    var g = Gguf(model)
    var nl = model_layers(g)
    var h = g.tensors[tensor(g, "blk.0.ffn_down.weight")].dims[1]
    var tok = Tokenizer(model)
    var prompts = List[String]()
    for l in read_text(opt(a, "prompts", "")).split("\n"):
        if l.byte_length() > 0:
            prompts.append(String(l))
    var m = len(prompts)
    var reqs = List[String]()
    var sys = List[String]()
    sys.append(opt(a, "sys_a", ""))
    sys.append(opt(a, "sys_b", ""))
    for who in range(2):
        for p in prompts:
            var ids: List[Int]
            if opt(a, "chat", "1") == "1":
                var msgs = String("[")
                if sys[who].byte_length() > 0:
                    msgs += "{\"role\":\"system\",\"content\":\"" + jesc(sys[who]) + "\"},"
                msgs += "{\"role\":\"user\",\"content\":\"" + jesc(p) + "\"}]"
                var cj = "{\"messages\":" + msgs + ",\"add_generation_prompt\":true}"
                ids = tok.encode(render_chat(tok.chat_template, cj, tok.token_str(tok.bos_id), tok.token_str(tok.eos_id), tok.token_str(tok.pad_id)), add_special=False)
            else:
                ids = tok.encode(sys[who] + p)
            var pl = String()
            for t in range(len(ids)):
                pl += ("," if t > 0 else "") + String(ids[t])
            reqs.append("{\"id\":" + String(len(reqs) + 1) + ",\"prompt\":[" + pl + "],\"n\":1}")
    var dump = odir + "/contrast.dump.bin"
    with open(dump, "w") as f:
        f.write("")
    _ = run_spark(reqs, a, odir, "BARO_DUMP_LAYERS='" + dump + "' BARO_PREFILL_ROWS=0")
    var raw = open(dump, "r").read_bytes()
    if len(raw) != 2 * m * nl * h * 4:
        raise Error("dump has " + String(len(raw)) + " bytes, want " + String(2 * m * nl * h * 4))
    var f = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var d = List[Float32](length=nl * h, fill=0)
    for r in range(m):
        for i in range(nl * h):
            d[i] += (f[(m + r) * nl * h + i] - f[r * nl * h + i] if swap else f[r * nl * h + i] - f[(m + r) * nl * h + i]) / Float32(m)
    var o = Out(odir + "/contrast.f32")
    for i in range(nl * h):
        o.put(Int(bitcast[DType.uint32](d[i])), 4)
    o.close()
    var all = List[Tuple[Float32, Int]]()
    for L in range(nl):
        var di = tensor(g, "blk." + String(L) + ".ffn_down.weight")
        var ffn = g.tensors[di].dims[0]
        var D = load_rows(g, di, 0, h)
        var dn: Float64 = 0
        for j in range(h):
            dn += Float64(d[L * h + j]) ** 2
        var dot = List[Float32](length=ffn, fill=0)
        var nn = List[Float32](length=ffn, fill=0)
        for j in range(h):
            var dj = d[L * h + j]
            for n in range(ffn):
                var x = D[j * ffn + n]
                dot[n] += x * dj
                nn[n] += x * x
        for n in range(ffn):
            var den = sqrt(Float64(nn[n])) * sqrt(dn)
            all.append((Float32(Float64(dot[n]) / den) if den > 0 else Float32(0), L * 1000000 + n))
    sort(all)
    var js = String()
    for q in range(2 * top):
        var it = all[len(all) - 1 - q] if q < top else all[q - top]
        js += "{\"layer\":" + String(it[1] // 1000000) + ",\"neuron\":" + String(it[1] % 1000000) + ",\"cos\":" + String(it[0]) + "}\n"
    with open(odir + "/reverse.jsonl", "w") as o2:
        o2.write(js)
    print("reverse:", nl, "layers,", len(all), "neurons ranked ->", odir + "/reverse.jsonl")


def main():
    try:
        run_all()
    except e:
        print("FAIL reverse:", e)
        exit(1)
