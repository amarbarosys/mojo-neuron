"""Sweep MODEL.gguf OUTDIR key=value ...: neurons x doses x prompts through ONE resident mojo-baro spark.

mode=sweep (default)  rollouts, appended to OUTDIR/<layer>.sweep.jsonl (docs/format.md)
mode=slens            steer lens at +dose, mean over the prompts as contexts: OUTDIR/<layer>.steer[.<dose>].bin

keys: layer= neurons=100,105-110|all doses=0.4,0.8 prompts=FILE (one prompt per line) ctrl=N (random controls)
      n=24 (generated tokens) k=16 (slens) rho=FILE (f32[n_layer], tools/rho.py) chat=1 (chat template)
      baro=DIR spark=BIN pack=DIR runner="gpu-wait run --timeout 900 --" (none = run directly)
The requests go to spark in one batch (startup paid once); vectors come from a bank file, one seek per unit.
"""
from std.sys import argv, exit
from std.os import getenv
from std.math import sqrt, cos, sin, log
from std.memory import bitcast
from std.time import perf_counter_ns
from ggufx.model import Gguf, Out
from subprocess import run
from tokenizer import Tokenizer
from minja import render_chat
from lens import tensor, load_rows
from cvec import model_layers

comptime BANK = "bank.f32"


def opt(a: List[String], key: String, dflt: String) -> String:
    for x in a:
        if x.startswith(key + "="):
            return String(x[byte=key.byte_length() + 1:])
    return dflt


def ints(s: String, n_all: Int) raises -> List[Int]:
    var out = List[Int]()
    if s == "all":
        for i in range(n_all):
            out.append(i)
        return out^
    for p in s.split(","):
        var r = p.split("-")
        if len(r) == 2:
            for i in range(Int(r[0]), Int(r[1]) + 1):
                out.append(i)
        else:
            out.append(Int(p))
    return out^


def jesc(s: String) -> String:
    return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\t", "\\t")


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def num(b: Span[UInt8, _], mut i: Int) raises -> Float64:
    """Parse the number at b[i], advance i past it."""
    var j = i
    while j < len(b) and (b[j] == 45 or b[j] == 43 or b[j] == 46 or b[j] == 101 or b[j] == 69 or (b[j] >= 48 and b[j] <= 57)):
        j += 1
    var v = Float64(String(unsafe_from_utf8=b[i:j])) if j > i else 0.0
    i = j
    return v


def at(line: String, key: String, mut i: Int) -> Bool:
    var p = line.find(key, i)
    if p < 0:
        return False
    i = p + key.byte_length()
    return True


struct Pair(Copyable, Movable):
    var id: List[Int]
    var v: List[Float32]

    def __init__(out self):
        self.id = List[Int]()
        self.v = List[Float32]()


def pairs(line: String, key: String) raises -> Pair:
    var r = Pair()
    var i = 0
    if not at(line, key, i):
        return r^
    var b = line.as_bytes()
    while i < len(b) and b[i] != 93:
        if b[i] == 91:
            i += 1
            r.id.append(Int(num(b, i)))
            i += 1
            r.v.append(Float32(num(b, i)))
        i += 1
    return r^


struct Unit(Copyable, Movable):
    var label: String  # "base", "ctrl0", or the neuron number
    var row: Int  # bank row, -1 for base
    var neuron: Int

    def __init__(out self, label: String, row: Int, neuron: Int):
        self.label = label
        self.row = row
        self.neuron = neuron


struct Req(Copyable, Movable):
    var unit: Int
    var dose: Float64
    var ctx: Int  # prompt index
    var toks: List[Int]
    var up: Pair
    var dn: Pair
    var ms: Float64
    var kl: Float64
    var ndone: Int
    var nunit: Int
    var fin_len: Bool
    var want: Bool  # a steer lens line is expected from this request (per-request path, last context of a group)

    def __init__(out self, unit: Int, dose: Float64, ctx: Int):
        self.unit = unit
        self.dose = dose
        self.ctx = ctx
        self.toks = List[Int]()
        self.up = Pair()
        self.dn = Pair()
        self.ms = 0.0
        self.kl = -1.0
        self.ndone = 0
        self.nunit = 0
        self.fin_len = False
        self.want = False


def run_spark(reqs: List[String], a: List[String], odir: String, extra: String = "") raises -> List[String]:
    var baro = opt(a, "baro", String("$HOME/Projects/mojo/mojo-baro-lanes/ablate"))
    var spark = opt(a, "spark", baro + "/.work/steer/spark-q05")
    var pack = opt(a, "pack", baro + "/.work/abl/q/pack")
    var runner = opt(a, "runner", String("gpu-wait run --timeout 900 --"))
    var rq = odir + "/.req.jsonl"
    var out = odir + "/.resp.jsonl"
    var body = String()
    for r in reqs:
        body += r + "\n"
    with open(rq, "w") as f:
        f.write(body)
    var inner = "cd '" + baro + "' && env BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 " + extra + " " + opt(a, "env", "") + " BARO_PACK='" + pack + "' '" + spark + "' < '" + rq + "' > '" + out + "' 2> '" + out + ".err'"
    var argv = List[String]()
    if runner != "none":
        for w in runner.split(" "):
            argv.append(String(w))
    argv.append("bash")
    argv.append("-c")
    argv.append(inner)
    var r = run(argv, timeout=3000.0)
    if r.returncode != 0:
        raise Error("spark failed (exit " + String(r.returncode) + "): see " + out + ".err")
    var lines = List[String]()
    for l in read_text(out).split("\n"):
        lines.append(String(l))
    return lines^


def collect(lines: List[String], mut rs: List[Req], nu: Int, n_gen: Int, nops: Int, preq: Bool = False) raises:
    """Parse spark's response and require it complete: every request answered exactly once, every token line attached to a
    known request, every steer-lens unit present once. Anything else (dropped, duplicated, malformed, unknown, out of range)
    raises, so no record is written from an empty Req."""
    var op_done = List[Int](length=nops, fill=0)
    for l in lines:
        if not l.startswith("{\"id\":"):
            continue  # status lines (ready, banners)
        var i = 0
        _ = at(l, "\"id\":", i)
        var b = l.as_bytes()
        if i >= len(b) or b[i] < 48 or b[i] > 57:
            raise Error("malformed response line (no id): " + String(l))
        var id = Int(num(b, i)) - 1
        if l.find("\"error\"") >= 0:
            raise Error("spark request " + String(id + 1) + ": " + l)
        if nu > 0:
            if id < 0 or id >= nops:
                raise Error("response id " + String(id + 1) + " out of range (1.." + String(nops) + "): " + String(l))
            if l.find("\"unit\":") >= 0:
                var j = 0
                _ = at(l, "\"unit\":", j)
                var u = Int(num(b, j))
                if u < 0 or u >= nu:
                    raise Error("steer lens unit " + String(u) + " out of range in: " + String(l))
                var ix = id * nu + u
                if rs[ix].nunit > 0:
                    raise Error("duplicate steer lens line for dose " + String(id) + " unit " + String(u))
                rs[ix].up = pairs(l, "\"up\":[")
                rs[ix].dn = pairs(l, "\"dn\":[")
                j = 0
                if not at(l, "\"kl\":", j) or len(rs[ix].up.id) == 0 or len(rs[ix].dn.id) == 0:
                    raise Error("malformed steer lens line: " + String(l))
                rs[ix].kl = num(b, j)
                rs[ix].nunit = 1
            elif l.find("\"done\"") >= 0:
                op_done[id] += 1
                if op_done[id] > 1:
                    raise Error("duplicate done for op " + String(id + 1))
            else:
                raise Error("unrecognised response line: " + String(l))
            continue
        if id < 0 or id >= len(rs):
            raise Error("response id " + String(id + 1) + " out of range (1.." + String(len(rs)) + "): " + String(l))
        if preq and l.find("\"slens\"") >= 0:
            if rs[id].nunit > 0:
                raise Error("duplicate steer lens line for request " + String(id + 1))
            rs[id].up = pairs(l, "\"up\":[")
            rs[id].dn = pairs(l, "\"dn\":[")
            var j = 0
            if not at(l, "\"kl\":", j) or len(rs[id].up.id) == 0 or len(rs[id].dn.id) == 0:
                raise Error("malformed steer lens line: " + String(l))
            rs[id].kl = num(b, j)
            rs[id].nunit = 1
        elif l.find("\"done\"") >= 0:
            rs[id].ndone += 1
            if rs[id].ndone > 1:
                raise Error("duplicate done for request " + String(id + 1))
            var j = 0
            var ms = 0.0
            if at(l, "\"prefill_s\":", j):
                ms += num(b, j) * 1000
            j = 0
            if at(l, "\"decode_s\":", j):
                ms += num(b, j) * 1000
            rs[id].ms = ms
            rs[id].fin_len = l.find("\"finish\":\"length\"") >= 0
        elif l.find("\"tok\":") >= 0:
            var j = 0
            _ = at(l, "\"tok\":", j)
            if j >= len(b) or not ((b[j] >= 48 and b[j] <= 57) or b[j] == 45):
                raise Error("malformed token line: " + String(l))
            if rs[id].ndone > 0:
                raise Error("token after done for request " + String(id + 1))
            rs[id].toks.append(Int(num(b, j)))
        else:
            raise Error("unrecognised response line: " + String(l))
    if nu > 0:
        for d in range(nops):
            if op_done[d] != 1:
                raise Error("op " + String(d + 1) + " has no done line: spark's response is incomplete")
        for k in range(len(rs)):
            if rs[k].nunit != 1:
                raise Error("steer lens unit " + String(k % nu) + " of dose " + String(k // nu) + " missing from spark's response")
        return
    for k in range(len(rs)):
        if rs[k].ndone != 1:
            raise Error("request " + String(k + 1) + " has no done line: spark's response is incomplete")
        if rs[k].want and rs[k].nunit != 1:
            raise Error("request " + String(k + 1) + " has no steer lens line: the response is incomplete")
        if len(rs[k].toks) < 1 or len(rs[k].toks) > n_gen or (rs[k].fin_len and len(rs[k].toks) != n_gen):
            raise Error("request " + String(k + 1) + " returned " + String(len(rs[k].toks)) + " tokens, want " + String(n_gen))


def run_all() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    if len(a) < 3:
        raise Error("usage: sweep MODEL.gguf OUTDIR layer=.. neurons=.. doses=.. prompts=FILE [mode=slens] ...")
    var model = a[1]
    var odir = a[2] if a[2].startswith("/") else getenv("PWD") + "/" + a[2]
    var mode = opt(a, "mode", "sweep")
    var layer = Int(opt(a, "layer", "0"))
    var n_gen = Int(opt(a, "n", "24")) if mode == "sweep" else 1
    var kk = Int(opt(a, "k", "16"))
    var nctrl = Int(opt(a, "ctrl", "0"))
    var g = Gguf(model)
    var di = tensor(g, "blk." + String(layer) + ".ffn_down.weight")
    if di < 0:
        raise Error("no blk." + String(layer) + ".ffn_down.weight")
    var ffn = g.tensors[di].dims[0]
    var h = g.tensors[di].dims[1]
    var nl = model_layers(g)
    var rho = List[Float32](length=nl, fill=1.0)
    var rf = opt(a, "rho", "")
    if rf != "":
        var rb = open(rf, "r").read_bytes()
        if len(rb) != nl * 4:
            raise Error("rho file must be f32[" + String(nl) + "]")
        for i in range(nl * 4):
            rho.unsafe_ptr().unsafe_bitcast[UInt8]()[i] = rb[i]
    else:
        print("WARN no rho=: doses are raw multiples of the unit vector")
    var neurons = ints(opt(a, "neurons", "0"), ffn)
    var doses = List[Float64]()
    var gain = opt(a, "gains", "") != ""  # neuron gain ablation (L:N:k), instead of an added vector
    if gain and nctrl > 0:
        raise Error("ctrl= has no meaning with gains=")
    for d in (opt(a, "gains", "") if gain else opt(a, "doses", "0.5")).split(","):
        doses.append(Float64(d))
    var parts = model.split("/")
    var stem = String(parts[len(parts) - 1]).replace(".gguf", "").lower()
    var tok = Tokenizer(model)
    var prompts = List[String]()
    for l in read_text(opt(a, "prompts", "")).split("\n"):
        if l.byte_length() > 0:
            prompts.append(String(l).replace("\\n", "\n"))  # a literal \n in the file is a newline (chat=0 prompts with a template)
    var ids = List[List[Int]]()
    for p in prompts:
        if opt(a, "chat", "0") == "1":
            var cj = "{\"messages\":[{\"role\":\"user\",\"content\":\"" + jesc(p) + "\"}],\"add_generation_prompt\":true}"
            var txt = render_chat(tok.chat_template, cj, tok.token_str(tok.bos_id), tok.token_str(tok.eos_id), tok.token_str(tok.pad_id))
            ids.append(tok.encode(txt, add_special=False))
        else:
            ids.append(tok.encode(p))
    # bank: unit vectors of the neurons, then random unit controls
    var D = load_rows(g, di, 0, h)
    var units = List[Unit]()
    units.append(Unit("base", -1, -1))
    var bank = Out(odir + "/" + String(layer) + "." + BANK)
    var row = 0
    for n in neurons:
        var s: Float64 = 0
        for j in range(h):
            s += Float64(D[j * ffn + n]) ** 2
        var inv = 1.0 / sqrt(s) if s > 0 else 0.0
        for j in range(h):
            bank.put(Int(bitcast[DType.uint32](Float32(Float64(D[j * ffn + n]) * inv))), 4)
        units.append(Unit(String(n), row, n))
        row += 1
    var seed: UInt64 = 0x9E3779B97F4A7C15 ^ UInt64(layer)
    for c in range(nctrl):
        var v = List[Float64](length=h, fill=0)
        var s: Float64 = 0
        for j in range(h):
            var u = List[Float64]()
            for _ in range(2):
                seed ^= seed << 13
                seed ^= seed >> 7
                seed ^= seed << 17
                u.append((Float64(seed >> 11) + 1.0) / 9007199254740994.0)
            v[j] = sqrt(-2.0 * log(u[0])) * cos(6.283185307179586 * u[1])
            s += v[j] * v[j]
        for j in range(h):
            bank.put(Int(bitcast[DType.uint32](Float32(v[j] / sqrt(s)))), 4)
        units.append(Unit("ctrl" + String(c), row, -1))
        row += 1
    bank.close()
    # requests
    var reqs = List[String]()
    var meta = List[Req]()
    var bankp = odir + "/" + String(layer) + "." + BANK
    var preq = mode == "slens" and opt(a, "req", "0") == "1"
    if preq:
        # per-request steer lens (engine.mojo): a base request per context, then per unit and dose one request per context, the
        # sum over contexts accumulated inside the engine (slens_flags bit0 start, bit1 emit)
        for pi in range(len(ids)):
            var pl = String()
            for t in range(len(ids[pi])):
                pl += ("," if t > 0 else "") + String(ids[pi][t])
            meta.append(Req(0, 0.0, pi))
            reqs.append("{\"id\":" + String(len(meta)) + ",\"prompt\":[" + pl + "],\"n\":1,\"slens_ctx\":" + String(pi) + ",\"slens_k\":0}")
        for ui in range(1, len(units)):
            for d in doses:
                for pi in range(len(ids)):
                    var pl = String()
                    for t in range(len(ids[pi])):
                        pl += ("," if t > 0 else "") + String(ids[pi][t])
                    var r = Req(ui, d, pi)
                    r.want = pi == len(ids) - 1
                    meta.append(r^)
                    reqs.append("{\"id\":" + String(len(meta)) + ",\"prompt\":[" + pl + "],\"n\":1,\"slens_ctx\":" + String(pi) + ",\"slens_k\":" + String(kk)
                        + ",\"slens_flags\":" + String((1 if pi == 0 else 0) + (2 if pi == len(ids) - 1 else 0))
                        + ",\"steer_bank\":\"" + bankp + "\",\"steer_layer\":" + String(layer) + ",\"steer_row\":" + String(units[ui].row)
                        + ",\"steer_scale\":" + String(d * Float64(rho[layer])) + "}")
    elif mode == "slens":
        # one batched op per dose: all bank rows x all prompts inside spark (docs/format.md steer.bin)
        var pls = String()
        for pi in range(len(ids)):
            var one = String()
            for t in range(len(ids[pi])):
                one += ("," if t > 0 else "") + String(ids[pi][t])
            pls += ("," if pi > 0 else "") + "[" + one + "]"
        for di in range(len(doses)):
            reqs.append("{\"op\":\"slens\",\"id\":" + String(di + 1) + ",\"layer\":" + String(layer) + ",\"bank\":\"" + bankp + "\",\"row0\":0,\"rows\":"
                + String(len(units) - 1) + ",\"scale\":" + String(doses[di] * Float64(rho[layer])) + ",\"k\":" + String(kk) + ",\"prompts\":[" + pls + "]}")
            for ui in range(1, len(units)):
                meta.append(Req(ui, doses[di], 0))
    else:
        for pi in range(len(ids)):
            var pl = String()
            for t in range(len(ids[pi])):
                pl += ("," if t > 0 else "") + String(ids[pi][t])
            var common = "\"prompt\":[" + pl + "],\"n\":" + String(n_gen)
            # base request: slens base (k = 0) or plain rollout
            meta.append(Req(0, 0.0, pi))
            reqs.append("{\"id\":" + String(len(meta)) + "," + common + (",\"slens_ctx\":" + String(pi) + ",\"slens_k\":0" if mode == "slens" else "") + "}")
        for ui in range(1, len(units)):
            for d in doses:
                for pi in range(len(ids)):
                    var pl = String()
                    for t in range(len(ids[pi])):
                        pl += ("," if t > 0 else "") + String(ids[pi][t])
                    meta.append(Req(ui, d, pi))
                    reqs.append("{\"id\":" + String(len(meta)) + ",\"prompt\":[" + pl + "],\"n\":" + String(n_gen)
                        + (",\"slens_ctx\":" + String(pi) + ",\"slens_k\":" + String(kk) + ",\"slens_flags\":" + String((1 if pi == 0 else 0) + (2 if pi == len(ids) - 1 else 0)) if mode == "slens" else "")
                        + (",\"gain\":\"" + String(layer) + ":" + String(units[ui].neuron) + ":" + String(d) + "\"}" if gain else
                           ",\"steer_bank\":\"" + bankp + "\",\"steer_layer\":" + String(layer) + ",\"steer_row\":" + String(units[ui].row)
                           + ",\"steer_scale\":" + String(d * Float64(rho[layer])) + "}"))
    print("sweep:", len(reqs), "requests,", len(units) - 1, "units,", len(doses), "doses,", len(ids), "prompts, mode", mode)
    var t0 = perf_counter_ns()
    var rfile = opt(a, "resp", "")  # parse a saved response instead of running spark (the completeness gate feeds mutants through this)
    var lines: List[String]
    if rfile != "":
        lines = List[String]()
        for l in read_text(rfile).split("\n"):
            lines.append(String(l))
    else:
        lines = run_spark(reqs, a, odir)
    collect(lines, meta, len(units) - 1 if (mode == "slens" and not preq) else 0, n_gen, len(doses), preq)
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    print("sweep: spark wall", wall, "s,", wall / Float64(len(reqs)) * 1000, "ms/request")
    if mode == "sweep":
        var js = String()
        for r in meta:
            ref u = units[r.unit]
            var lab = u.label if u.neuron < 0 else u.label
            var nj = "\"" + lab + "\"" if u.neuron < 0 else lab
            var base_toks = meta[r.ctx].toks.copy()
            var fd = -1
            for t in range(min(len(r.toks), len(base_toks))):
                if r.toks[t] != base_toks[t]:
                    fd = t
                    break
            var tl = String()
            for t in range(len(r.toks)):
                tl += ("," if t > 0 else "") + String(r.toks[t])
            js += "{\"model\":\"" + jesc(stem) + "\",\"layer\":" + String(layer) + ",\"neuron\":" + nj
                + ",\"prompt\":\"" + jesc(prompts[r.ctx]) + "\",\"dose\":" + String(r.dose) + ",\"dose_unit\":\"" + ("gain" if gain else "rel") + "\",\"text\":\"" + jesc(tok.decode(r.toks))
                + "\",\"tokens\":[" + tl + "],\"prompt_id\":" + String(r.ctx) + ",\"first_diff\":" + String(fd) + ",\"ms\":" + String(Int(r.ms)) + "}\n"
        with open(odir + "/" + String(layer) + ".sweep.jsonl", "a") as f:
            f.write(js)
        print("sweep: wrote", odir + "/" + String(layer) + ".sweep.jsonl")
        return
    # slens: per (unit, dose) mean over contexts of dlp, top k up and down
    for d in doses:
        var o = Out(odir + "/" + String(layer) + (".steer.bin" if len(doses) == 1 else ".steer." + String(d) + ".bin"))
        o.put(0x54534e4d, 4)
        o.put(1, 4)
        o.put(layer, 4)
        o.put(ffn, 4)
        o.put(kk, 4)
        o.put(Int(bitcast[DType.uint32](Float32(d))), 4)
        var upi = List[List[Int]]()
        var upv = List[List[Float32]]()
        var dni = List[List[Int]]()
        var dnv = List[List[Float32]]()
        for _ in range(ffn):
            upi.append(List[Int](length=kk, fill=-1))
            upv.append(List[Float32](length=kk, fill=0))
            dni.append(List[Int](length=kk, fill=-1))
            dnv.append(List[Float32](length=kk, fill=0))
        var klv = List[Float32](length=ffn, fill=-1)
        var nullkl = List[Float64]()
        for ui in range(1, len(units)):
            if units[ui].neuron < 0:
                for r in meta:
                    if r.unit == ui and r.dose == d and r.kl >= 0:
                        nullkl.append(r.kl)
                continue
            var n = units[ui].neuron
            for r in meta:
                if r.unit != ui or r.dose != d or len(r.up.id) == 0:
                    continue
                for q in range(min(kk, len(r.up.id))):
                    upi[n][q] = r.up.id[q]
                    upv[n][q] = r.up.v[q]
                    dni[n][q] = r.dn.id[q]
                    dnv[n][q] = r.dn.v[q]
                klv[n] = Float32(r.kl)
        for n in range(ffn):
            for q in range(kk):
                o.put(upi[n][q] & 0xFFFFFFFF, 4)
            for q in range(kk):
                o.put(Int(bitcast[DType.uint32](upv[n][q])), 4)
            for q in range(kk):
                o.put(dni[n][q] & 0xFFFFFFFF, 4)
            for q in range(kk):
                o.put(Int(bitcast[DType.uint32](dnv[n][q])), 4)
        o.close()
        var ko = Out(odir + "/" + String(layer) + (".steer.kl.f32" if len(doses) == 1 else ".steer." + String(d) + ".kl.f32"))
        for n in range(ffn):
            ko.put(Int(bitcast[DType.uint32](klv[n])), 4)
        ko.close()
        sort(nullkl)
        var nj = String("null")
        if len(nullkl) > 0:
            nj = "{\"n\":" + String(len(nullkl)) + ",\"p50\":" + String(nullkl[len(nullkl) // 2]) + ",\"p95\":" + String(nullkl[min(len(nullkl) - 1, len(nullkl) * 95 // 100)]) + ",\"max\":" + String(nullkl[len(nullkl) - 1]) + "}"
        var cj = String()
        for pi in range(len(prompts)):
            cj += ("," if pi > 0 else "") + "\"" + jesc(prompts[pi]) + "\""
        with open(odir + "/" + String(layer) + (".steer.json" if len(doses) == 1 else ".steer." + String(d) + ".json"), "w") as f:
            f.write("{\"dose\":" + String(d) + ",\"contexts\":[" + cj + "],\"null_kl\":" + nj + "}\n")
    print("sweep: wrote steer.bin for", len(doses), "dose(s)")


def main():
    try:
        run_all()
    except e:
        print("FAIL sweep:", e)
        exit(1)
