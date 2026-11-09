"""The calls: one function per verb, JSON in, JSON out. The CLI, the MCP server and the app all go through `call`."""
from std.os import getenv, makedirs, remove
from std.time import perf_counter_ns
from std.os.path import exists
from std.subprocess import run
from json import Json, parse
from ggufx.model import Gguf
from mneuron.ggio import kv_int, kv_str, llama_ids, abs_path
from mneuron.store import root, mdir, models, model_info, open_model, layers_with_lens, Lens, Steer, tokens, read_text
from mneuron.rankcore import rank_file, Unit
from mneuron.conn import downstream, upstream, Conn
from mneuron.fwd import fires as cpu_fires
from cvec import write_cvec
from mneuron.rlog import Rlog
from sense.sound import sound_wav
from sense.scan import Scan
from sense.image import turn_prompt, jobs_json, render_cmd
from product.radar import radar
from product.persona import persona
from product.wiki import wiki
from product.replay import replay
from probe.calls import concept_call, probe_call  # lane probe
from product.fingerprint import fingerprint
from lens2.lensdiff import lensdiff_call  # lane agent
from lens2.tlens import tlens_call  # lane agent
from agent.cast import cast_call  # lane agent


def need(a: Json, key: String) raises -> Json:
    if not a.has(key):
        raise Error("missing argument '" + key + "'")
    return a.get(key)


def int_arg(a: Json, key: String, default: Int) -> Int:
    if a.has(key):
        return a.get(key).int()
    return default


def causal_call(tool: String, a: Json) raises -> Json:
    """surgery {model, out, layer, neuron, op, k?, vec?, via?} / ablate-scan {model, out, layer, neurons, prompts, k?, brain?, ...} (src/causal/): runs the
    tool binary ($MNEURON_BIN/<tool>, default ./.work/<tool>) and returns its output text."""
    var bin = getenv("MNEURON_BIN")
    var cmd = (bin if bin.byte_length() > 0 else String(".work")) + "/" + tool + " '" + need(a, "model").str() + "' '" + need(a, "out").str() + "'"
    for key in ["layer", "neuron", "neurons", "op", "k", "vec", "via", "prompts", "brain", "baro", "spark", "pack", "runner", "env", "resp"]:
        if a.has(String(key)):
            var v = a.get(String(key))
            cmd += " '" + String(key) + "=" + ((String(v.int()) if v.num() == Float64(v.int()) else String(v.num())) if v.is_num() else v.str()) + "'"
    var o = Json.object()
    var txt = String(String(run(cmd + " 2>&1")).strip())
    o.set("output", Json.of(txt))
    return o^


def tok_f(t: String, z: Float32) -> Json:
    var p = Json.array()
    p.push(Json.of(t))
    p.push(Json.of(Float64(z)))
    return p^


def lens_json(model: String, layer: Int, n: Int, tk: List[String], kind: String = "") raises -> Json:
    var l = Lens(model, layer, kind)
    if n < 0 or n >= l.ffn:
        raise Error("neuron " + String(n) + " out of range 0.." + String(l.ffn - 1))
    var top = Json.array()
    var bot = Json.array()
    for i in range(l.k):
        if l.id(n, False, i) >= 0:
            top.push(tok_f(tk[l.id(n, False, i)], l.z(n, False, i)))
        if l.id(n, True, i) >= 0:
            bot.push(tok_f(tk[l.id(n, True, i)], l.z(n, True, i)))
    var o = Json.object()
    o.set("top", top)
    o.set("bottom", bot)
    return o^


def conn_json(cs: List[Conn]) -> Json:
    var a = Json.array()
    for c in cs:
        var o = Json.object()
        o.set("layer", Json.of(c.layer))
        o.set("neuron", Json.of(c.neuron))
        o.set("role", Json.of(c.role))
        o.set("cos", Json.of(c.cos))
        a.push(o)
    return a^


def rank_json(model: String, layer: Int, k: Int, only: Int) raises -> Json:
    """Top k units of the sweep, or (only >= 0) the record of one neuron with the control statistics."""
    var p = mdir(model) + "/" + String(layer) + ".sweep.jsonl"
    if not exists(p):
        raise Error("no sweep for " + model + " layer " + String(layer) + " (lane engine writes " + p + ")")
    var us = rank_file(p)
    var cm: Float64 = 0
    var cs: Float64 = 0
    var nc = 0
    for u in us:
        if u.name.startswith("ctrl"):
            cm += u.mass
            nc += 1
    if nc > 0:
        cm /= Float64(nc)
        for u in us:
            if u.name.startswith("ctrl"):
                cs += (u.mass - cm) * (u.mass - cm)
        cs = (cs / Float64(nc)) ** 0.5
    var out = Json.array()
    for u in us:
        if u.name.startswith("ctrl") or u.name == "base":
            continue
        if only >= 0 and u.name != String(only):
            continue
        var o = Json.object()
        o.set("neuron", Json.of(Int(u.name)))
        o.set("score", Json.of(u.mass))
        o.set("pct", Json.of(u.pct))
        o.set("control_mean", Json.of(cm))
        o.set("control_std", Json.of(cs))
        o.set("words", Json.of(u.top))
        out.push(o)
        if only < 0 and len(out.items()) >= k:
            break
    return out^


def tokenize(model: String, text: String) raises -> List[Int]:
    return llama_ids(model_info(model).text("gguf"), text, root() + "/.fires-text.txt")


def fires_call(a: Json) raises -> Json:
    var model = need(a, "model").str()
    var layer = need(a, "layer").int()
    var text = need(a, "text").str()
    var k = int_arg(a, "k", 20)
    var g = open_model(model)
    var ids = tokenize(model, text)
    var tk = tokens(model)
    var acts = cpu_fires(g, ids, [layer])
    var T = len(ids)
    var ffn = len(acts[0]) // T
    var toks = Json.array()
    for t in range(T):
        toks.push(Json.of(tk[ids[t]]))
    var o = Json.object()
    o.set("tokens", toks)
    if a.has("neuron"):
        var n = a.get("neuron").int()
        var v = Json.array()
        for t in range(T):
            v.push(Json.of(Float64(acts[0][t * ffn + n])))
        o.set("activations", v)
        return o^
    var best = List[Float32]()  # per neuron max over tokens
    var at = List[Int]()
    for n in range(ffn):
        var m = Float32.MIN
        var tt = 0
        for t in range(T):
            if acts[0][t * ffn + n] > m:
                m = acts[0][t * ffn + n]
                tt = t
        best.append(m)
        at.append(tt)
    var picked = List[Int]()
    for _ in range(min(k, ffn)):
        var bi = -1
        for n in range(ffn):
            var used = False
            for q in picked:
                if q == n:
                    used = True
            if not used and (bi < 0 or best[n] > best[bi]):
                bi = n
        picked.append(bi)
    var top = Json.array()
    for n in picked:
        var e = Json.object()
        e.set("neuron", Json.of(n))
        e.set("act", Json.of(Float64(best[n])))
        e.set("token_index", Json.of(at[n]))
        e.set("token", Json.of(tk[ids[at[n]]]))
        top.push(e)
    o.set("top", top)
    return o^


def search_lens(a: Json) raises -> Json:
    var model = need(a, "model").str()
    var layer = need(a, "layer").int()
    var q = String(need(a, "token").str().strip()).lower()
    var k = int_arg(a, "k", 50)
    var l = Lens(model, layer)
    var tk = tokens(model)
    var out = Json.array()
    for n in range(l.ffn):
        if len(out.items()) >= k:  # first k hits in neuron order
            break
        for bottom in [False, True]:
            for i in range(l.k):
                if len(out.items()) >= k:
                    break
                var id = l.id(n, bottom, i)
                if id >= 0 and String(tk[id].strip()).lower() == q:
                    var e = Json.object()
                    e.set("neuron", Json.of(n))
                    e.set("pole", Json.of(String("bottom") if bottom else String("top")))
                    e.set("rank", Json.of(i))
                    e.set("z", Json.of(Float64(l.z(n, bottom, i))))
                    out.push(e)
    return out^


def parse_recipe(r: String, n_layer: Int, ffn: Int) raises -> Json:
    """`L:N:dose,L:N:dose` (docs/format.md). Layers 1..n_layer-1, neuron < ffn."""
    var units = Json.array()
    var norm = String()
    for part in r.split(","):
        var f = String(String(part).strip()).split(":")
        if len(f) != 3:
            raise Error("bad recipe unit '" + String(part) + "': want L:N:dose")
        var L = Int(String(f[0]))
        var n = Int(String(f[1]))
        var ds = String(f[2])
        var d = Float64(ds[byte=1:]) if ds.startswith("+") else Float64(ds)
        if L < 1 or L >= n_layer:
            raise Error("layer " + String(L) + " outside 1.." + String(n_layer - 1) + " (llama.cpp cannot steer layer 0)")
        if n < 0 or n >= ffn:
            raise Error("neuron " + String(n) + " outside 0.." + String(ffn - 1))
        var u = Json.object()
        u.set("layer", Json.of(L))
        u.set("neuron", Json.of(n))
        u.set("dose", Json.of(d))
        units.push(u)
        norm += ("," if norm.byte_length() > 0 else "") + String(L) + ":" + String(n) + ":" + ("+" if d >= 0 else "") + String(d)
    var o = Json.object()
    o.set("units", units)
    o.set("recipe", Json.of(norm))
    return o^


def dossier(a: Json) raises -> Json:
    var model = need(a, "model").str()
    var layer = need(a, "layer").int()
    var n = need(a, "neuron").int()
    var g = open_model(model)
    var tk = tokens(model)
    var span = int_arg(a, "span", 3)
    var o = Json.object()
    o.set("v", Json.of(1))
    o.set("model", Json.of(model))
    o.set("layer", Json.of(layer))
    o.set("neuron", Json.of(n))
    o.set("lens", lens_json(model, layer, n, tk))
    if exists(mdir(model) + "/" + String(layer) + ".rlens.bin"):
        o.set("r_lens", lens_json(model, layer, n, tk, "r"))
    var cn = Json.object()
    cn.set("up", conn_json(upstream(g, layer, n, span, 12)))
    cn.set("down", conn_json(downstream(g, layer, n, span, 12)))
    o.set("connections", cn)
    var sweep = mdir(model) + "/" + String(layer) + ".sweep.jsonl"
    if exists(sweep):
        var r = rank_json(model, layer, 0, n)
        if len(r.items()) > 0:
            o.set("rank", r.at(0))
        var rolls = Json.array()
        var bases = Json.array()  # the engine's unsteered rows ("neuron":"base", dose 0, one per prompt), kept so the chat can show the baseline
        for line in read_text(sweep).split("\n"):
            if line.byte_length() == 0:
                continue
            var d = parse(String(line))
            if d.get("neuron").kind() == 4 and d.get("neuron").str() == "base" and len(bases.items()) < 64:
                var b = Json.object()
                b.set("prompt", d.get("prompt"))
                b.set("text", d.get("text"))
                bases.push(b)
            elif d.get("neuron").kind() != 4 and d.get("neuron").int() == n and len(rolls.items()) < 12:
                var e = Json.object()
                e.set("dose", d.get("dose"))
                e.set("prompt", d.get("prompt"))
                e.set("text", d.get("text"))
                rolls.push(e)
        var st = Json.object()
        st.set("rollouts", rolls)
        st.set("base", bases)
        o.set("steer", st)
    var spath = mdir(model) + "/" + String(layer) + ".steer.bin"
    if exists(spath):  # the engine's steer lens: next-token log-prob change at +dose
        var sl = Steer(spath)
        if n < sl.ffn:
            var up = Json.array()
            var dn = Json.array()
            for i in range(sl.k):
                if sl.id(n, False, i) >= 0:
                    up.push(tok_f(tk[sl.id(n, False, i)], sl.dlp(n, False, i)))
                if sl.id(n, True, i) >= 0:
                    dn.push(tok_f(tk[sl.id(n, True, i)], sl.dlp(n, True, i)))
            var e = Json.object()
            e.set("dose", Json.of(Float64(sl.dose)))
            e.set("up", up)
            e.set("down", dn)
            var arr = Json.array()
            arr.push(e)
            if not o.has("steer"):
                o.set("steer", Json.object())
            var st = o.get("steer")
            st.set("slens", arr)
            if not st.has("rollouts"):
                st.set("rollouts", Json.array())
            o.set("steer", st)
    var fi = Json.array()
    if a.has("text"):
        var fa = Json.object()
        fa.set("model", Json.of(model))
        fa.set("layer", Json.of(layer))
        fa.set("neuron", Json.of(n))
        fa.set("text", a.get("text"))
        var f = fires_call(fa)
        var e = Json.object()
        e.set("text", a.get("text"))
        e.set("tokens", f.get("tokens"))
        e.set("activations", f.get("activations"))
        fi.push(e)
    o.set("fires", fi)
    if int_arg(a, "save", 1) == 1:
        var dir = mdir(model) + "/" + String(layer) + ".dossier"
        makedirs(dir, exist_ok=True)
        with open(dir + "/" + String(n) + ".json", "w") as f:
            f.write(o.dumps(False, 2))
    return o^


def brain_note(a: Json) raises -> Json:
    """Export an explored neuron as a note of the Brain network: (private notes)"""
    var d = dossier(a)
    var model = d.text("model")
    var layer = d.get("layer").int()
    var n = d.get("neuron").int()
    var name = "L" + String(layer) + "-N" + String(n)
    var broot = getenv("MNEURON_BRAIN")
    if broot.byte_length() == 0:
        broot = String(getenv("HOME")) + "/Brain/mojo-neuron/neurons"
    var dir = broot + "/" + model
    makedirs(dir, exist_ok=True)
    var s = "# mojo-neuron neuron " + model + " layer " + String(layer) + " neuron " + String(n) + "\n\n"
    var top = String()
    for t in d.get("lens").get("top").items():
        top += " `" + t.at(0).str() + "`"
    var bot = String()
    for t in d.get("lens").get("bottom").items():
        bot += " `" + t.at(0).str() + "`"
    s += "MLP neuron " + String(n) + " of layer " + String(layer) + " in model " + model + ". Its output direction promotes the tokens" + top + " and suppresses" + bot + " (logit lens).\n"
    if d.has("rank"):
        s += "\nSweep: cluster mass " + String(d.get("rank").get("score").num()) + " against control mean " + String(d.get("rank").get("control_mean").num()) + "; distinctive words when steered: " + d.get("rank").text("words") + ".\n"
    s += "\n## Connections\n\n"
    var links = String()
    for kind in ["down", "up"]:
        for c in d.get("connections").get(String(kind)).items():
            var tgt = "L" + String(c.get("layer").int()) + "-N" + String(c.get("neuron").int())
            s += "- " + String(kind) + "stream: [[mojo-neuron/neurons/" + model + "/" + tgt + "]] " + c.text("role") + " cos " + String(c.get("cos").num()) + "\n"
    s += "\n## Related\n- [[mojo-neuron/2026-11-04-start]]\n"
    with open(dir + "/" + name + ".md", "w") as f:
        f.write(s)
    if int_arg(a, "sync", 0) == 1:  # index it now (rebuild-if-stale, a few seconds); brain-ask in bm25 mode finds it at once
        var cmd = getenv("MNEURON_SYNC")  # test hook: replaces the sync command
        if cmd.byte_length() == 0:
            cmd = String("~/iTools/brain/coios-brain-sync/coios-brain-sync.sh")
        _ = run(cmd + " >/dev/null 2>&1; echo ok")
    var o = Json.object()
    o.set("path", Json.of(dir + "/" + name + ".md"))
    return o^


def sq(s: String) -> String:
    """Shell single-quote."""
    return "'" + s.replace("'", "'\\''") + "'"


def live(a: Json) raises -> Json:
    """Live steered chat: one resident engine sweep (the engine lane's `sweep`, one spark process) of one neuron at -dose and +dose
    plus the unsteered row, on one prompt. Needs <store>/<model>/live.json with the engine paths; the run goes through gpu-wait and is
    logged in the GPU ledger. Rows land in <store>/<model>/live/<layer>.sweep.jsonl (docs/format.md), never in the ranked sweep file."""
    var model = need(a, "model").str()
    var layer = need(a, "layer").int()
    var neuron = need(a, "neuron").int()
    var prompt = String(need(a, "prompt").str().replace("\n", " ").strip())
    if prompt.byte_length() == 0:
        raise Error("live: empty prompt")
    var dose = a.get("dose").num() if a.has("dose") else 0.5
    if dose <= 0:
        raise Error("live: dose must be positive (the chat runs -dose and +dose)")
    var n = int_arg(a, "n", 24)
    var cfgp = mdir(model) + "/live.json"
    if not exists(cfgp):
        raise Error("live: no " + cfgp + ". Write {\"sweep\":BIN,\"gguf\":FILE,\"spark\":BIN,\"pack\":DIR,\"baro\":DIR,\"rho\":F32FILE(optional)} for the engine lane's sweep (its scripts/sweep-check.sh shows the values)")
    var c = parse(read_text(cfgp))
    for k in ["sweep", "gguf", "spark", "pack", "baro"]:
        if c.text(String(k)).byte_length() == 0:
            raise Error("live: " + cfgp + " lacks '" + String(k) + "'")
    var dir = abs_path(mdir(model)) + "/live"  # the sweep runs under gpu-wait, whose cwd may differ
    makedirs(dir, exist_ok=True)
    var out = dir + "/" + String(layer) + ".sweep.jsonl"
    if exists(out):
        remove(out)
    with open(dir + "/prompt.txt", "w") as f:
        f.write(prompt + "\n")
    var runner = c.text("runner", "gpu-wait run --label mneuron-live --timeout 300 --")
    var cmd = sq(c.text("sweep")) + " " + sq(c.text("gguf")) + " " + sq(dir) + " layer=" + String(layer) + " neurons=" + String(neuron)
    cmd += " doses=" + sq("-" + String(dose) + "," + String(dose)) + " prompts=" + sq(dir + "/prompt.txt") + " ctrl=0 n=" + String(n)
    cmd += " spark=" + sq(c.text("spark")) + " pack=" + sq(c.text("pack")) + " baro=" + sq(c.text("baro")) + " " + sq("runner=" + runner)
    if c.text("rho").byte_length() > 0:
        cmd += " rho=" + sq(c.text("rho"))
    var t0 = perf_counter_ns()
    var log = String()
    try:
        log = run(cmd + " 2>&1")
    except e:
        log = String(e)
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    var ledger = getenv("MNEURON_LEDGER")
    if ledger.byte_length() == 0:
        ledger = root() + "/../gpu-ledger.md"
    if secs > 1.0:  # a launch that died at once (bad path) used no GPU
        var cents = Int(secs * 100.0 / 60.0 + 0.5)
        with open(ledger, "a") as f:
            f.write("app | live chat sweep " + model + " L" + String(layer) + " N" + String(neuron) + " (1 prompt, 2 doses + base, n=" + String(n) + ") | " + String(cents // 100) + "." + String(cents % 100 + 100)[byte=1:3] + "\n")
    if not exists(out):
        var tail = log if log.byte_length() < 600 else String(log[byte = log.byte_length() - 600 : log.byte_length()])
        raise Error("live: the engine sweep wrote no rows: " + tail)
    var o = Json.object()
    o.set("prompt", Json.of(prompt))
    var neg = String()
    var base = String()
    var pos = String()
    for line in read_text(out).split("\n"):
        if line.byte_length() == 0:
            continue
        var d = parse(String(line))
        if d.get("neuron").kind() == 4:
            if d.get("neuron").str() == "base":
                base = d.text("text")
        elif d.get("neuron").int() == neuron:
            if d.get("dose").num() < 0:
                neg = d.text("text")
            elif d.get("dose").num() > 0:
                pos = d.text("text")
    o.set("neg", Json.of(neg))
    o.set("base", Json.of(base))
    o.set("pos", Json.of(pos))
    o.set("dose", Json.of(dose))
    o.set("wall_s", Json.of(secs))
    return o^


def call(name: String, a: Json) raises -> Json:
    if name == "models":
        var out = Json.array()
        for m in models():
            var info = model_info(m)
            var e = Json.object()
            e.set("model", Json.of(m))
            e.set("arch", info.get("arch"))
            e.set("n_layer", info.get("n_layer"))
            e.set("ffn", info.get("ffn"))
            var ls = Json.array()
            for l in layers_with_lens(m):
                ls.push(Json.of(l))
            e.set("lens_layers", ls)
            out.push(e)
        return out^
    if name == "lens":
        var model = need(a, "model").str()
        var kind = String("r") if a.has("kind") and a.get("kind").str() == "r" else String("")
        return lens_json(model, need(a, "layer").int(), need(a, "neuron").int(), tokens(model), kind)
    if name == "search_lens":
        return search_lens(a)
    if name == "rank":
        return rank_json(need(a, "model").str(), need(a, "layer").int(), int_arg(a, "k", 50), -1)
    if name == "connections":
        var g = open_model(need(a, "model").str())
        var L = need(a, "layer").int()
        var n = need(a, "neuron").int()
        var o = Json.object()
        o.set("up", conn_json(upstream(g, L, n, int_arg(a, "span", 3), int_arg(a, "k", 12))))
        o.set("down", conn_json(downstream(g, L, n, int_arg(a, "span", 3), int_arg(a, "k", 12))))
        return o^
    if name == "fires":
        return fires_call(a)
    if name == "dossier":
        return dossier(a)
    if name == "recipe":
        var info = model_info(need(a, "model").str())
        return parse_recipe(need(a, "recipe").str(), info.get("n_layer").int(), info.get("ffn").int())
    if name == "cvec":  # a recipe as a llama.cpp control vector (+ the engine's f32 steer file): rel doses when model.json has rho
        var model = need(a, "model").str()
        var info = model_info(model)
        var rho = List[Float32]()
        if info.has("rho") and len(info.get("rho").items()) == info.get("n_layer").int():
            for r in info.get("rho").items():
                rho.append(Float32(r.num()))
        var out = a.text("out") if a.has("out") else mdir(model) + "/dial"
        var parsed = parse_recipe(need(a, "recipe").str(), info.get("n_layer").int(), info.get("ffn").int())
        var rl = Rlog("cvec", parsed.text("recipe"), info.text("gguf"))
        var made = write_cvec(info.text("gguf"), parsed.text("recipe"), out, rho)
        rl.done("layers=" + String(made[0]) + " max_abs=" + String(made[1]) + " mode=" + ("rel" if len(rho) > 0 else "raw"), out + ".cvec.gguf", [Float64(made[0]), made[1]], 0.0, 1e6)
        var o = Json.object()
        o.set("layers", Json.of(made[0]))
        o.set("recipe", parsed.get("recipe"))
        o.set("mode", Json.of(String("rel") if len(rho) > 0 else String("raw")))
        o.set("gguf", Json.of(out + ".cvec.gguf"))
        o.set("f32", Json.of(out + ".f32"))
        return o^
    if name == "sound":  # a capture as a WAV: one note per token, one voice per layer (src/sense/sound.mojo)
        var cap = need(a, "cap").str()
        var wav = need(a, "out").str()
        var rl = Rlog("sound", cap, "")
        var r = sound_wav(cap, wav, int_arg(a, "start", 0), int_arg(a, "tokens", 0))
        rl.done("samples=" + String(r[0]) + " peak=" + String(r[1]), wav, [Float64(r[0]), r[1]], 0.0, 1e9)
        var o = Json.object()
        o.set("samples", Json.of(r[0]))
        o.set("seconds", Json.of(Float64(r[0]) / 24000.0))
        o.set("peak", Json.of(r[1]))
        o.set("wav", Json.of(wav))
        return o^
    if name == "scan":  # the live scan of a capture, one shot: the strip checksum and the strongest cells of the last token
        var cap = need(a, "cap").str()
        var sc = Scan(cap)
        var rl = Rlog("scan", cap, "")
        _ = sc.poll(int_arg(a, "tokens", 0))
        var nm = List[Float64]()
        for x in sc.norm:
            nm.append(Float64(x))
        rl.done(sc.sig(), cap + "/L" + String(sc.layer) + ".f16", nm, 0.0, 1e9)
        var o = Json.object()
        o.set("tokens", Json.of(sc.n))
        o.set("layer", Json.of(sc.layer))
        o.set("sig", Json.of(sc.sig()))
        return o^
    if name == "image":  # one picture per turn: {"turns":[{"text":..,"words":[..]}],"out":DIR,"dry":1?}
        var turns = need(a, "turns").items()
        var outd = need(a, "out").str()
        var prompts = List[String]()
        var pj = Json.array()
        for t in turns:
            var ws = List[String]()
            for w in t.get("words").items():
                ws.append(w.str())
            prompts.append(turn_prompt(t.text("text"), ws))
            pj.push(Json.of(prompts[len(prompts) - 1]))
        var o = Json.object()
        o.set("prompts", pj)
        if a.has("dry"):
            return o^
        makedirs(outd, exist_ok=True)
        var jf = outd + "/jobs.json"
        with open(jf, "w") as f:
            f.write(jobs_json(prompts))
        var rl = Rlog("image", String(len(prompts)) + " turns", "")
        var t0 = perf_counter_ns()
        var runner = a.text("runner", "gpu-wait run --label mneuron-image --priority 20 --vram 21 --timeout 900 --")
        var log = String()
        try:
            log = run(render_cmd(jf, outd, runner) + " 2>&1")
        except e:
            log = String(e)
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        var files = Json.array()
        for i in range(len(prompts)):
            var png = outd + "/turn-" + String(i + 1) + ".png"
            if exists(png):
                files.push(Json.of(png))
        rl.done("turns=" + String(len(prompts)) + " images=" + String(len(files.items())) + " wall_s=" + String(secs), outd, [Float64(len(files.items())), secs], Float64(len(prompts)), 1e5)
        if len(files.items()) != len(prompts):
            var tail = log if log.byte_length() < 600 else String(log[byte = log.byte_length() - 600 : log.byte_length()])
            raise Error("image: " + String(len(files.items())) + " of " + String(len(prompts)) + " pictures written: " + tail)
        o.set("images", files)
        o.set("wall_s", Json.of(secs))
        return o^
    if name == "export_brain":
        return brain_note(a)
    if name == "live":
        return live(a)
    if name == "radar":  # lane product
        return radar(a)
    if name == "persona":  # lane product
        return persona(a)
    if name == "wiki":  # lane product
        return wiki(a)
    if name == "replay":  # lane product
        return replay(a)
    # ## lane probe
    if name == "concept":
        return concept_call(a)
    if name == "probe":
        return probe_call(a)
    if name == "fingerprint":  # lane product
        return fingerprint(a)
    # ## lane agent
    if name == "lens-diff":
        return lensdiff_call(a)
    if name == "tlens":
        return tlens_call(a)
    if name == "cast":
        return cast_call(a)
    if name == "surgery":
        return causal_call(name, a)
    if name == "ablate-scan":  # the ablation scan tool; "scan" is the live capture scan above
        return causal_call("scan", a)
    raise Error("unknown call '" + name + "'. Calls: models lens search_lens rank connections fires dossier recipe cvec sound scan image export_brain live radar persona wiki replay concept probe fingerprint lens-diff tlens cast surgery ablate-scan")
