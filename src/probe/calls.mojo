"""Lane probe calls for `nx` (registered in src/mneuron/calls.mojo, section `lane probe`): concept, probe."""
from std.os import makedirs
from std.os.path import exists
from std.subprocess import run
from json import Json, parse
from ggufx.model import Gguf
from mneuron.ggio import tensor, put_f32, write_atomic
from mneuron.rlog import Rlog
from mneuron.store import root, model_info, read_text
from mneuron.ggio import vocab
from cvec import model_layers
from probe.concept import parse_expr, build, lens_of
from probe.lie import i32s, cap_rows, load_probe


def concept_call(a: Json) raises -> Json:
    """{gguf|model, expr, scale?=1, layer?=0, k?=10, out?}: writes <out>/concept.f32 (BARO_STEER layout) and returns the lens of every touched layer."""
    var path = a.text("gguf")
    if path.byte_length() == 0:
        if not a.has("model"):
            raise Error("concept: give 'gguf' (a path) or 'model' (a store name)")
        path = model_info(a.get("model").str()).text("gguf")
    if not a.has("expr"):
        raise Error("concept: missing argument 'expr'")
    var expr = a.get("expr").str()
    var scale = a.get("scale").num() if a.has("scale") else 1.0
    var k = a.get("k").int() if a.has("k") else 10
    var g = Gguf(path)
    var nl = model_layers(g)  # residual layers: the MTP head's block is not one (a 33-layer file made the engine refuse the steer)
    var h = g.tensors[tensor(g, "output_norm.weight")].dims[0]
    var rl = Rlog("concept", expr, path)
    var vec = build(g, parse_expr(expr, a.get("layer").int() if a.has("layer") else 0), scale, h, nl)
    var out = a.text("out")
    if out.byte_length() == 0:
        out = root() + "/../concept"
    makedirs(out, exist_ok=True)
    var raw = List[UInt8]()
    for i in range(nl * h):
        put_f32(raw, vec[i])
    write_atomic(out + "/concept.f32", raw)
    var o = Json.object()
    o.set("vector", Json.of(out + "/concept.f32"))
    o.set("n_layer", Json.of(nl))
    o.set("hidden", Json.of(h))
    var layers = Json.array()
    var norms = List[Float64]()
    for L in range(nl):
        var row = List[Float32](length=h, fill=0)
        var s: Float64 = 0
        for j in range(h):
            row[j] = vec[L * h + j]
            s += Float64(row[j]) ** 2
        if s == 0:
            continue
        var e = Json.object()
        e.set("layer", Json.of(L))
        e.set("norm", Json.of(s ** 0.5))
        e.set("lens", lens_of(g, row, h, k))
        layers.push(e)
        norms.append(s ** 0.5)
    o.set("layers", layers)
    rl.done("concept " + expr + ": " + String(len(norms)) + " layers", out + "/concept.f32", norms, 0.0, 1e6)
    return o^


def probe_call(a: Json) raises -> Json:
    """{cap?|text?, dir?, layer?, req?, gguf?}: the live lie score per token (logit of P(true), docs/prereg-f24.md) of one request of a capture.
    `text` captures it first through the engine (scripts/probe-live.sh, under gpu-wait). dir = a lie_fit output dir (default <store>/../probe/lie);
    layer defaults to its best_layer; req is 1-based among the capture's requests (default the last); gguf adds token strings."""
    var dir = a.text("dir")
    if dir.byte_length() == 0:
        dir = root() + "/../probe/lie"
    if not exists(dir + "/fit.json"):
        raise Error("probe: no " + dir + "/fit.json (run scripts/probe-run.sh first)")
    var layer = a.get("layer").int() if a.has("layer") else parse(read_text(dir + "/fit.json")).get("best_layer").int()
    var cap = a.text("cap")
    if a.has("text"):
        var txt = a.get("text").str().replace("\n", " ")
        cap = root() + "/../probe/live-cap"
        var out = String(run("bash scripts/probe-live.sh " + sq(txt) + " " + sq(cap) + " 2>&1 | tail -1"))
        if not exists(cap + "/req.i32"):
            raise Error("probe: live capture failed: " + out)
    if cap.byte_length() == 0:
        raise Error("probe: give 'cap' (a capture dir) or 'text'")
    var rl = Rlog("probe", cap + " L" + String(layer), a.text("gguf"))
    var req = i32s(cap + "/req.i32")
    var toks = i32s(cap + "/tokens.i32")
    var ids = List[Int]()
    var want = a.get("req").int() if a.has("req") else -1
    var cur = 0
    var seen = 0
    for i in range(len(req)):
        if i == 0 or req[i] != req[i - 1]:
            seen += 1
        if (want < 0 and req[i] == req[len(req) - 1]) or seen == want:
            ids.append(i)
            cur = req[i]
    if len(ids) == 0:
        raise Error("probe: request " + String(want) + " not in the capture")
    var meta = parse(read_text(cap + "/meta.json"))
    var h = meta.get("h").int()
    var p = load_probe(dir + "/L" + String(layer) + ".probe")
    if len(p.w) != h:
        raise Error("probe: probe h " + String(len(p.w)) + " != capture h " + String(h))
    var rows = cap_rows(cap, layer, ids, h)
    var tj = Json.array()
    var sj = Json.array()
    var vals = List[Float64]()
    var names = List[String]()
    if a.text("gguf").byte_length() > 0:
        names = vocab(Gguf(a.text("gguf")))
    var sum: Float64 = 0
    for k in range(len(ids)):
        var sc = Float64(p.score(rows, k, h))
        tj.push(Json.of(names[toks[ids[k]]] if len(names) > 0 else String(toks[ids[k]])))
        sj.push(Json.of(sc))
        vals.append(sc)
        sum += sc
    var o = Json.object()
    o.set("layer", Json.of(layer))
    o.set("request", Json.of(cur))
    o.set("tokens", tj)
    o.set("scores", sj)
    o.set("last", Json.of(vals[len(vals) - 1]))
    o.set("mean", Json.of(sum / Float64(len(vals))))
    rl.done("probe L" + String(layer) + " last " + String(vals[len(vals) - 1]), cap, vals, -1e4, 1e4)
    return o^


def sq(s: String) -> String:
    return "'" + s.replace("'", "'\\''") + "'"
