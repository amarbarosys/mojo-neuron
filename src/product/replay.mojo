"""Brain-scan replay (F10): `nx replay record {cap, req, id}` copies request `req` of a capture dir (docs/format-capture.md) to
$REPLAY_DIR/<id> (default <main checkout>/.work/replay), same layout, positions from 0, req.i32 all 0; `nx replay frame {id, k, layer?}` returns
record k: token and the f16 rows widened to f32 (exact), what the app scrubber draws. `-D REPLAY_MUT=1` serves record k+1 (the gate must catch it)."""
from std.os import getenv
from std.memory import bitcast
from std.sys import get_defined_int
from json import Json, parse
from mneuron.rlog import Rlog, sh
from mneuron.store import read_text
from product.wiki import sq

comptime MUT = get_defined_int["REPLAY_MUT", 0]()


def rdir() -> String:
    var e = String(getenv("REPLAY_DIR"))
    return e if e.byte_length() > 0 else String(getenv("HOME")) + "/Projects/mojo/mojo-neuron/.work/replay"


def i32s(path: String) raises -> List[Int]:
    var out = List[Int]()
    with open(path, "r") as f:
        var b = f.read_bytes()
        for i in range(len(b) // 4):
            var v = Int(b[4 * i]) | (Int(b[4 * i + 1]) << 8) | (Int(b[4 * i + 2]) << 16) | (Int(b[4 * i + 3]) << 24)
            out.append(v - (1 << 32) if v >= (1 << 31) else v)
    return out^


def record(a: Json) raises -> Json:
    var cap = a.text("cap")
    var id = a.text("id")
    var meta = parse(read_text(cap + "/meta.json"))
    var rec = meta.text("fields").byte_length() * meta.get("h").int() * 2
    var req = i32s(cap + "/req.i32")
    var idx = List[Int]()
    for i in range(len(req)):
        if req[i] == a.get("req").int():
            idx.append(i)
    if len(idx) == 0:
        raise Error("request " + String(a.get("req").int()) + " not in " + cap)
    var d = rdir() + "/" + id
    _ = sh("rm -rf " + sq(d) + "; mkdir -p " + sq(d))
    var names = List[String]()
    for l in meta.get("layers").items():
        names.append("L" + String(l.int()) + ".f16")
    names.append("tokens.i32")
    for nm in names:
        var size = 4 if nm == "tokens.i32" else rec
        var out = List[UInt8]()
        with open(cap + "/" + nm, "r") as f:
            for i in idx:
                f.seek(i * size)
                out.extend(f.read_bytes(size))
        with open(d + "/" + nm, "w") as w:
            w.write_bytes(out)
    with open(d + "/req.i32", "w") as w:
        w.write_bytes(List[UInt8](length=4 * len(idx), fill=0))
    _ = sh("cp " + sq(cap + "/meta.json") + " " + sq(d + "/meta.json"))
    var o = Json.object()
    o.set("id", Json.of(id))
    o.set("tokens", Json.of(len(idx)))
    var rl = Rlog("replay", a.dumps(), "")
    rl.done("record " + id + " tokens=" + String(len(idx)), "", List[Float64](), 0.0, 1.0)
    return o^


def frame(a: Json) raises -> Json:
    var d = rdir() + "/" + a.text("id")
    var meta = parse(read_text(d + "/meta.json"))
    var h = meta.get("h").int()
    var fields = meta.text("fields")
    var layer = a.get("layer").int() if a.has("layer") else meta.get("layers").at(0).int()
    var toks = i32s(d + "/tokens.i32")
    var k = a.get("k").int()
    if k < 0 or k >= len(toks):
        raise Error("frame " + String(k) + " outside 0.." + String(len(toks)))
    var kk = (k + 1) % len(toks) if MUT == 1 else k
    var rec = fields.byte_length() * h * 2
    var b = List[UInt8]()
    with open(d + "/L" + String(layer) + ".f16", "r") as f:
        f.seek(kk * rec)
        b = f.read_bytes(rec)
    var rows = Json.array()
    for r in range(fields.byte_length()):
        var row = Json.array()
        for i in range(h):
            var u = UInt16(b[(r * h + i) * 2]) | (UInt16(b[(r * h + i) * 2 + 1]) << 8)
            row.push(Json.of(Float64(Float32(bitcast[DType.float16](u)))))
        rows.push(row)
    var o = Json.object()
    o.set("k", Json.of(k))
    o.set("token", Json.of(toks[kk]))
    o.set("layer", Json.of(layer))
    o.set("fields", Json.of(fields))
    o.set("rows", rows)
    var rl = Rlog("replay", a.dumps(), "")
    rl.done("frame " + a.text("id") + " " + String(k), "", List[Float64](), 0.0, 1.0)
    return o^


def replay(a: Json) raises -> Json:
    var op = a.text("op")
    if op == "record":
        return record(a)
    if op == "frame":
        return frame(a)
    raise Error("replay op must be record or frame")
