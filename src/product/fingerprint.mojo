"""Sonification fingerprint (F4, docs/prereg-f4.md). `nx fingerprint {op, name, gguf, layer, dir?}`: the committed probe text goes through `tccap`
($NX_TCCAP, CPU capture of one layer) -> C3 `sound_wav` -> a 64-band log spectrum (Goertzel at 100..11000 Hz, log-spaced), centred and unit-length.
enroll stores <dir>/<name>.json (+ <name>.wav to listen to); match fingerprints afresh and returns every enrolled name with its cosine distance,
nearest first, and `unknown` when the nearest distance exceeds 3 x the largest self-distance on record (0 when nothing is on record to re-run)."""
from std.math import cos, log, sqrt
from std.os import getenv
from json import Json, parse
from mneuron.rlog import Rlog, sh
from mneuron.store import read_text
from sense.sound import sound_wav
from product.wiki import sq

comptime BANDS = 64


def fdir(a: Json) -> String:
    var e = a.text("dir", String(getenv("FP_DIR")))
    return e if e.byte_length() > 0 else String(getenv("HOME")) + "/Projects/mojo/mojo-neuron/.work/fingerprints"


def signature(a: Json, d: String, name: String) raises -> List[Float64]:
    var tc = String(getenv("NX_TCCAP"))
    tc = tc if tc.byte_length() > 0 else String(getenv("HOME")) + "/Projects/mojo/mojo-neuron/.work/from-lanes/features/tccap"
    var cap = d + "/" + name + ".cap"
    _ = sh("rm -rf " + sq(cap) + "; mkdir -p " + sq(d) + "; " + sq(tc) + " " + sq(a.text("gguf")) + " " + sq(a.text("probe", "tests/fingerprint-probe.txt")) + " " + String(a.get("layer").int()) + " " + sq(cap) + " --windows 1 --seg 64 >/dev/null")
    _ = sound_wav(cap, d + "/" + name + ".wav", 0, 0)
    var b = open(d + "/" + name + ".wav", "r").read_bytes()
    var n = (len(b) - 44) // 2
    var x = List[Float64](length=n, fill=0)
    for i in range(n):
        var v = Int(b[44 + 2 * i]) | (Int(b[45 + 2 * i]) << 8)
        x[i] = Float64(v - 65536 if v >= 32768 else v)
    var sig = List[Float64]()
    for k in range(BANDS):  # Goertzel at f_k
        var w = 6.283185307179586 * 100.0 * (110.0 ** (Float64(k) / Float64(BANDS - 1))) / 24000.0
        var c = 2.0 * cos(w)
        var s1 = 0.0
        var s2 = 0.0
        for i in range(n):
            var s0 = x[i] + c * s1 - s2
            s2 = s1
            s1 = s0
        sig.append(0.5 * log((s1 * s1 + s2 * s2 - c * s1 * s2) / Float64(n * n) + 1e-12))
    var m = 0.0
    for v in sig:
        m += v / Float64(BANDS)
    var nn = 0.0
    for i in range(BANDS):
        sig[i] -= m
        nn += sig[i] * sig[i]
    for i in range(BANDS):
        sig[i] /= sqrt(nn)
    return sig^


def dist(a: List[Float64], b: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(BANDS):
        s += a[i] * b[i]
    return 1.0 - s


def fingerprint(a: Json) raises -> Json:
    var d = fdir(a)
    var name = a.text("name")
    var rl = Rlog("fingerprint", a.dumps(), a.text("gguf"))
    var sig = signature(a, d, "fresh-" + name if a.text("op") == "match" else name)
    var o = Json.object()
    if a.text("op") == "enroll":
        var j = Json.object()
        var arr = Json.array()
        for v in sig:
            arr.push(Json.of(v))
        j.set("name", Json.of(name))
        j.set("sig", arr)
        with open(d + "/" + name + ".json", "w") as f:
            f.write(j.dumps())
        o.set("enrolled", Json.of(name))
        rl.done("enroll " + name, "", sig, -1.0, 1.0)
        return o^
    var names = List[String]()
    var ds = List[Float64]()
    for f in sh("ls " + sq(d) + " | grep -E '^[^.]+\\.json$' || true").split("\n"):
        if f.byte_length() > 0:
            var e = parse(read_text(d + "/" + String(f)))
            var s2 = List[Float64]()
            for v in e.get("sig").items():
                s2.append(v.num())
            if len(s2) != BANDS:
                continue  # not a signature file
            var dd = dist(sig, s2)
            var i = len(ds)
            while i > 0 and ds[i - 1] > dd:
                i -= 1
            ds.insert(i, dd)
            names.insert(i, e.text("name"))
    if len(ds) == 0:
        raise Error("nothing enrolled in " + d)
    var rows = Json.array()
    for i in range(len(ds)):
        var r = Json.object()
        r.set("name", Json.of(names[i]))
        r.set("dist", Json.of(ds[i]))
        rows.push(r)
    o.set("nearest", Json.of(names[0]))
    o.set("dist", Json.of(ds[0]))
    o.set("second", Json.of(ds[1] if len(ds) > 1 else 2.0))
    o.set("unknown", Json.of(1 if ds[0] > 3.0 * max(a.get("noise").num(), 1e-9) else 0))
    o.set("all", rows)
    rl.done("match " + name + " -> " + names[0], "", List[Float64](), 0.0, 1.0)
    return o^
