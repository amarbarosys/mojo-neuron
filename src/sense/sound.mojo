"""Sonification of a capture (docs/format-capture.md): each token is one note, each captured layer one voice.
Voice pitch = 110 Hz * 2^(3 * layer / n_layers) (three octaves over the depth); its timbre = the 6 strongest channels of the layer's FFN output
row (field y, else the last field), channel c adds harmonic 1 + c % 8 with weight |v|/sum, sign = phase; loudness = row norm / loudest row.
Deterministic: fixed loop order, Float64 phases, 16-bit WAV (24 kHz mono) -> same capture, same bytes."""
from std.math import sin, sqrt
from std.os.path import exists
from json import parse
from mneuron.ggio import put_u32, write_file

comptime RATE = 24000
comptime NOTE = 2880  # 0.12 s
comptime RAMP = 240  # 10 ms attack and release against clicks
comptime TOP = 6


def f16_at(b: List[UInt8], i: Int) -> Float64:
    return Float64(UnsafePointer(to=b[0]).bitcast[Float16]()[i])


def sound_wav(dir: String, path: String, start: Int, count: Int) raises -> Tuple[Int, Float64]:
    """Render tokens start..start+count (0: to the end) of capture `dir` to WAV `path`; returns (samples, peak amplitude)."""
    var meta = parse(open(dir + "/meta.json", "r").read())
    var h = meta.get("h").int()
    var nl = meta.get("n_layers").int()
    var fields = meta.text("fields")
    var row = fields.find("y")
    if row < 0:
        row = fields.byte_length() - 1
    var nf = fields.byte_length()
    var layers = List[Int]()
    for l in meta.get("layers").items():
        layers.append(l.int())
    var n = len(open(dir + "/tokens.i32", "r").read_bytes()) // 4
    var last = n if count <= 0 else min(n, start + count)
    var files = List[List[UInt8]]()
    for l in layers:  # only the requested records: a live capture file is gigabytes
        with open(dir + "/L" + String(l) + ".f16", "r") as f:
            _ = f.seek(start * nf * h * 2)
            files.append(f.read_bytes((last - start) * nf * h * 2))
    var norms = List[Float64](length=(last - start) * len(layers), fill=0)
    var mx = Float64(1e-30)
    for t in range(start, last):
        for li in range(len(layers)):
            var s: Float64 = 0
            for c in range(h):
                s += f16_at(files[li], ((t - start) * nf + row) * h + c) ** 2
            norms[(t - start) * len(layers) + li] = sqrt(s)
            mx = max(mx, sqrt(s))
    var pcm = List[UInt8]()
    var peak: Float64 = 0
    var tau = 6.283185307179586
    for t in range(start, last):
        var chan = List[Int]()
        var wt = List[Float64]()
        var tone = List[Float64](length=NOTE, fill=0)
        for li in range(len(layers)):
            var f0 = 110.0 * 2.0 ** (3.0 * Float64(layers[li]) / Float64(max(nl, 1)))
            var base = ((t - start) * nf + row) * h
            var used = List[Bool](length=h, fill=False)
            var sum: Float64 = 0
            var pick = List[Int]()
            for _ in range(TOP):
                var b = -1
                for c in range(h):
                    if not used[c] and (b < 0 or abs(f16_at(files[li], base + c)) > abs(f16_at(files[li], base + b))):
                        b = c
                used[b] = True
                pick.append(b)
                sum += abs(f16_at(files[li], base + b))
            var loud = norms[(t - start) * len(layers) + li] / mx / Float64(len(layers))
            for k in pick:
                var v = f16_at(files[li], base + k)
                var hn = Float64(1 + k % 8)
                var a = loud * abs(v) / max(sum, 1e-30) * (1.0 if v >= 0 else -1.0)
                for i in range(NOTE):
                    tone[i] += a * sin(tau * f0 * hn * Float64(i) / Float64(RATE))
        for i in range(NOTE):
            var e = min(1.0, min(Float64(i), Float64(NOTE - 1 - i)) / Float64(RAMP))
            var y = 0.8 * e * tone[i]
            peak = max(peak, abs(y))
            var q = Int(y * 32767.0 + (0.5 if y >= 0 else -0.5))
            pcm.append(UInt8(q & 255))
            pcm.append(UInt8((q >> 8) & 255))
    var wav = List[UInt8]()
    for c in "RIFF".as_bytes():
        wav.append(c)
    put_u32(wav, 36 + len(pcm))
    for c in "WAVEfmt ".as_bytes():
        wav.append(c)
    put_u32(wav, 16)
    wav.append(1)
    wav.append(0)
    wav.append(1)
    wav.append(0)
    put_u32(wav, RATE)
    put_u32(wav, RATE * 2)
    wav.append(2)
    wav.append(0)
    wav.append(16)
    wav.append(0)
    for c in "data".as_bytes():
        wav.append(c)
    put_u32(wav, len(pcm))
    var nb = len(pcm)
    wav.extend(pcm^)
    write_file(path, wav)
    return (nb // 2, peak)
