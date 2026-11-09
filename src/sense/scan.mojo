"""Live brain scan: tail a capture (docs/format-capture.md) and keep, per token, the TOP strongest channels of the FFN output row
(field y, else the last field) of the first captured layer: channel id and f16 value as stored, plus the row norm. `poll` reads only the
records that appeared since the last call, so it follows an engine that is still generating. A channel is a residual direction, not a
neuron: neuron activations would need the FFN weights on top of the captured input (not done)."""
from std.math import sqrt
from json import parse

comptime TOP = 6


struct Scan(Movable):
    var dir: String
    var h: Int
    var nf: Int
    var row: Int
    var layer: Int
    var n: Int
    var ids: List[Int32]
    var top: List[Int32]  # n * TOP channel ids, strongest first
    var val: List[Float32]  # n * TOP stored values
    var norm: List[Float32]

    def __init__(out self, dir: String) raises:
        self.dir = dir
        var m = parse(open(dir + "/meta.json", "r").read())
        self.h = m.get("h").int()
        var fields = m.text("fields")
        self.nf = fields.byte_length()
        self.row = fields.find("y") if fields.find("y") >= 0 else self.nf - 1
        self.layer = m.get("layers").items()[0].int()
        self.n = 0
        self.ids = List[Int32]()
        self.top = List[Int32]()
        self.val = List[Float32]()
        self.norm = List[Float32]()

    def path(self) -> String:
        return self.dir + "/L" + String(self.layer) + ".f16"

    def poll(mut self, limit: Int = 0) raises -> Int:
        """Read the tokens recorded since the last call (at most `limit` in all when > 0); returns how many."""
        var tb = open(self.dir + "/tokens.i32", "r").read_bytes()
        var want = len(tb) // 4
        if limit > 0:
            want = min(want, limit)
        if want <= self.n:
            return 0
        var rec = self.nf * self.h * 2
        var raw = List[UInt8]()
        with open(self.path(), "r") as f:
            _ = f.seek(self.n * rec)
            raw = f.read_bytes((want - self.n) * rec)
        var got = len(raw) // rec  # the layer file may trail the token file by a record
        for t in range(got):
            self.ids.append(UnsafePointer(to=tb[0]).bitcast[Int32]()[self.n + t])
            var v = UnsafePointer(to=raw[0]).bitcast[Float16]() + (t * self.nf + self.row) * self.h
            var used = List[Bool](length=self.h, fill=False)
            var s: Float64 = 0
            for c in range(self.h):
                s += Float64(v[c]) ** 2
            for _ in range(TOP):
                var b = -1
                for c in range(self.h):
                    if not used[c] and (b < 0 or abs(Float32(v[c])) > abs(Float32(v[b]))):
                        b = c
                used[b] = True
                self.top.append(Int32(b))
                self.val.append(Float32(v[b]))
            self.norm.append(Float32(sqrt(s)))
        self.n += got
        return got

    def series(self, c: Int) raises -> List[Float32]:
        """Channel c over every token read so far (one 2-byte read per token)."""
        var out = List[Float32](length=self.n, fill=0)
        var rec = self.nf * self.h * 2
        with open(self.path(), "r") as f:
            for t in range(self.n):
                _ = f.seek(t * rec + (self.row * self.h + c) * 2)
                out[t] = Float32(UnsafePointer(to=f.read_bytes(2)[0]).bitcast[Float16]()[0])
        return out^

    def sig(self) -> String:
        """Checksum of what is on the strip: tokens, sum of |value| and sum of channel ids over all TOP cells."""
        var a: Float64 = 0
        var ch = 0
        for i in range(len(self.val)):
            a += abs(Float64(self.val[i]))
            ch += Int(self.top[i])
        return "n=" + String(self.n) + " sumabs=" + String(Int(a * 1000.0 + 0.5) // 1000) + "." + String(Int(a * 1000.0 + 0.5) % 1000 + 1000)[byte=1:4] + " sumch=" + String(ch)
