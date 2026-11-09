"""The shared neuron store (docs/format.md): paths, model.json, lens.bin reader, token text."""
from std.os import listdir, getenv, makedirs
from std.os.path import exists
from std.memory import bitcast
from json import Json, parse
from ggufx.model import Gguf
from mneuron.ggio import vocab


def root() -> String:
    var e = getenv("MNEURON_DIR")
    if e.byte_length() > 0:
        return e
    return String(getenv("HOME")) + "/Projects/mojo/mojo-neuron/.work/neurons"


def mdir(model: String) -> String:
    return root() + "/" + model


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def model_info(model: String) raises -> Json:
    var p = mdir(model) + "/model.json"
    if not exists(p):
        raise Error("unknown model " + model + ": no " + p + " (run lens --out first)")
    return parse(read_text(p))


def open_model(model: String) raises -> Gguf:
    return Gguf(model_info(model).text("gguf"))


def models() raises -> List[String]:
    var out = List[String]()
    if not exists(root()):
        return out^
    for d in listdir(root()):
        if exists(root() + "/" + d + "/model.json"):
            out.append(String(d))
    return out^


def layers_with_lens(model: String) raises -> List[Int]:
    var out = List[Int]()
    for f in listdir(mdir(model)):
        if f.endswith(".lens.bin"):
            out.append(Int(String(f[byte=0 : f.byte_length() - 9])))
    for i in range(len(out)):  # insertion sort
        var j = i
        while j > 0 and out[j - 1] > out[j]:
            out.swap_elements(j - 1, j)
            j -= 1
    return out^


def u32(b: List[UInt8], at: Int) -> Int:
    return Int(b[at]) | (Int(b[at + 1]) << 8) | (Int(b[at + 2]) << 16) | (Int(b[at + 3]) << 24)


def f32(b: List[UInt8], at: Int) -> Float32:
    return bitcast[DType.float32](UInt32(u32(b, at)))


struct Lens(Movable):
    var b: List[UInt8]
    var layer: Int
    var ffn: Int
    var hidden: Int
    var k: Int

    def __init__(out self, model: String, layer: Int, kind: String = "") raises:
        """kind "" is the logit lens (<layer>.lens.bin), "r" the R readout (<layer>.rlens.bin)."""
        var p = mdir(model) + "/" + String(layer) + "." + kind + "lens.bin"
        if not exists(p):
            raise Error("no lens for " + model + " layer " + String(layer) + ": run `lens MODEL " + String(layer) + " 8 --out DIR`")
        with open(p, "r") as f:
            self.b = f.read_bytes()
        if len(self.b) < 24 or self.b[0] != 77 or self.b[1] != 78 or self.b[2] != 76 or self.b[3] != 78:
            raise Error(p + ": not a lens.bin")
        self.layer = u32(self.b, 8)
        self.ffn = u32(self.b, 12)
        self.hidden = u32(self.b, 16)
        self.k = u32(self.b, 20)
        if len(self.b) != 24 + self.ffn * self.k * 16:
            raise Error(p + ": size does not match its header")

    def at(self, n: Int, part: Int, i: Int) -> Int:
        """Byte offset of record n, part 0 top ids, 1 top z, 2 bottom ids, 3 bottom z, element i."""
        return 24 + n * self.k * 16 + part * self.k * 4 + i * 4

    def id(self, n: Int, bottom: Bool, i: Int) -> Int:
        var v = u32(self.b, self.at(n, 2 if bottom else 0, i))
        return v - (1 << 32) if v >= (1 << 31) else v

    def z(self, n: Int, bottom: Bool, i: Int) -> Float32:
        return f32(self.b, self.at(n, 3 if bottom else 1, i))


def tokens(model: String) raises -> List[String]:
    return vocab(open_model(model))


struct Steer(Movable):
    """<layer>.steer.bin (docs/format.md): per neuron the tokens whose log-prob rose most (up) and fell most (down) at +dose."""
    var b: List[UInt8]
    var ffn: Int
    var k: Int
    var dose: Float32

    def __init__(out self, path: String) raises:
        with open(path, "r") as f:
            self.b = f.read_bytes()
        if len(self.b) < 24 or self.b[0] != 77 or self.b[1] != 78 or self.b[2] != 83 or self.b[3] != 84:
            raise Error(path + ": not a steer.bin")
        self.ffn = u32(self.b, 12)
        self.k = u32(self.b, 16)
        self.dose = f32(self.b, 20)
        if len(self.b) != 24 + self.ffn * self.k * 16:
            raise Error(path + ": size does not match its header")

    def id(self, n: Int, down: Bool, i: Int) -> Int:
        var v = u32(self.b, 24 + n * self.k * 16 + (2 if down else 0) * self.k * 4 + i * 4)
        return v - (1 << 32) if v >= (1 << 31) else v

    def dlp(self, n: Int, down: Bool, i: Int) -> Float32:
        return f32(self.b, 24 + n * self.k * 16 + (3 if down else 1) * self.k * 4 + i * 4)
