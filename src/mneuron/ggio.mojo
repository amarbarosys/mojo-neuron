"""GGUF helpers shared by the neuron tools: tensor lookup, row dequant, vocabulary text, metadata, little-endian writers."""
from std.memory import bitcast
from std.os import getenv
from std.os.path import realpath
from std.ffi import external_call
from std.subprocess import run
from ggufx.model import Gguf, pread_into
from ggufx.dequant import quant_block, deq_rows
from ggufx.values import uint_at, peek, read_str


def tensor(g: Gguf, name: String) raises -> Int:
    for k in range(len(g.tensors)):
        if g.tensors[k].name == name:
            return k
    return -1


def load_rows(g: Gguf, ti: Int, row0: Int, n: Int) raises -> List[Float32]:
    var t = g.tensors[ti].ttype
    var ne0 = g.tensors[ti].dims[0]
    var info = quant_block(t)
    if info[0] == 0:
        raise Error(g.tensors[ti].name + ": no dequantizer for ggml type " + String(t))
    var rb = ne0 // info[0] * info[1]
    var src = List[UInt8](length=n * rb, fill=0)
    var dst = List[Float32](length=n * ne0, fill=0)
    pread_into(g.fd, src, 0, n * rb, g.data_offset + g.tensors[ti].offset + row0 * rb)
    deq_rows(t, src.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), ne0, n,
             dst.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    _ = len(src)  # raw pointer above: keep src alive past the call
    return dst^


def unbyte(t: String) -> String:
    """GPT-2 byte-level token to text: printable bytes map to themselves, the rest sit at U+0100 up."""
    var out = List[UInt8]()
    var low = List[Int]()
    for b in range(256):
        if not ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174):
            low.append(b)
    for cp in t.codepoints():
        var c = Int(cp)
        if c >= 256 and c - 256 < len(low):
            out.append(UInt8(low[c - 256]))
        elif c < 256:
            out.append(UInt8(c))
        else:
            for x in String(cp).as_bytes():
                out.append(x)
    return String(from_utf8_lossy=Span(out))


def vocab(g: Gguf) raises -> List[String]:
    var at = -1
    for k in range(len(g.kvs)):
        if g.kvs[k].key == "tokenizer.ggml.tokens":
            at = g.kvs[k].start
    if at < 0:
        raise Error("no tokenizer.ggml.tokens")
    var n = Int(uint_at(g, at + 4, 8))
    var out = List[String](capacity=n)
    var p = at + 12
    for _ in range(n):
        var m = Int(uint_at(g, p, 8))
        var b = peek(g, p + 8, m)
        var t = String(unsafe_from_utf8=Span(b))
        out.append(unbyte(t).replace("\t", "\\t").replace("\n", "\\n"))
        p += 8 + m
    return out^


def kv_at(g: Gguf, key: String) -> Int:
    """Payload offset of metadata `key`, -1 when absent."""
    for k in range(len(g.kvs)):
        if g.kvs[k].key == key:
            return g.kvs[k].start
    return -1


def kv_int(g: Gguf, key: String) raises -> Int:
    var at = kv_at(g, key)
    if at < 0:
        raise Error("missing metadata " + key)
    return Int(uint_at(g, at, 4))


def kv_f32(g: Gguf, key: String, default: Float32) raises -> Float32:
    var at = kv_at(g, key)
    if at < 0:
        return default
    return bitcast[DType.float32](UInt32(uint_at(g, at, 4)))


def kv_str(g: Gguf, key: String) raises -> String:
    var at = kv_at(g, key)
    if at < 0:
        raise Error("missing metadata " + key)
    return read_str(g, at)


def put_u32(mut b: List[UInt8], v: Int):
    for i in range(4):
        b.append(UInt8((v >> (8 * i)) & 255))


def put_f32(mut b: List[UInt8], v: Float32):
    put_u32(b, Int(bitcast[DType.uint32](v)))


def write_file(path: String, b: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(b))


def write_atomic(path: String, b: List[UInt8]) raises:
    """Write beside the target, then rename(2) over it: a reader never sees a half-written file."""
    var tmp = path + ".tmp"
    write_file(tmp, b)
    var ca = tmp.copy()
    var cb = path.copy()
    if external_call["rename", Int32](ca.as_c_string_span().ptr(), cb.as_c_string_span().ptr()) != 0:
        raise Error("rename " + tmp + " -> " + path + " failed")


def abs_path(p: String) raises -> String:
    """Absolute form of p, relative to the working directory, without resolving symlinks."""
    if p.startswith("/"):
        return p
    var i = 0
    while p[byte=i : i + 2] == "./":
        i += 2
    return String(realpath(".")) + "/" + String(p[byte=i : p.byte_length()])


def llama_ids(gguf: String, text: String, tmp: String) raises -> List[Int]:
    """Token ids of `text` as llama.cpp tokenizes it for this model (host llama-tokenize; LLAMA_BIN overrides the bin dir)."""
    var bd = getenv("LLAMA_BIN")
    if bd.byte_length() == 0:
        bd = String(getenv("HOME")) + "/Models/llama.cpp/build/bin"
    with open(tmp, "w") as f:
        f.write(text)
    var cmd = "LD_LIBRARY_PATH='" + bd + "' '" + bd + "/llama-tokenize' -m '" + gguf + "' -f '" + tmp + "' --ids --log-disable 2>/dev/null | tail -1"
    var ids = List[Int]()
    for p in String(run(cmd)).replace("[", "").replace("]", "").replace("\n", "").split(","):
        var t = String(String(p).strip())
        if t.byte_length() > 0:
            ids.append(Int(t))
    if len(ids) == 0:
        raise Error("llama-tokenize returned no ids (set LLAMA_BIN to the llama.cpp bin dir)")
    return ids^


def read_rmap(path: String, layer: Int, h: Int) raises -> List[Float32]:
    """The h x h transport map of `layer` from an rlens.bin (see src/rlens.mojo): row i of the result is output i."""
    var b: List[UInt8]
    with open(path, "r") as f:
        b = f.read_bytes()
    if len(b) < 16 or b[0] != 77 or b[1] != 78 or b[2] != 82 or b[3] != 76:
        raise Error(path + ": not an rlens.bin")
    var hh = Int(b[8]) | (Int(b[9]) << 8) | (Int(b[10]) << 16) | (Int(b[11]) << 24)
    var cnt = Int(b[12]) | (Int(b[13]) << 8) | (Int(b[14]) << 16) | (Int(b[15]) << 24)
    if hh != h:
        raise Error(path + ": hidden " + String(hh) + " != model hidden " + String(h))
    var at = 16
    for _ in range(cnt):
        var l = Int(b[at]) | (Int(b[at + 1]) << 8) | (Int(b[at + 2]) << 16) | (Int(b[at + 3]) << 24)
        if l == layer:
            var m = List[Float32](length=h * h, fill=0)
            for i in range(h * h):
                var o = at + 4 + 4 * i
                m[i] = bitcast[DType.float32](UInt32(Int(b[o]) | (Int(b[o + 1]) << 8) | (Int(b[o + 2]) << 16) | (Int(b[o + 3]) << 24)))
            return m^
        at += 4 + 4 * h * h
    raise Error(path + ": no map for layer " + String(layer))
