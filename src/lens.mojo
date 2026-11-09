"""Lens MODEL LAYER K [CHECK] [--out DIR]: logit lens of every MLP neuron of one layer, on CPU.

For neuron n: g = unit(W_down[:, n]) * output_norm, z = W_U g. Prints one line per neuron:
`n<TAB>top k tokens<TAB>bottom k tokens`, tokens joined by U+241F. W_U is streamed in row chunks;
each thread owns a block of neurons, so no locks. CHECK > 0: recompute CHECK sampled neurons in f64
by brute force and compare top-k ids (golden), and against the neighbour neuron (mutant must differ).
--out DIR writes DIR/<MODEL stem>/<layer>.lens.bin and model.json (docs/format.md) instead of text.
--rmap FILE reads each write direction through the ridge map of that layer first (src/rlens.mojo): the R readout, written as <layer>.rlens.bin.
"""
from std.sys import argv, exit
from std.math import sqrt
from std.os import makedirs
from ggufx.model import Gguf
from ggufx.par import parallelize
from json import Json, parse
from std.os.path import exists
from mneuron.ggio import tensor, load_rows, vocab, kv_int, kv_str, put_u32, put_f32, write_file, write_atomic, abs_path, read_rmap
from mneuron.fwd import matmul
from ggufx.model import Gguf, pread_into
from ggufx.dequant import quant_block, deq_rows
from ggufx.values import uint_at, peek

comptime W = 16
comptime ROWS = 2048  # W_U rows per chunk
comptime NB = 8  # neurons per task


@fieldwise_init
struct Job(Copyable):
    var G: Pointer[Float32, MutUntrackedOrigin]  # FFN x H, row n = g of neuron n
    var U: Pointer[Float32, MutUntrackedOrigin]  # rows x H
    var tv: Pointer[Float32, MutUntrackedOrigin]  # FFN x 2K: top k then bottom k (negated), ascending
    var ti: Pointer[Int32, MutUntrackedOrigin]
    var h: Int
    var ffn: Int
    var k: Int
    var rows: Int
    var row0: Int


@always_inline
def push(v: Pointer[Float32, MutUntrackedOrigin], ix: Pointer[Int32, MutUntrackedOrigin], k: Int, z: Float32, id: Int):
    """Sorted ascending, the first slot is the smallest kept; insert z if it beats it."""
    if z <= v[unsafe_offset=0] or z != z or z - z != 0:  # also drops NaN and inf (a W_U row that dequantizes to junk)
        return
    var j = 0
    while j + 1 < k and v[unsafe_offset=j + 1] < z:
        v[unsafe_offset=j] = v[unsafe_offset=j + 1]
        ix[unsafe_offset=j] = ix[unsafe_offset=j + 1]
        j += 1
    v[unsafe_offset=j] = z
    ix[unsafe_offset=j] = Int32(id)


def task(ctx: Int, w: Int):
    var c = Pointer[Job, MutUntrackedOrigin](unsafe_from_address=ctx)
    var h = c[].h
    var k = c[].k
    var n0 = w * NB
    var n1 = min(n0 + NB, c[].ffn)
    for r in range(c[].rows):
        var u = c[].U.unsafe_offset(r * h)
        for n in range(n0, n1):
            var gp = c[].G.unsafe_offset(n * h)
            var acc = SIMD[DType.float32, W](0)
            for j in range(0, h, W):
                acc += gp.unsafe_offset(j).unsafe_load[width=W]() * u.unsafe_offset(j).unsafe_load[width=W]()
            var z = acc.reduce_add()
            push(c[].tv.unsafe_offset(n * 2 * k), c[].ti.unsafe_offset(n * 2 * k), k, z, c[].row0 + r)
            push(c[].tv.unsafe_offset(n * 2 * k + k), c[].ti.unsafe_offset(n * 2 * k + k), k, -z, c[].row0 + r)


def z64(G: List[Float32], U: List[Float32], n: Int, h: Int, r: Int, sgn: Float64) -> Float64:
    var z: Float64 = 0
    for j in range(h):
        z += Float64(G[n * h + j]) * Float64(U[r * h + j])
    return z * sgn


def brute_top(G: List[Float32], U: List[Float32], n: Int, h: Int, v: Int, k: Int, sgn: Float64 = 1.0) -> List[Int]:
    """Best k rows by sgn * z in f64, best first (sgn -1: the bottom pole)."""
    var best = List[Float64]()
    var ids = List[Int]()
    for r in range(v):
        var p = len(best)
        best.append(z64(G, U, n, h, r, sgn))
        ids.append(r)
        while p > 0 and best[p - 1] < best[p]:
            best.swap_elements(p - 1, p)
            ids.swap_elements(p - 1, p)
            p -= 1
        if len(best) > k:
            _ = best.pop()
            _ = ids.pop()
    return ids^


def same(G: List[Float32], U: List[Float32], n: Int, h: Int, a: List[Int], tv: List[Int32], at: Int, k: Int, sgn: Float64, rev: Bool = False) -> Bool:
    """Rank order, not set membership: position j of the stored list (stored ascending, slot k-1 is best) must hold the
    f64 rank-j id, or an id whose f64 score ties it to 1e-5 relative (f32 accumulation noise swaps near-ties). rev: the mutant, best-last."""
    for j in range(k):
        var s = Int(tv[at + (j if rev else k - 1 - j)])
        if s == a[j]:
            continue
        if s < 0:
            return False
        var want = z64(G, U, n, h, a[j], sgn)
        var got = z64(G, U, n, h, s, sgn)
        var d = want - got
        if d < 0:
            d = -d
        if d > 1e-5 * (abs(want) + 1.0):
            return False
    return True


def stem(path: String) -> String:
    var parts = path.split("/")
    var name = String(parts[len(parts) - 1])
    var dot = name.rfind(".")
    var base = String(name[byte=0:dot]) if dot > 0 else name
    return base.lower()


def write_bin(g: Gguf, path: String, root: String, L: Int, ffn: Int, h: Int, v: Int, k: Int, ti: List[Int32], tv: List[Float32], suffix: String) raises:
    """Write lens.bin v1 (docs/format.md): per neuron top ids, top z, bottom ids, bottom z, best first."""
    var b = List[UInt8](capacity=24 + ffn * k * 16)
    for c in String("MNLN").as_bytes():
        b.append(c)
    put_u32(b, 1)
    put_u32(b, L)
    put_u32(b, ffn)
    put_u32(b, h)
    put_u32(b, k)
    for n in range(ffn):
        var o = n * 2 * k
        for i in range(k):
            put_u32(b, Int(ti[o + k - 1 - i]) & 0xFFFFFFFF)
        for i in range(k):
            put_f32(b, tv[o + k - 1 - i] if ti[o + k - 1 - i] >= 0 else Float32(0))
        for i in range(k):
            put_u32(b, Int(ti[o + 2 * k - 1 - i]) & 0xFFFFFFFF)
        for i in range(k):
            put_f32(b, -tv[o + 2 * k - 1 - i] if ti[o + 2 * k - 1 - i] >= 0 else Float32(0))
    var dir = root + "/" + stem(path)
    makedirs(dir, exist_ok=True)
    write_file(dir + "/" + String(L) + suffix, b)
    var arch = kv_str(g, "general.architecture")
    var mpath = dir + "/model.json"
    var m = Json.object()
    if exists(mpath):  # merge: the engine owns rho, rho_src and any field this tool does not know; a corrupt file stops the run
        with open(mpath, "r") as f:
            m = parse(f.read())
        if m.kind() != 6:
            raise Error(mpath + " is not a JSON object; fix or remove it, lens will not overwrite it")
    m.set("v", Json.of(1))
    m.set("model", Json.of(stem(path)))
    m.set("gguf", Json.of(abs_path(path)))
    m.set("arch", Json.of(arch))
    m.set("n_layer", Json.of(kv_int(g, arch + ".block_count")))
    m.set("hidden", Json.of(h))
    m.set("ffn", Json.of(ffn))
    m.set("vocab", Json.of(v))
    if not m.has("rho"):
        m.set("rho", Json.array())
        m.set("rho_src", Json.of(String("not measured by lens; lane engine fills")))
    write_atomic(mpath, List[UInt8]((m.dumps() + "\n").as_bytes()))
    print("wrote", dir + "/" + String(L) + suffix)


def run() raises:
    var a = List[String]()
    var root = String()
    var rmap = String()
    var tcd = String()
    var tcn = 0
    var av = argv()
    var i = 1
    while i < len(av):
        if String(av[i]) == "--out" and i + 1 < len(av):
            root = String(av[i + 1])
            i += 2
        elif String(av[i]) == "--rmap" and i + 1 < len(av):
            rmap = String(av[i + 1])
            i += 2
        elif String(av[i]) == "--tc" and i + 2 < len(av):
            tcd = String(av[i + 1])
            tcn = Int(String(av[i + 2]))
            i += 3
        else:
            a.append(String(av[i]))
            i += 1
    if len(a) < 3:
        raise Error("usage: lens MODEL LAYER K [CHECK] [--out DIR] [--rmap rlens.bin] [--tc DIR NF]")
    var g = Gguf(a[0])
    var L = Int(a[1])
    var k = Int(a[2])
    var check = Int(a[3]) if len(a) > 3 else 0
    var di = tensor(g, "blk." + String(L) + ".ffn_down.weight")
    var ni = tensor(g, "output_norm.weight")
    var ui = tensor(g, "output.weight")
    if ui < 0:
        ui = tensor(g, "token_embd.weight")
    if di < 0 or ni < 0 or ui < 0:
        raise Error("missing ffn_down, output_norm or W_U for layer " + String(L))
    var ffn = tcn if tcn > 0 else g.tensors[di].dims[0]
    var h = g.tensors[di].dims[1]
    var v = g.tensors[ui].dims[1]
    if h % W != 0 or g.tensors[ui].dims[0] != h:
        raise Error("hidden size " + String(h) + " not a multiple of " + String(W) + " or W_U mismatch")
    var nw = load_rows(g, ni, 0, 1)
    var D = List[Float32]()
    if tcn > 0:  # features: decoder rows of tc_train's tc.bin (We h x NF, be, Wd NF x h, bd), stored transposed like W_down (h x NF)
        var tb = open(tcd + "/tc.bin", "r").read_bytes()
        var tp = tb.unsafe_ptr().unsafe_bitcast[Float32]()
        D = List[Float32](length=h * ffn, fill=0)
        for n in range(ffn):
            for j in range(h):
                D[j * ffn + n] = tp[h * ffn + ffn + n * h + j]
    else:
        D = load_rows(g, di, 0, h)
    var G = List[Float32](length=ffn * h, fill=0)
    for n in range(ffn):
        var s: Float64 = 0
        for j in range(h):
            s += Float64(D[j * ffn + n]) ** 2
        var inv = Float32(1.0 / sqrt(s)) if s > 0 else Float32(0)
        for j in range(h):
            G[n * h + j] = D[j * ffn + n] * inv * nw[j]
    if rmap.byte_length() > 0:
        var M = read_rmap(rmap, L, h)
        var Uu = List[Float32](length=ffn * h, fill=0)
        for n in range(ffn):
            var s: Float64 = 0
            for j in range(h):
                s += Float64(D[j * ffn + n]) ** 2
            var inv = Float32(1.0 / sqrt(s)) if s > 0 else Float32(0)
            for j in range(h):
                Uu[n * h + j] = D[j * ffn + n] * inv
        var Tm = matmul(M, Uu, h, h, ffn)
        for n in range(ffn):
            var s: Float64 = 0
            for j in range(h):
                s += Float64(Tm[n * h + j]) ** 2
            var inv = Float32(1.0 / sqrt(s)) if s > 0 else Float32(0)
            for j in range(h):
                G[n * h + j] = Tm[n * h + j] * inv * nw[j]
    var tv = List[Float32](length=ffn * 2 * k, fill=Float32.MIN)
    var ti = List[Int32](length=ffn * 2 * k, fill=-1)
    var U = List[Float32](length=(v if check > 0 else ROWS) * h, fill=0)
    var r0 = 0
    while r0 < v:
        var m = min(ROWS, v - r0)
        var off = r0 * h if check > 0 else 0
        var chunk = load_rows(g, ui, r0, m)
        for j in range(m * h):
            U[off + j] = chunk[j]
        var job: List[Job] = [Job(G.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                      U.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_offset(off),
                      tv.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                      ti.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), h, ffn, k, m, r0)]  # heap ctx, see gguf deq_rows
        parallelize(task, Int(job.unsafe_ptr()), (ffn + NB - 1) // NB)
        _ = len(job) + len(G) + len(U) + len(tv) + len(ti)
        r0 += m
    if check > 0:
        var bad = 0
        for s in range(check):
            var n = s * (ffn // check)
            for pole in range(2):
                var sgn: Float64 = 1.0 if pole == 0 else -1.0
                var ids = brute_top(G, U, n, h, v, k, sgn)
                var at = n * 2 * k + pole * k
                if not same(G, U, n, h, ids, ti, at, k, sgn):
                    bad += 1
                if k > 1 and same(G, U, n, h, ids, ti, at, k, sgn, True):
                    raise Error("mutant passed: reversed rank order of neuron " + String(n) + " pole " + String(pole))
                if same(G, U, n, h, ids, ti, (((n + 1) % ffn) * 2 * k) + pole * k, k, sgn):
                    raise Error("mutant passed: neighbour neuron " + String(n + 1) + " has the same ranked list")
                for j in range(k - 1):  # kept scores must be monotone (slot k-1 is best)
                    if tv[at + j] > tv[at + j + 1]:
                        raise Error("scores not monotone at neuron " + String(n))
        if bad > 0:
            raise Error(String(bad) + " of " + String(check) + " sampled neurons differ from the f64 brute force")
        print("PASS lens check:", check, "neurons, top and bottom", k, "rank order matches f64, reversed and neighbour mutants differ")
        return
    if root.byte_length() > 0:
        write_bin(g, a[0], root, L, ffn, h, v, k, ti, tv, ".rlens.bin" if rmap.byte_length() > 0 else (".tclens.bin" if tcn > 0 else ".lens.bin"))
        return
    var tok = vocab(g)
    var sep = String("\u241f")
    for n in range(ffn):
        var line = String(n) + "\t"
        for i in range(k):
            var id = Int(ti[n * 2 * k + k - 1 - i])
            line += (sep if i > 0 else "") + (tok[id] if id >= 0 else "")
        line += "\t"
        for i in range(k):
            var id = Int(ti[n * 2 * k + 2 * k - 1 - i])
            line += (sep if i > 0 else "") + (tok[id] if id >= 0 else "")
        print(line)


def main():
    try:
        run()
    except e:
        print("FAIL lens:", e)
        exit(1)
