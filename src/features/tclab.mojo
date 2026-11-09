"""tclab: numeric parts of the B4 label gate (docs/prereg-b4.md), 0.5B-class dense model, one transcoder layer.
  tclab ctx   TC NF K MODEL LAYER CAP PICK OUT      OUT/ctx_<f>.txt (top 10 contexts), OUT/mid_<f>.txt (20 at ranks 50-500), OUT/p99.tsv (p99 of the 32-token window max)
  tclab act   TC NF K MODEL LAYER PICK TEXTS OUT    OUT: per text (one per line) the max activation of each picked feature, tab separated
  tclab steer TC NF K MODEL LAYER CAP PICK PROMPTS WORDS DOSE OUT   logprob of WORDS[j]'s first token after PROMPTS under steering feature i (dose x mean |y| x unit decoder row,
                                                    added after block LAYER); row "base" unsteered. OUT: i, j, delta; last column per feature i: pass = delta_ii > median_j!=i delta_ji
Windows are 32 tokens (padded at the end; the causal forward ignores the pad)."""
from std.sys import argv
from std.math import exp, log, sqrt
from ggufx.model import Gguf
from mneuron.ggio import kv_int, kv_str, kv_f32, tensor, load_rows, llama_ids, vocab, write_file
from mneuron.fwd import forward, matmul, rmsnorm
from mneuron.store import read_text
from mneuron.rlog import Rlog
from tcenc import Enc

comptime SEG = 32


def feats_of(path: String) raises -> List[Int]:
    var r = List[Int]()
    for t in read_text(path).split(" "):
        var s = String(String(t).strip())
        if s.byte_length() > 0:
            r.append(Int(s))
    return r^


def lines_of(path: String) raises -> List[String]:
    var r = List[String]()
    for t in read_text(path).split("\n"):
        var s = String(String(t).strip())
        if s.byte_length() > 0:
            r.append(s)
    return r^


def put(path: String, s: String) raises:
    var b = List[UInt8]()
    for c in s.as_bytes():
        b.append(c)
    write_file(path, b)


def windows(g: Gguf, gg: String, texts: List[String], tmp: String) raises -> Tuple[List[Int], List[Int]]:
    """Token ids of every text padded/truncated to SEG, and the real length of each."""
    var ids = List[Int]()
    var lens = List[Int]()
    for t in texts:
        var tk = llama_ids(gg, t, tmp)
        var n = min(len(tk), SEG)
        for j in range(SEG):
            ids.append(tk[j] if j < n else tk[n - 1])
        lens.append(n)
    return (ids^, lens^)


def main() raises:
    var a = argv()
    var mode = String(a[1])
    var tc = String(a[2])
    var nf = Int(String(a[3]))
    var k = Int(String(a[4]))
    var gg = String(a[5])
    var layer = Int(String(a[6]))
    var g = Gguf(gg)
    var rl = Rlog("tclab", mode + " " + tc, gg)
    var h = 0
    if mode == "ctx":
        var cap = String(a[7])
        var fs = feats_of(String(a[8]))
        var out = String(a[9])
        var meta = open(cap + "/meta.json", "r").read_bytes()
        var tokb = open(cap + "/tokens.i32", "r").read_bytes()
        var n = len(tokb) // 4
        var tok = tokb.unsafe_ptr().unsafe_bitcast[Int32]()
        h = kv_int(g, kv_str(g, "general.architecture") + ".embedding_length")
        var enc = Enc(tc, h, nf, k)
        var pool = 12288
        var t0 = n - pool
        var rec = open(cap + "/L" + String(layer) + ".f16", "r")
        _ = rec.seek(t0 * 2 * h * 2)
        var rb = rec.read_bytes(pool * 2 * h * 2)
        var hp = rb.unsafe_ptr().unsafe_bitcast[Float16]()
        var x = List[Float32](length=pool * h, fill=0)
        for t in range(pool):
            for j in range(h):
                x[t * h + j] = hp[t * 2 * h + j].cast[DType.float32]()
        var A = enc.acts(x, pool)
        var voc = vocab(g)
        var p99 = String()
        for f in fs:
            var col = List[Float32]()
            for t in range(pool):
                col.append(A[t * nf + f])
            var keys = List[Int]()  # activation to 1e-4, then position: sorts by activation
            for t in range(pool):
                keys.append(Int(col[t] * 10000) * 16384 + t)
            sort(keys)
            var order = List[Int]()
            for q in range(pool - 1, -1, -1):
                order.append(keys[q] % 16384)
            var wm = List[Float32]()
            for w in range(pool // SEG):
                var m = Float32(0)
                for j in range(SEG):
                    m = max(m, col[w * SEG + j])
                wm.append(m)
            sort(wm)
            p99 += String(f) + "\t" + String(wm[len(wm) * 99 // 100]) + "\n"
            var top = String()
            var taken = List[Int]()
            for t in order:
                var ok = col[t] > 0 and len(taken) < 10
                for u in taken:
                    if abs(u - t) < 16:
                        ok = False
                if ok:
                    taken.append(t)
            for t in taken:
                var s = String()
                for u in range(max(0, t - 8), min(pool, t + 9)):
                    s += ("<<" + voc[Int(tok[t0 + u])] + ">>") if u == t else voc[Int(tok[t0 + u])]
                top += s.replace("\n", " ") + "\n"
            put(out + "/ctx_" + String(f) + ".txt", top)
            var mid = String()
            var nact = 0
            for t in order:
                if col[t] > 0:
                    nact += 1
            var hi = min(500, nact)
            var cnt = 0
            var q = 50
            while q < hi and cnt < 20:
                var t = order[q]
                var s = String()
                for u in range(max(0, t - 8), min(pool, t + 9)):
                    s += ("<<" + voc[Int(tok[t0 + u])] + ">>") if u == t else voc[Int(tok[t0 + u])]
                mid += s.replace("\n", " ") + "\n"
                cnt += 1
                q += max(1, (hi - 50) // 20)
            put(out + "/mid_" + String(f) + ".txt", mid)
        _ = len(tokb) + len(rb) + len(meta)
        put(out + "/p99.tsv", p99)
        print("ctx done", len(fs), "features, pool", pool)
    elif mode == "act":
        var fs = feats_of(String(a[7]))
        var texts = lines_of(String(a[8]))
        var w = windows(g, gg, texts, String(a[9]) + ".tmp")
        var r = forward(g, w[0], List[Int](), List[Int](), SEG, layer)
        h = len(r[2][0]) // len(w[0])
        var enc = Enc(tc, h, nf, k)
        var A = enc.acts(r[2][0], len(w[0]))
        var s = String()
        for ti in range(len(texts)):
            for q in range(len(fs)):
                var m = Float32(0)
                for j in range(w[1][ti]):
                    m = max(m, A[(ti * SEG + j) * nf + fs[q]])
                s += String(m) + ("\t" if q + 1 < len(fs) else "\n")
        put(String(a[9]), s)
        print("act done", len(texts), "texts")
    elif mode == "steer":
        var cap = String(a[7])
        var fs = feats_of(String(a[8]))
        var prompts = lines_of(String(a[9]))
        var words = lines_of(String(a[10]))
        var dose = Float64(String(a[11]))
        var out = String(a[12])
        h = kv_int(g, kv_str(g, "general.architecture") + ".embedding_length")
        var enc = Enc(tc, h, nf, k)
        var rec = open(cap + "/L" + String(layer) + ".f16", "r")
        var rb = rec.read_bytes(2000 * 2 * h * 2)
        var hp = rb.unsafe_ptr().unsafe_bitcast[Float16]()
        var ybar: Float64 = 0
        for t in range(2000):
            var q: Float64 = 0
            for j in range(h):
                q += Float64(hp[t * 2 * h + h + j]) ** 2
            ybar += sqrt(q) / 2000.0
        _ = len(rb)
        var w = windows(g, gg, prompts, out + ".tmp")
        var wid = List[Int]()
        for wd in words:
            wid.append(llama_ids(gg, " " + wd, out + ".tmp")[0])
        var arch = kv_str(g, "general.architecture")
        var nl = kv_int(g, arch + ".block_count")
        var ni = tensor(g, "output_norm.weight")
        var ui = tensor(g, "output.weight")
        if ui < 0:
            ui = tensor(g, "token_embd.weight")
        var nv = g.tensors[ui].dims[1]
        var wu = load_rows(g, ui, 0, nv)
        var nw_ = load_rows(g, ni, 0, 1)
        var eps = kv_f32(g, arch + ".attention.layer_norm_rms_epsilon", 1e-5)
        var P = len(prompts)
        # lp[v][j]: mean over prompts of the log-probability of word j's first token, v = 0 unsteered, 1.. steering feature v-1
        var lp = List[List[Float64]]()
        for v in range(len(fs) + 1):
            var inj = List[Float32]()
            if v > 0:
                var nrm: Float64 = 0
                for j in range(h):
                    nrm += Float64(enc.wd[fs[v - 1] * h + j]) ** 2
                nrm = sqrt(nrm)
                for j in range(h):
                    inj.append(Float32(Float64(enc.wd[fs[v - 1] * h + j]) / nrm * dose * ybar))
            var r = forward(g, w[0], List[Int](), [nl - 1], SEG, -1, List[Float32](), layer if v > 0 else -1, inj)
            var xn = rmsnorm(r[1][0], nw_, len(w[0]), h, eps)
            var rows = List[Float32](length=P * h, fill=0)
            for p in range(P):
                for j in range(h):
                    rows[p * h + j] = xn[(p * SEG + w[1][p] - 1) * h + j]
            var lg = matmul(wu, rows, nv, h, P)
            var row = List[Float64](length=len(words), fill=0)
            for p in range(P):
                var mx = Float32.MIN
                for t in range(nv):
                    mx = max(mx, lg[p * nv + t])
                var sm: Float64 = 0
                for t in range(nv):
                    sm += exp(Float64(lg[p * nv + t] - mx))
                for j in range(len(words)):
                    row[j] += (Float64(lg[p * nv + wid[j]] - mx) - log(sm)) / Float64(P)
            lp.append(row^)
        var s = String()
        var npass = 0
        for i in range(len(fs)):
            var dii = lp[i + 1][i] - lp[0][i]
            var others = List[Float64]()
            for j in range(len(fs)):
                if j != i:
                    others.append(lp[j + 1][i] - lp[0][i])
            sort(others)
            var med = others[len(others) // 2]
            var ok = dii > med
            npass += Int(ok)
            s += String(fs[i]) + "\t" + words[i] + "\t" + String(dii) + "\t" + String(med) + "\t" + ("pass" if ok else "fail") + "\n"
        put(out, s)
        print("steer-first: " + String(npass) + " of " + String(len(fs)) + " features move their label word more than the median control")
        var vals = List[Float64]()
        vals.append(Float64(npass))
        rl.done("steer_first_pass=" + String(npass) + "/" + String(len(fs)), out, vals, 0.0, 1e6)
