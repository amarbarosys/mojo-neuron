"""CPU forward of dense decoder GGUFs (llama, qwen2, granite) that stops at a layer and returns the MLP activations
a[t][n] = silu(gate_n(x_t)) * up_n(x_t), the quantity ffn_down reads. Weights are dequantized one layer at a time and
applied to all tokens at once, so cost is one pass over the weights plus 2*T*params flops. f32 throughout."""
from std.math import sqrt, exp, cos, sin, log
from ggufx.model import Gguf
from ggufx.par import parallelize
from mneuron.ggio import tensor, load_rows, kv_int, kv_f32, kv_str, kv_at

comptime W = 16


@fieldwise_init
struct MM(Copyable):
    var w: Pointer[Float32, ImmUntrackedOrigin]  # out x inn
    var x: Pointer[Float32, ImmUntrackedOrigin]  # T x inn
    var y: Pointer[Float32, MutUntrackedOrigin]  # T x out
    var nout: Int
    var inn: Int
    var T: Int


def mm_task(ctx: Int, i: Int):
    var c = Pointer[MM, ImmUntrackedOrigin](unsafe_from_address=ctx)
    var inn = c[].inn
    var o0 = i * 8
    var o1 = min(o0 + 8, c[].nout)
    for o in range(o0, o1):
        var wp = c[].w.unsafe_offset(o * inn)
        for t in range(c[].T):
            var xp = c[].x.unsafe_offset(t * inn)
            var acc = SIMD[DType.float32, W](0)
            var j = 0
            while j + W <= inn:
                acc += wp.unsafe_offset(j).unsafe_load[width=W]() * xp.unsafe_offset(j).unsafe_load[width=W]()
                j += W
            var s = acc.reduce_add()
            while j < inn:
                s += wp[unsafe_offset=j] * xp[unsafe_offset=j]
                j += 1
            c[].y[unsafe_offset=t * c[].nout + o] = s


def matmul(w: List[Float32], x: List[Float32], nout: Int, inn: Int, T: Int) -> List[Float32]:
    """y[t][o] = sum_j w[o][j] * x[t][j]."""
    var y = List[Float32](length=T * nout, fill=0)
    var job: List[MM] = [MM(w.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](), x.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
                            y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), nout, inn, T)]  # heap ctx, see lens
    parallelize(mm_task, Int(job.unsafe_ptr()), (nout + 7) // 8)
    _ = len(job) + len(w) + len(x) + len(y)
    return y^


def rmsnorm(x: List[Float32], w: List[Float32], T: Int, h: Int, eps: Float32) -> List[Float32]:
    var y = List[Float32](length=T * h, fill=0)
    for t in range(T):
        var s: Float64 = 0
        for j in range(h):
            s += Float64(x[t * h + j]) * Float64(x[t * h + j])
        var r = Float32(1.0 / sqrt(s / Float64(h) + Float64(eps)))
        for j in range(h):
            y[t * h + j] = x[t * h + j] * r * w[j]
    return y^


def add_bias(mut y: List[Float32], b: List[Float32], T: Int, nout: Int):
    for t in range(T):
        for o in range(nout):
            y[t * nout + o] += b[o]


def rope(mut v: List[Float32], T: Int, nh: Int, hd: Int, rd: Int, base: Float32, neox: Bool, ff: List[Float32], seg: Int):
    for t in range(T):
        for hh in range(nh):
            var o = (t * nh + hh) * hd
            for i in range(rd // 2):
                var fr = Float64(base) ** (-2.0 * Float64(i) / Float64(rd))
                if len(ff) > i:
                    fr = fr / Float64(ff[i])
                var a = Float64(t % seg if seg > 0 else t) * fr
                var c = Float32(cos(a))
                var s = Float32(sin(a))
                var i0 = o + (i if neox else 2 * i)
                var i1 = o + (i + rd // 2 if neox else 2 * i + 1)
                var x0 = v[i0]
                var x1 = v[i1]
                v[i0] = x0 * c - x1 * s
                v[i1] = x0 * s + x1 * c


def fires(g: Gguf, ids: List[Int], layers: List[Int]) raises -> List[List[Float32]]:
    """Activations after silu(gate)*up of each requested layer: one T x ffn list per layer (same order as `layers`)."""
    return forward(g, ids, layers, List[Int]())[0].copy()


def forward(g: Gguf, ids: List[Int], layers: List[Int], rl: List[Int], seg: Int = 0, ml: Int = -1, ov: List[Float32] = List[Float32](), il: Int = -1, inj: List[Float32] = List[Float32]()) raises -> Tuple[List[List[Float32]], List[List[Float32]], List[List[Float32]]]:
    """Activations of `layers` and the residual stream (T x hidden, after the block, before any final norm) after each block in `rl`.
    inj (hidden, with il >= 0) is added to the residual of every position after block il (a steering vector).
    ov (T x hidden, with ml >= 0): replaces the MLP output of block ml (splice a transcoder, or zeros to ablate the MLP).
    ml >= 0: third result = [MLP input (post ffn_norm), MLP output] of block ml, T x hidden each.
    seg > 0: ids holds T/seg independent sequences of seg tokens each (positions restart, attention stays inside a sequence)."""
    var arch = kv_str(g, "general.architecture")
    if arch != "llama" and arch != "qwen2" and arch != "granite":
        raise Error("fires: CPU forward supports llama, qwen2, granite; got " + arch)
    var T = len(ids)
    var nh = kv_int(g, arch + ".attention.head_count")
    var nkv = kv_int(g, arch + ".attention.head_count_kv") if kv_at(g, arch + ".attention.head_count_kv") >= 0 else nh
    var eps = kv_f32(g, arch + ".attention.layer_norm_rms_epsilon", 1e-5)
    var base = kv_f32(g, arch + ".rope.freq_base", 10000.0)
    var emb = tensor(g, "token_embd.weight")
    var h = g.tensors[emb].dims[0]
    var maxl = 0
    for l in layers:
        maxl = max(maxl, l)
    for l in rl:
        maxl = max(maxl, l)
    maxl = max(maxl, ml)
    var hd = g.tensors[tensor(g, "blk.0.attn_q.weight")].dims[1] // nh
    var rd = kv_int(g, arch + ".rope.dimension_count") if kv_at(g, arch + ".rope.dimension_count") >= 0 else hd
    var neox = arch == "qwen2"
    var ff = List[Float32]()
    var fi = tensor(g, "rope_freqs.weight")
    if fi >= 0:
        ff = load_rows(g, fi, 0, 1)
    var escale = kv_f32(g, arch + ".embedding_scale", 1.0)
    var rscale = kv_f32(g, arch + ".residual_scale", 1.0)
    var ascale = kv_f32(g, arch + ".attention.scale", Float32(1.0 / sqrt(Float64(hd))))
    var x = List[Float32](length=T * h, fill=0)
    for t in range(T):
        var row = load_rows(g, emb, ids[t], 1)
        for j in range(h):
            x[t * h + j] = row[j] * escale
    var res = List[List[Float32]]()
    for _ in range(len(layers)):
        res.append(List[Float32]())
    var rres = List[List[Float32]]()
    for _ in range(len(rl)):
        rres.append(List[Float32]())
    var mlp = List[List[Float32]]()
    var qd = nh * hd
    var kd = nkv * hd
    for l in range(maxl + 1):
        var p = "blk." + String(l) + "."
        var xn = rmsnorm(x, load_rows(g, tensor(g, p + "attn_norm.weight"), 0, 1), T, h, eps)
        var q = matmul(load_rows(g, tensor(g, p + "attn_q.weight"), 0, qd), xn, qd, h, T)
        var k = matmul(load_rows(g, tensor(g, p + "attn_k.weight"), 0, kd), xn, kd, h, T)
        var v = matmul(load_rows(g, tensor(g, p + "attn_v.weight"), 0, kd), xn, kd, h, T)
        if tensor(g, p + "attn_q.bias") >= 0:
            add_bias(q, load_rows(g, tensor(g, p + "attn_q.bias"), 0, 1), T, qd)
            add_bias(k, load_rows(g, tensor(g, p + "attn_k.bias"), 0, 1), T, kd)
            add_bias(v, load_rows(g, tensor(g, p + "attn_v.bias"), 0, 1), T, kd)
        rope(q, T, nh, hd, rd, base, neox, ff, seg)
        rope(k, T, nkv, hd, rd, base, neox, ff, seg)
        var att = List[Float32](length=T * qd, fill=0)
        var grp = nh // nkv
        for t in range(T):
            for hh in range(nh):
                var kh = hh // grp
                var u0 = (t // seg) * seg if seg > 0 else 0
                var sc = List[Float32](length=t + 1 - u0, fill=0)
                var mx = Float32.MIN
                for u in range(u0, t + 1):
                    var d: Float32 = 0
                    for j in range(hd):
                        d += q[(t * nh + hh) * hd + j] * k[(u * nkv + kh) * hd + j]
                    sc[u - u0] = d * ascale
                    mx = max(mx, sc[u - u0])
                var den: Float32 = 0
                for u in range(u0, t + 1):
                    sc[u - u0] = Float32(exp(Float64(sc[u - u0] - mx)))
                    den += sc[u - u0]
                for u in range(u0, t + 1):
                    var wgt = sc[u - u0] / den
                    for j in range(hd):
                        att[(t * nh + hh) * hd + j] += wgt * v[(u * nkv + kh) * hd + j]
        var o = matmul(load_rows(g, tensor(g, p + "attn_output.weight"), 0, h), att, h, qd, T)
        for i in range(T * h):
            x[i] += o[i] * rscale
        var xn2 = rmsnorm(x, load_rows(g, tensor(g, p + "ffn_norm.weight"), 0, 1), T, h, eps)
        var gi = tensor(g, p + "ffn_gate.weight")
        var ffn = g.tensors[gi].dims[1]
        var ga = matmul(load_rows(g, gi, 0, ffn), xn2, ffn, h, T)
        var up = matmul(load_rows(g, tensor(g, p + "ffn_up.weight"), 0, ffn), xn2, ffn, h, T)
        for i in range(T * ffn):
            var z = Float64(ga[i])
            ga[i] = Float32(z / (1.0 + exp(-z))) * up[i]
        for r in range(len(layers)):
            if layers[r] == l:
                res[r] = ga.copy()
        var want_res = False
        for r in range(len(rl)):
            if rl[r] == l:
                want_res = True
        if l < maxl or want_res or l == ml:
            var dn = matmul(load_rows(g, tensor(g, p + "ffn_down.weight"), 0, h), ga, h, ffn, T)  # rows of down: [h][ffn]
            if l == ml:
                mlp.append(xn2.copy())
                mlp.append(dn.copy())
                if len(ov) > 0:
                    dn = ov.copy()
            for i in range(T * h):
                x[i] += dn[i] * rscale
            if l == il and len(inj) == h:
                for t in range(T):
                    for j in range(h):
                        x[t * h + j] += inj[j]
            for r in range(len(rl)):
                if rl[r] == l:
                    rres[r] = x.copy()
    return (res^, rres^, mlp^)
