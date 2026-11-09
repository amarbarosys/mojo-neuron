"""TopK transcoder trainer on the GPU (B1). x (B x din) -> z = x We + be -> TopK -> yh = a Wd + bd, loss = SSE / (B * vtot) = FVU.
GEMMs, `F`, `grid` come from miniagi/train/ops.mojo (mojo-baro). Tie rule: active = z >= t_K and z > 0, t_K = K-th largest value of the row.
`-D MUT=n` plants a defect for the parity gate: 1 TopK keeps K+1, 2 no W_dec renorm, 3 db_dec sign flipped."""
from std.math import sqrt
from std.sys import get_defined_int
from std.memory import bitcast
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, THREADS, grid, gemm
from fast import Wmma

comptime MUT = get_defined_int["MUT", 0]()
comptime H16 = MutPointer[Float16, MutUntrackedOrigin]


def split_k(s: H16, x: F, y: F, xm: F, rows32: Int32, h32: Int32, stride32: Int32, sx: Float32):
    """Records [x | y (| r)] of f16 into the f32 x and y buffers (rows x h)."""
    var i = global_idx.x
    var h = Int(h32)
    if i < Int(rows32) * h:
        var r = i // h
        var j = i % h
        x[unsafe_offset=i] = (s[unsafe_offset=r * Int(stride32) + j].cast[DType.float32]() - xm[unsafe_offset=j]) * sx
        y[unsafe_offset=i] = s[unsafe_offset=r * Int(stride32) + h + j].cast[DType.float32]()


def addb_k(z: F, b: F, n32: Int32, f32: Int32):
    var i = global_idx.x
    if i < Int(n32):
        z[unsafe_offset=i] = z[unsafe_offset=i] + b[unsafe_offset=i % Int(f32)]


def topk_k(z: F, a: F, rows32: Int32, f32: Int32, k32: Int32):
    """One thread per row: bisect the largest bit pattern t (positive floats order as their int32 bits) with count(bits >= t) >= K."""
    var r = global_idx.x
    if r >= Int(rows32):
        return
    var f = Int(f32)
    var k = Int(k32) + (1 if MUT == 1 else 0)
    var zr = z.unsafe_offset(r * f)
    var ar = a.unsafe_offset(r * f)
    var lo = Int32(1)
    var hi = Int32(0x7F800000)
    var pos = 0
    for j in range(f):
        if bitcast[DType.int32, 1](zr[unsafe_offset=j]) >= 1:
            pos += 1
    if pos <= k:
        hi = lo
    while lo < hi:
        var mid = lo + (hi - lo + 1) // 2
        var c = 0
        for j in range(f):
            if bitcast[DType.int32, 1](zr[unsafe_offset=j]) >= mid:
                c += 1
        if c >= k:
            lo = mid
        else:
            hi = mid - 1
    for j in range(f):
        var v = zr[unsafe_offset=j]
        ar[unsafe_offset=j] = v if bitcast[DType.int32, 1](v) >= lo else Float32(0)


def dy_k(yh: F, y: F, dy: F, n32: Int32, sc: Float32):
    var i = global_idx.x
    if i < Int(n32):
        dy[unsafe_offset=i] = (yh[unsafe_offset=i] - y[unsafe_offset=i]) * sc


def sse_k(yh: F, y: F, so: F, rows32: Int32, d32: Int32):
    var r = global_idx.x
    if r >= Int(rows32):
        return
    var s: Float32 = 0
    for j in range(Int(d32)):
        var e = yh[unsafe_offset=r * Int(d32) + j] - y[unsafe_offset=r * Int(d32) + j]
        s += e * e
    so[unsafe_offset=r] = s


def colsum_k(x: F, o: F, rows32: Int32, cols32: Int32, sg: Float32):
    var j = global_idx.x
    if j >= Int(cols32):
        return
    var s: Float32 = 0
    for r in range(Int(rows32)):
        s += x[unsafe_offset=r * Int(cols32) + j]
    o[unsafe_offset=j] = s * sg


def mask_k(da: F, a: F, dz: F, n32: Int32):
    var i = global_idx.x
    if i < Int(n32):
        dz[unsafe_offset=i] = da[unsafe_offset=i] if a[unsafe_offset=i] > 0 else Float32(0)


def adam_k(p: F, g: F, m: F, v: F, n32: Int32, lr: Float32, b1: Float32, b2: Float32, eps: Float32, c1: Float32, c2: Float32):
    var i = global_idx.x
    if i >= Int(n32):
        return
    var gi = g[unsafe_offset=i]
    var mi = b1 * m[unsafe_offset=i] + (1 - b1) * gi
    var vi = b2 * v[unsafe_offset=i] + (1 - b2) * gi * gi
    m[unsafe_offset=i] = mi
    v[unsafe_offset=i] = vi
    p[unsafe_offset=i] = p[unsafe_offset=i] - lr * (mi * c1) / (sqrt(vi * c2) + eps)


def renorm_k(w: F, rows32: Int32, d32: Int32):
    var r = global_idx.x
    if r >= Int(rows32):
        return
    var d = Int(d32)
    var s: Float32 = 0
    for j in range(d):
        s += w[unsafe_offset=r * d + j] * w[unsafe_offset=r * d + j]
    var inv = 1 / sqrt(s)
    for j in range(d):
        w[unsafe_offset=r * d + j] = w[unsafe_offset=r * d + j] * inv


def stat_k(a: F, cnt: F, sm: F, rows32: Int32, f32: Int32):
    """Per feature: how many rows it fired on and the sum of its activations (held-out feature stats)."""
    var j = global_idx.x
    if j >= Int(f32):
        return
    var c: Float32 = 0
    var s: Float32 = 0
    for r in range(Int(rows32)):
        var v = a[unsafe_offset=r * Int(f32) + j]
        if v > 0:
            c += 1
            s += v
    cnt[unsafe_offset=j] = cnt[unsafe_offset=j] + c
    sm[unsafe_offset=j] = sm[unsafe_offset=j] + s


struct Buf(Movable):
    """Device fp32 buffer; the raw pointer `p` does not keep it alive, so hold the Buf until after the last launch."""
    var d: DeviceBuffer[DType.float32]
    var n: Int

    def __init__(out self, ctx: DeviceContext, n: Int, zero: Bool = True) raises:
        self.n = n
        self.d = ctx.enqueue_create_buffer[DType.float32](n)
        if zero:
            ctx.enqueue_memset(self.d, 0)

    def p(self) -> F:
        return F(unsafe_from_address=Int(self.d.unsafe_ptr()))

    def put(self, ctx: DeviceContext, h: List[Float32]) raises:
        ctx.enqueue_copy(dst_buf=self.d, src_ptr=h.unsafe_ptr())
        ctx.synchronize()

    def get(self, ctx: DeviceContext) raises -> List[Float32]:
        var h = List[Float32](length=self.n, fill=0)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=self.d)
        ctx.synchronize()
        return h^


struct Tc(Movable):
    var din: Int
    var dout: Int
    var nf: Int
    var k: Int
    var rows: Int
    var vtot: Float32  # per-token total variance of y (dataset), the FVU denominator
    var t: Int
    var we: Buf  # din x nf
    var be: Buf
    var wd: Buf  # nf x dout
    var bd: Buf
    var gwe: Buf
    var gbe: Buf
    var gwd: Buf
    var gbd: Buf
    var m: List[Buf]  # Adam moments, one per parameter tensor (we, be, wd, bd)
    var v: List[Buf]
    var x: Buf  # rows x din
    var y: Buf  # rows x dout
    var z: Buf
    var a: Buf
    var yh: Buf
    var dy: Buf
    var da: Buf
    var dz: Buf
    var se: Buf  # rows: per-row SSE
    var b2: Float64  # Adam beta2
    var fast: Bool  # GEMMs on fp16 WMMA (fp32 accumulate) instead of the fp32 reference GEMM
    var wm: Wmma

    def __init__(out self, ctx: DeviceContext, din: Int, dout: Int, nf: Int, k: Int, rows: Int, vtot: Float32, fast: Bool = False) raises:
        self.din = din
        self.dout = dout
        self.nf = nf
        self.k = k
        self.rows = rows
        self.vtot = vtot
        self.t = 0
        self.we = Buf(ctx, din * nf)
        self.be = Buf(ctx, nf)
        self.wd = Buf(ctx, nf * dout)
        self.bd = Buf(ctx, dout)
        self.gwe = Buf(ctx, din * nf)
        self.gbe = Buf(ctx, nf)
        self.gwd = Buf(ctx, nf * dout)
        self.gbd = Buf(ctx, dout)
        self.m = List[Buf]()
        self.v = List[Buf]()
        for n in [din * nf, nf, nf * dout, dout]:
            self.m.append(Buf(ctx, n))
            self.v.append(Buf(ctx, n))
        self.x = Buf(ctx, rows * din)
        self.y = Buf(ctx, rows * dout)
        self.z = Buf(ctx, rows * nf)
        self.a = Buf(ctx, rows * nf)
        self.yh = Buf(ctx, rows * dout)
        self.dy = Buf(ctx, rows * dout)
        self.da = Buf(ctx, rows * nf)
        self.dz = Buf(ctx, rows * nf)
        self.se = Buf(ctx, rows)
        self.fast = fast
        self.b2 = 0.999
        self.wm = Wmma(ctx, max(max(rows, din), dout) * nf if fast else 1)

    def load_xy(self, ctx: DeviceContext, src: DeviceBuffer[DType.float16], stride: Int, xm: Buf, sx: Float32) raises:
        """src holds `rows` records of `stride` f16 each: x then y (then any further field); x becomes (x - xm) * sx."""
        ctx.enqueue_function[split_k](H16(unsafe_from_address=Int(src.unsafe_ptr())), self.x.p(), self.y.p(), xm.p(), Int32(self.rows), Int32(self.din), Int32(stride), sx, grid_dim=grid(self.rows * self.din), block_dim=THREADS)

    def mm[ta: Bool, tb: Bool](self, ctx: DeviceContext, a: F, b: F, c: F, m: Int, n: Int, k: Int) raises:
        if self.fast:
            self.wm.gemm[ta, tb](ctx, a, b, c, m, n, k)
        else:
            gemm[ta, tb](ctx, a, b, c, m, n, k)

    def forward(self, ctx: DeviceContext) raises:
        var r = self.rows
        self.mm[False, False](ctx, self.x.p(), self.we.p(), self.z.p(), r, self.nf, self.din)
        ctx.enqueue_function[addb_k](self.z.p(), self.be.p(), Int32(r * self.nf), Int32(self.nf), grid_dim=grid(r * self.nf), block_dim=THREADS)
        ctx.enqueue_function[topk_k](self.z.p(), self.a.p(), Int32(r), Int32(self.nf), Int32(self.k), grid_dim=grid(r), block_dim=THREADS)
        self.mm[False, False](ctx, self.a.p(), self.wd.p(), self.yh.p(), r, self.dout, self.nf)
        ctx.enqueue_function[addb_k](self.yh.p(), self.bd.p(), Int32(r * self.dout), Int32(self.dout), grid_dim=grid(r * self.dout), block_dim=THREADS)

    def grads(self, ctx: DeviceContext) raises:
        """forward + backward: fills gwe, gbe, gwd, gbd."""
        self.forward(ctx)
        var r = self.rows
        var sc = Float32(2.0) / (Float32(r) * self.vtot)
        ctx.enqueue_function[dy_k](self.yh.p(), self.y.p(), self.dy.p(), Int32(r * self.dout), sc, grid_dim=grid(r * self.dout), block_dim=THREADS)
        self.mm[True, False](ctx, self.a.p(), self.dy.p(), self.gwd.p(), self.nf, self.dout, r)
        ctx.enqueue_function[colsum_k](self.dy.p(), self.gbd.p(), Int32(r), Int32(self.dout), Float32(-1 if MUT == 3 else 1), grid_dim=grid(self.dout), block_dim=THREADS)
        self.mm[False, True](ctx, self.dy.p(), self.wd.p(), self.da.p(), r, self.nf, self.dout)
        ctx.enqueue_function[mask_k](self.da.p(), self.a.p(), self.dz.p(), Int32(r * self.nf), grid_dim=grid(r * self.nf), block_dim=THREADS)
        self.mm[True, False](ctx, self.x.p(), self.dz.p(), self.gwe.p(), self.din, self.nf, r)
        ctx.enqueue_function[colsum_k](self.dz.p(), self.gbe.p(), Int32(r), Int32(self.nf), Float32(1), grid_dim=grid(self.nf), block_dim=THREADS)

    def update(mut self, ctx: DeviceContext, lr: Float32) raises:
        self.t += 1
        var b1 = Float32(0.9)
        var b2 = Float32(self.b2)
        var c1 = Float32(1.0 / (1.0 - 0.9 ** Float64(self.t)))
        var c2 = Float32(1.0 / (1.0 - self.b2 ** Float64(self.t)))
        for i in range(4):
            var n = self.we.n if i == 0 else (self.be.n if i == 1 else (self.wd.n if i == 2 else self.bd.n))
            var p = self.we.p() if i == 0 else (self.be.p() if i == 1 else (self.wd.p() if i == 2 else self.bd.p()))
            var g = self.gwe.p() if i == 0 else (self.gbe.p() if i == 1 else (self.gwd.p() if i == 2 else self.gbd.p()))
            ctx.enqueue_function[adam_k](p, g, self.m[i].p(), self.v[i].p(), Int32(n), lr, b1, b2, Float32(1e-8), c1, c2, grid_dim=grid(n), block_dim=THREADS)
        if MUT != 2:
            ctx.enqueue_function[renorm_k](self.wd.p(), Int32(self.nf), Int32(self.dout), grid_dim=grid(self.nf), block_dim=THREADS)

    def sse(self, ctx: DeviceContext) raises -> Float64:
        """Sum of squared errors of the last forward over all rows."""
        ctx.enqueue_function[sse_k](self.yh.p(), self.y.p(), self.se.p(), Int32(self.rows), Int32(self.dout), grid_dim=grid(self.rows), block_dim=THREADS)
        var h = self.se.get(ctx)
        var s: Float64 = 0
        for i in range(self.rows):
            s += Float64(h[i])
        return s
