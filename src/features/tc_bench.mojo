"""tc_bench B DIN NF K: milliseconds per stage of one transcoder step on random data (each stage synchronised)."""
from std.sys import argv
from std.time import perf_counter_ns
from std.random import random_float64
from max.gpu.host import DeviceContext
from ops import F, THREADS, grid, gemm
from tc import *


def ms(t0: Int) -> Float64:
    return Float64(Int(perf_counter_ns()) - t0) / 1e6


def main() raises:
    var av = argv()
    var B = Int(String(av[1]))
    var d = Int(String(av[2]))
    var nf = Int(String(av[3]))
    var k = Int(String(av[4]))
    var ctx = DeviceContext()
    var tc = Tc(ctx, d, d, nf, k, B, 1.0)
    var h = List[Float32](length=max(B * d, d * nf), fill=0)
    for i in range(len(h)):
        h[i] = Float32(random_float64(-0.1, 0.1))
    tc.x.put(ctx, List[Float32](h[0 : B * d]))
    tc.y.put(ctx, List[Float32](h[0 : B * d]))
    tc.we.put(ctx, List[Float32](h[0 : d * nf]))
    tc.wd.put(ctx, List[Float32](h[0 : d * nf]))
    for rep in range(2):
        var t = Int(perf_counter_ns())
        gemm[False, False](ctx, tc.x.p(), tc.we.p(), tc.z.p(), B, nf, d)
        ctx.synchronize()
        var g1 = ms(t)
        t = Int(perf_counter_ns())
        ctx.enqueue_function[topk_k](tc.z.p(), tc.a.p(), Int32(B), Int32(nf), Int32(k), grid_dim=grid(B), block_dim=THREADS)
        ctx.synchronize()
        var tk = ms(t)
        t = Int(perf_counter_ns())
        gemm[True, False](ctx, tc.x.p(), tc.dz.p(), tc.gwe.p(), d, nf, B)
        ctx.synchronize()
        var g2 = ms(t)
        t = Int(perf_counter_ns())
        gemm[False, True](ctx, tc.dy.p(), tc.wd.p(), tc.da.p(), B, nf, d)
        ctx.synchronize()
        var g3 = ms(t)
        t = Int(perf_counter_ns())
        tc.update(ctx, 1e-4)
        ctx.synchronize()
        var up = ms(t)
        print("gemm_fwd", g1, "topk", tk, "gemm_ta(dW)", g2, "gemm_tb(da)", g3, "update", up)
