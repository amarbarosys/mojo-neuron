"""Check MODEL ORACLE [FULL.f32]: CPU forward against llama.cpp's ffn_swiglu (tools/swiglu_oracle.py). Edge check: the first and last
three elements of every token at every layer (abs 0.01 + 5 percent, llama.cpp quantizes activations); a mutant with the tokens
reversed must disagree on most of them. Full check (FULL.f32 from tools/swiglu_full.py, an independent numpy f32 forward that is
itself anchored to llama.cpp's edge values): EVERY activation of every layer, tolerance fixed before the first run at abs 2e-3 +
1e-2 relative, at most 0.01 percent of values outside. Mutants that the edge check cannot see must fail it: a token/neuron
transpose, a block of interior neurons zeroed, a one-neuron shift."""
from std.sys import argv, exit
from ggufx.model import Gguf
from std.pathlib import Path
from mneuron.fwd import fires


def close(a: Float32, b: Float64) -> Bool:
    var d = Float64(a) - b
    d = d if d >= 0 else -d
    var m = b if b >= 0 else -b
    return d <= 0.01 + 0.05 * m


def bad_count(got: List[List[Float32]], layers: List[Int], want: List[List[Float64]], T: Int, ffn: Int) -> Int:
    """Count mismatches; want[i] holds 6 values of (layers[i // T], token i % T)."""
    var bad = 0
    for i in range(len(want)):
        ref a = got[i // T]
        var t = i % T
        for j in range(6):
            var idx = t * ffn + (j if j < 3 else ffn - 6 + j)
            if not close(a[idx], want[i][j]):
                bad += 1
    return bad


def full_bad(got: List[List[Float32]], want: List[Float32], nl: Int, T: Int, ffn: Int, mode: Int) -> Int:
    """Count all T x ffn x layers values outside tolerance. mode 0: as is; 1: got read transposed (stride mutant); 2: neurons 100..199
    zeroed; 3: neuron index shifted by one."""
    var bad = 0
    for l in range(nl):
        for t in range(T):
            for n in range(ffn):
                var gi = t * ffn + n
                if mode == 1:
                    gi = (n * T + t) % (T * ffn)
                elif mode == 3:
                    gi = t * ffn + (n + 1) % ffn
                var g = got[l][gi]
                if mode == 2 and n >= 100 and n < 200:
                    g = 0
                var d = Float64(g) - Float64(want[(l * T + t) * ffn + n])
                d = d if d >= 0 else -d
                var m = Float64(want[(l * T + t) * ffn + n])
                m = m if m >= 0 else -m
                if d > 0.002 + 0.01 * m:
                    bad += 1
    return bad


def run() raises:
    var av = argv()
    if len(av) < 3:
        raise Error("usage: fires_check MODEL ORACLE")
    var g = Gguf(String(av[1]))
    var ids = List[Int]()
    var layers = List[Int]()
    var want = List[List[Float64]]()
    with open(String(av[2]), "r") as f:
        for line in f.read().split("\n"):
            var w = String(line).split(" ")
            if len(w) == 0 or String(line).byte_length() == 0:
                continue
            if String(w[0]) == "ids":
                for p in String(w[1]).split(","):
                    ids.append(Int(String(p)))
            elif String(w[1]) != "sum":
                var L = Int(String(w[0])[byte=1:])
                if len(layers) == 0 or layers[len(layers) - 1] != L:
                    layers.append(L)
                var vals = List[Float64]()
                for j in range(2, 8):
                    vals.append(Float64(String(w[j])))
                want.append(vals^)
    var T = len(ids)
    if len(want) != T * len(layers):
        raise Error("oracle has " + String(len(want)) + " rows, expected " + String(T * len(layers)))
    var got = fires(g, ids, layers)
    var ffn = len(got[0]) // T
    var bad = bad_count(got, layers, want, T, ffn)
    var total = len(want) * 6
    print("layers", len(layers), "tokens", T, "values", total, "outside tolerance", bad)
    if bad * 100 > total * 3:
        raise Error(String(bad) + " of " + String(total) + " values differ from llama.cpp")
    var rev = List[Int]()
    for i in range(T - 1, -1, -1):
        rev.append(ids[i])
    var mut = fires(g, rev, layers)
    var mbad = bad_count(mut, layers, want, T, ffn)
    print("mutant (reversed tokens) outside tolerance", mbad)
    if mbad * 100 < total * 20:
        raise Error("mutant passed: only " + String(mbad) + " of " + String(total) + " values changed")
    if len(av) > 3:
        var raw = Path(String(av[3])).read_bytes()
        var nl = len(layers)
        if len(raw) != nl * T * ffn * 4:
            raise Error("full oracle holds " + String(len(raw)) + " bytes, expected " + String(nl * T * ffn * 4))
        var want = List[Float32](length=nl * T * ffn, fill=0)
        var fp = raw.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(nl * T * ffn):
            want[i] = fp[unsafe_offset=i]
        var n = nl * T * ffn
        var fb = full_bad(got, want, nl, T, ffn, 0)
        print("full check:", n, "values, outside tolerance", fb)
        if fb * 10000 > n:
            raise Error(String(fb) + " of " + String(n) + " activations differ from the full numpy oracle")
        var names = ["", "transpose", "interior neurons zeroed", "neuron shift"]
        for m in range(1, 4):
            var mb = full_bad(got, want, nl, T, ffn, m)
            print("full mutant", names[m], "outside tolerance", mb)
            if mb * 100 < n:  # must be seen by the full check (an edge-only check misses 2 of the 3)
                raise Error("full-check mutant passed: " + names[m] + " (" + String(mb) + " of " + String(n) + ")")
    print("PASS fires check:", len(layers), "layers x", T, "tokens match llama.cpp ffn_swiglu, mutant differs")


def main():
    try:
        run()
    except e:
        print("FAIL fires_check:", e)
        exit(1)
