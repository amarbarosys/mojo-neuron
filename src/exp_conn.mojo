"""exp_conn MODEL.gguf CORPUS LAYER [WINDOWS]: do neurons that are connected by weights co-fire? (docs/experiments.md E4)
For every 8th neuron n of LAYER, its strongest direct connection (|cos| of its write direction with a gate or up row of LAYER+1, `downstream`)
gives a target m. Signed correlation = Pearson(a[n], a'[m]) over all corpus positions, times sign(cos): positive means the target
moves the way the weights predict. Control: the same n against 20 random m of LAYER+1 (signed with the same cos sign)."""
from std.sys import argv, exit
from std.math import sqrt
from ggufx.model import Gguf
from mneuron.ggio import llama_ids
from mneuron.fwd import forward
from mneuron.conn import downstream
from mneuron.store import read_text

comptime SEG = 128


def pearson(a: List[Float32], an: Int, b: List[Float32], bn: Int, ffa: Int, ffb: Int, T: Int) -> Float64:
    var sa: Float64 = 0
    var sb: Float64 = 0
    var saa: Float64 = 0
    var sbb: Float64 = 0
    var sab: Float64 = 0
    for p in range(T):
        var x = Float64(a[p * ffa + an])
        var y = Float64(b[p * ffb + bn])
        sa += x
        sb += y
        saa += x * x
        sbb += y * y
        sab += x * y
    var t = Float64(T)
    var va = saa - sa * sa / t
    var vb = sbb - sb * sb / t
    if va <= 1e-12 or vb <= 1e-12:
        return 0.0
    return (sab - sa * sb / t) / sqrt(va * vb)


def run() raises:
    var av = argv()
    if len(av) < 4:
        raise Error("usage: exp_conn MODEL.gguf CORPUS LAYER [WINDOWS]")
    var windows = Int(String(av[4])) if len(av) > 4 else 40
    var g = Gguf(String(av[1]))
    var text = read_text(String(av[2]))
    var L = Int(String(av[3]))
    var seq = List[Int]()
    var w = 0
    var at = 0
    while w < windows and at + 700 <= text.byte_length():
        var ids = llama_ids(String(av[1]), String(text[byte=at : at + 700]), ".work/exp-text.txt")
        at += 700
        if len(ids) >= SEG:
            for j in range(SEG):
                seq.append(ids[j])
            w += 1
    var T = len(seq)
    var ls = List[Int]()
    ls.append(L)
    ls.append(L + 1)
    var r = forward(g, seq, ls, List[Int](), SEG)
    var a0 = r[0][0].copy()
    var a1 = r[0][1].copy()
    var ffn = len(a0) // T
    var real: Float64 = 0
    var chance: Float64 = 0
    var pos = 0
    var n_used = 0
    var nctl = 0
    var seed = 12345
    var cs = List[Float64]()  # |cos| and signed correlation of every connection, to look at the strongest quarter
    var rs = List[Float64]()
    for n in range(0, ffn, 8):
        var c = downstream(g, L, n, 1, 1)
        if len(c) == 0:
            continue
        var sg: Float64 = 1.0 if c[0].cos >= 0 else -1.0
        var rr = sg * pearson(a0, n, a1, c[0].neuron, ffn, ffn, T)
        real += rr
        cs.append(c[0].cos if c[0].cos >= 0 else -c[0].cos)
        rs.append(rr)
        pos += 1 if rr > 0 else 0
        n_used += 1
        for _ in range(20):
            seed = (seed * 1103515245 + 12345) % 2147483648
            chance += sg * pearson(a0, n, a1, seed % ffn, ffn, ffn, T)
            nctl += 1
    print("positions", T, "neurons", n_used, "layer", L, "->", L + 1)
    print("mean signed correlation, connected target:", real / Float64(n_used), "| fraction positive:", Float64(pos) / Float64(n_used))
    print("mean signed correlation, random target   :", chance / Float64(nctl))
    var lo: Float64 = 0
    var hi: Float64 = 1
    for _ in range(40):  # |cos| that leaves a quarter of the connections above it
        var mid = (lo + hi) / 2
        var cnt = 0
        for v in cs:
            if v > mid:
                cnt += 1
        if cnt * 4 > len(cs):
            lo = mid
        else:
            hi = mid
    var sq: Float64 = 0
    var nq = 0
    var pq = 0
    for i in range(len(cs)):
        if cs[i] > hi:
            sq += rs[i]
            nq += 1
            pq += 1 if rs[i] > 0 else 0
    print("strongest quarter (|cos| >", hi, ", n", nq, "): mean signed correlation", sq / Float64(max(nq, 1)), "| fraction positive:", Float64(pq) / Float64(max(nq, 1)))


def main():
    try:
        run()
    except e:
        print("FAIL exp_conn:", e)
        exit(1)
