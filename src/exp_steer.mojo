"""exp_steer MODEL LAYER STEER_POS.bin STEER_NEG.bin: does the logit lens of a neuron predict what steering it does? (docs/experiments.md E1)
For each neuron: hits = |lens top-k ids  ∩  tokens whose log-prob rose most at +dose| (also lens bottom vs the rise at -dose, and lens top vs the
FALL at +dose as a sanity). Control: the same count with the steer record of another neuron (20 shifts), which is what chance gives."""
from std.sys import argv, exit
from mneuron.store import Lens, Steer


def inter(l: Lens, n: Int, bottom: Bool, s: Steer, m: Int, down: Bool) -> Int:
    var c = 0
    for i in range(l.k):
        var a = l.id(n, bottom, i)
        if a < 0:
            continue
        for j in range(s.k):
            if s.id(m, down, j) == a:
                c += 1
                break
    return c


def report(name: String, l: Lens, s: Steer, bottom: Bool, down: Bool, strong: List[Bool]) -> None:
    var ffn = l.ffn
    var hits = 0
    var any = 0
    var sh = 0
    var sany = 0
    var hs = 0
    var as_ = 0
    var ns = 0
    var shifts = [1, 97, 389, 1021, 2003, 2999, 4001, 17, 211, 733, 1499, 2477, 3331, 3989, 123, 619, 1237, 1801, 2411, 3067]
    var sht = 0
    for n in range(ffn):
        var c = inter(l, n, bottom, s, n, down)
        hits += c
        any += 1 if c > 0 else 0
        if strong[n]:
            hs += c
            as_ += 1 if c > 0 else 0
            ns += 1
        for d in shifts:
            var cc = inter(l, n, bottom, s, (n + d) % ffn, down)
            sht += cc
            sany += 1 if cc > 0 else 0
    var nsh = len(shifts)
    print(name, "| mean hits", Float64(hits) / Float64(ffn), "| neurons with >=1:", Float64(any) / Float64(ffn),
          "| chance hits", Float64(sht) / Float64(ffn * nsh), "| chance >=1:", Float64(sany) / Float64(ffn * nsh),
          "| strongest 5% >=1:", Float64(as_) / Float64(max(ns, 1)))


def run() raises:
    var a = argv()
    if len(a) < 5:
        raise Error("usage: exp_steer MODEL LAYER STEER_POS STEER_NEG")
    var l = Lens(String(a[1]), Int(String(a[2])))
    var sp = Steer(String(a[3]))
    var sn = Steer(String(a[4]))
    if sp.ffn != l.ffn or sn.ffn != l.ffn:
        raise Error("ffn mismatch between lens and steer files")
    var z = List[Float32]()
    for n in range(l.ffn):
        z.append(l.z(n, False, 0))
    var lo: Float32 = 0
    var hi: Float32 = 0
    for v in z:
        hi = max(hi, v)
    for _ in range(40):  # bisect the z that leaves 5 percent above it
        var mid = (lo + hi) / 2
        var c = 0
        for v in z:
            if v > mid:
                c += 1
        if c * 20 > l.ffn:
            lo = mid
        else:
            hi = mid
    var strong = List[Bool]()
    for v in z:
        strong.append(v > hi)
    print("layer", l.layer, "ffn", l.ffn, "lens k", l.k, "steer k", sp.k, "dose +", sp.dose, "/", sn.dose)
    report("lens top    vs steer(+dose) rise", l, sp, False, False, strong)
    report("lens bottom vs steer(-dose) rise", l, sn, True, False, strong)
    report("lens top    vs steer(+dose) FALL (sanity: expect below chance)", l, sp, False, True, strong)
    report("lens bottom vs steer(+dose) FALL", l, sp, True, True, strong)


def main():
    try:
        run()
    except e:
        print("FAIL exp_steer:", e)
        exit(1)
