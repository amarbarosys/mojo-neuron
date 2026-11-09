"""exp_next MODEL.gguf CORPUS LENSMODEL LAYER [WINDOWS] [KIND]: does a neuron fire when the token it promotes comes next? (docs/experiments.md E2)
CPU forward over corpus windows gives a[p][n] of the layer. For each neuron, the Q=1 percent of positions with the largest positive
activation: how often is the NEXT token one of the neuron's lens top-k (its promoted tokens)? The most negative positions use the lens
bottom (a negative activation flips the write direction). Also the CURRENT token against lens top (an input-side check).
Control: the same positions scored with the lens of another neuron (10 shifts)."""
from std.sys import argv, exit
from ggufx.model import Gguf
from mneuron.ggio import llama_ids
from mneuron.fwd import forward
from mneuron.store import Lens, read_text

comptime SEG = 128


def member(l: Lens, n: Int, bottom: Bool, id: Int) -> Bool:
    for i in range(l.k):
        if l.id(n, bottom, i) == id:
            return True
    return False


def run() raises:
    var av = argv()
    if len(av) < 5:
        raise Error("usage: exp_next MODEL.gguf CORPUS LENSMODEL LAYER [WINDOWS]")
    var windows = Int(String(av[5])) if len(av) > 5 else 40
    var g = Gguf(String(av[1]))
    var text = read_text(String(av[2]))
    var layer = Int(String(av[4]))
    var l = Lens(String(av[3]), layer, String(av[6]) if len(av) > 6 else String(""))  # KIND r: the R-style lens file
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
    ls.append(layer)
    var r = forward(g, seq, ls, List[Int](), SEG)
    ref acts = r[0][0]
    var ffn = l.ffn
    var valid = List[Int]()  # positions that have a next token inside their window
    for p in range(T):
        if p % SEG != SEG - 1:
            valid.append(p)
    var Q = max(1, len(valid) // 100)
    print("positions", T, "valid", len(valid), "Q per neuron", Q, "neurons", ffn, "lens k", l.k)
    var shifts = [1, 97, 389, 1021, 2003, 2999, 4001, 17, 211, 733]
    # counts: [next-in-top @ pos firings, next-in-bottom @ neg firings, current-in-top @ pos firings]
    var real = List[Int](length=3, fill=0)
    var ctrl = List[Int](length=3, fill=0)
    var firings = 0
    for n in range(ffn):
        for sign in range(2):  # 0: largest positive, 1: most negative
            var bv = List[Float32]()
            var bp = List[Int]()
            for p in valid:
                var v = acts[p * ffn + n] * Float32(1.0 if sign == 0 else -1.0)
                var j = 0
                if len(bv) < Q:  # keep bv ascending: the smallest kept value sits in slot 0
                    bv.append(v)
                    bp.append(p)
                    j = len(bv) - 1
                    while j > 0 and bv[j - 1] > bv[j]:
                        bv.swap_elements(j - 1, j)
                        bp.swap_elements(j - 1, j)
                        j -= 1
                elif v > bv[0]:
                    bv[0] = v
                    bp[0] = p
                    while j + 1 < len(bv) and bv[j + 1] < bv[j]:
                        bv.swap_elements(j, j + 1)
                        bp.swap_elements(j, j + 1)
                        j += 1
            if n == 0 or n == ffn - 1:  # self-check of the top-Q selection: nothing outside it beats its smallest member
                for p in valid:
                    var v = acts[p * ffn + n] * Float32(1.0 if sign == 0 else -1.0)
                    var inside = False
                    for q in bp:
                        if q == p:
                            inside = True
                    if not inside and v > bv[0]:
                        raise Error("top-Q selection wrong for neuron " + String(n) + " sign " + String(sign))
            for q in range(len(bp)):
                if bv[q] <= 0:
                    continue
                var p = bp[q]
                firings += 1
                var nxt = seq[p + 1]
                var cur = seq[p]
                if sign == 0:
                    real[0] += 1 if member(l, n, False, nxt) else 0
                    real[2] += 1 if member(l, n, False, cur) else 0
                else:
                    real[1] += 1 if member(l, n, True, nxt) else 0
                for d in shifts:
                    var m = (n + d) % ffn
                    if sign == 0:
                        ctrl[0] += 1 if member(l, m, False, nxt) else 0
                        ctrl[2] += 1 if member(l, m, False, cur) else 0
                    else:
                        ctrl[1] += 1 if member(l, m, True, nxt) else 0
    var nsh = Float64(len(shifts))
    var names = ["next token in lens TOP at positive firings", "next token in lens BOTTOM at negative firings", "current token in lens TOP at positive firings"]
    print("firing (neuron, position) pairs", firings)
    for i in range(3):
        var c = Float64(ctrl[i]) / nsh
        print(names[i], "| hits", real[i], "| chance (other neurons' lens)", c, "| lift", Float64(real[i]) / max(c, 0.5))


def main():
    try:
        run()
    except e:
        print("FAIL exp_next:", e)
        exit(1)
