"""Lie_fit CAP LABELS OUT: fit the F24 logistic probe on every captured layer (docs/prereg-f24.md).
CAP = capture dir of one request per statement in LABELS order; LABELS = `label<TAB>pair<TAB>text` per line (label 1 = true).
Split by pair: 60% train, 20% validation (picks the layer), 20% test. Writes OUT/L<l>.probe and OUT/fit.json, ends with an Rlog line."""
from std.collections import Dict
from std.sys import argv, exit
from json import Json, parse
from mneuron.rlog import Rlog
from mneuron.store import read_text
from probe.lie import last_recs, f16_rows, fit, accuracy, save_probe, Probe, part, pick

comptime LAM: Float32 = 1e-2
comptime LR: Float32 = 0.5
comptime ITERS = 300


def pairwise(p: Probe, x: List[Float32], pairs: List[Int], y: List[Int], rows: List[Int], h: Int) -> Float64:
    """Fraction of pairs among `rows` whose true statement scores above its false twin."""
    var ok = 0
    var n = 0
    for a in rows:
        if y[a] != 1:
            continue
        for b in rows:
            if y[b] == 0 and pairs[b] == pairs[a]:
                n += 1
                if p.score(x, a, h) > p.score(x, b, h):
                    ok += 1
    return Float64(ok) / Float64(max(n, 1))


def main() raises:
    var a = argv()
    if len(a) < 4:
        print("usage: lie_fit CAP LABELS OUT")
        exit(1)
    var cap = String(a[1])
    var out = String(a[3])
    var meta = parse(read_text(cap + "/meta.json"))
    var h = meta.get("h").int()
    var nf = meta.text("fields").byte_length()
    var field = meta.text("fields").find("r")
    if field < 0:
        raise Error("lie_fit: capture has no r field (BARO_CAPTURE_FIELDS)")
    var layers = List[Int]()
    for l in meta.get("layers").items():
        layers.append(l.int())
    var y = List[Int]()
    var pairs = List[Int]()
    for line in read_text(String(a[2])).split("\n"):
        if line.byte_length() == 0:
            continue
        var f = line.split("\t")
        y.append(Int(f[0]))
        pairs.append(Int(f[1]))
    var recs = last_recs(cap)
    if len(recs) != len(y):
        raise Error("lie_fit: " + String(len(recs)) + " requests in the capture, " + String(len(y)) + " labels")
    var parts = List[Int](length=len(y), fill=0)
    for i in range(len(y)):
        parts[i] = part(pairs[i])
    var tr = pick(parts, 0)
    var va = pick(parts, 1)
    var te = pick(parts, 2)
    if len(a) > 5 and String(a[4]) == "posthoc-pair":  # annotation only (written after the result): flip train labels per PAIR, 5 seeds, at layer a[5]
        var lay = Int(String(a[5]))
        var xp = f16_rows(cap + "/L" + String(lay) + ".f16", h, nf, field, recs)
        var res = Json.object()
        var accs = Json.array()
        for seed in range(5):
            var ys = y.copy()
            var s = UInt64(seed * 2 + 101) * 6364136223846793005
            var flip = Dict[Int, Bool]()
            for i in tr:
                if pairs[i] not in flip:
                    s = s * 6364136223846793005 + 1442695040888963407
                    flip[pairs[i]] = ((s >> 40) & 1) == 1
                if flip[pairs[i]]:
                    ys[i] = 1 - ys[i]
            accs.push(Json.of(accuracy(fit(xp, ys, tr, h, LAM, LR, ITERS), xp, y, te, h)))
        res.set("layer", Json.of(lay))
        res.set("pair_shuffle_test", accs)
        with open(out + "/control-pair.json", "w") as f:
            f.write(res.dumps(False))
        print(res.dumps(False))
        return
    var rl = Rlog("lie_fit", String(a[1]) + " " + String(a[2]), "")
    var res = Json.object()
    var per = Json.array()
    var best = -1
    var best_va = -1.0
    var vals = List[Float64]()
    for li in range(len(layers)):
        var x = f16_rows(cap + "/L" + String(layers[li]) + ".f16", h, nf, field, recs)
        var p = fit(x, y, tr, h, LAM, LR, ITERS)
        var e = Json.object()
        e.set("layer", Json.of(layers[li]))
        e.set("train", Json.of(accuracy(p, x, y, tr, h)))
        var vacc = accuracy(p, x, y, va, h)
        e.set("val", Json.of(vacc))
        e.set("test", Json.of(accuracy(p, x, y, te, h)))
        e.set("pairwise_test", Json.of(pairwise(p, x, pairs, y, te, h)))
        per.push(e)
        vals.append(vacc)
        save_probe(out + "/L" + String(layers[li]) + ".probe", p)
        print("layer", layers[li], "val", vacc)
        if vacc > best_va:
            best_va = vacc
            best = li
    res.set("layers", per)
    res.set("best_layer", Json.of(layers[best]))
    res.set("n_train", Json.of(len(tr)))
    res.set("n_val", Json.of(len(va)))
    res.set("n_test", Json.of(len(te)))
    # shuffled-label control at the best layer: permute the train labels (LCG), same procedure, test accuracy, 5 seeds
    var xb = f16_rows(cap + "/L" + String(layers[best]) + ".f16", h, nf, field, recs)
    var pb = fit(xb, y, tr, h, LAM, LR, ITERS)
    var real = accuracy(pb, xb, y, te, h)
    var ctl = Json.array()
    var csum = 0.0
    var cmin = 1.0
    var cmax = 0.0
    for seed in range(5):
        var ys = y.copy()
        var s = UInt64(seed * 2 + 1) * 6364136223846793005
        for i in range(len(tr) - 1, 0, -1):  # Fisher-Yates over the train rows' labels
            s = s * 6364136223846793005 + 1442695040888963407
            var j = Int((s >> 33) % UInt64(i + 1))
            var t = ys[tr[i]]
            ys[tr[i]] = ys[tr[j]]
            ys[tr[j]] = t
        var pc = fit(xb, ys, tr, h, LAM, LR, ITERS)
        var ca = accuracy(pc, xb, y, te, h)
        ctl.push(Json.of(ca))
        csum += ca
        cmin = min(cmin, ca)
        cmax = max(cmax, ca)
    res.set("real_test", Json.of(real))
    res.set("control_test", ctl)
    with open(out + "/fit.json", "w") as f:
        f.write(res.dumps(False))
    var bar1 = real >= 0.80
    var bar2 = cmin >= 0.42 and cmax <= 0.58 and real - csum / 5.0 >= 0.20
    var msg = "best L" + String(layers[best]) + " test " + String(real) + " control mean " + String(csum / 5.0) + " [" + String(cmin) + "," + String(cmax) + "] bar1 " + String(bar1) + " bar2 " + String(bar2)
    print(msg)
    if bar1 and bar2:
        rl.done(msg, "", vals, 0.0, 1.0)
    else:
        rl.event("anomaly", "prereg F24 bar missed: " + msg)
        exit(2)
