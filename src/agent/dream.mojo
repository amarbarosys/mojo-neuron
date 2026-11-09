"""Dream run|score (F26, docs/prereg-f26.md): model dreams on the baro engine.
run MODEL OUTDIR rho=F [layer=16 nfeat=8 nctrl=4 steps=8 tokens=16 prompt=The lens=STEM]: stdin = engine stdout, stdout = engine stdin
  (scripts/dream.sh wires the two FIFOs). Per unit (base, features = top lens z, controls = seeded random neurons): `steps` requests,
  each = prompt + dream so far, steer layer along the unit decoder column at rho*0.2*(step+1), `tokens` new tokens fed back.
  Writes OUTDIR/dream.jsonl (one line per dream) and OUTDIR/bank.f32.
score OUTDIR [layer=16 lens=STEM]: overlap of each dream with the lens top-k ids of its feature, the prereg bars, rlog lines."""
from std.sys import argv, get_defined_int
from std.ffi import external_call
from std.math import sqrt
from std.memory import bitcast
from ggufx.model import Gguf, Out
from tokenizer import Tokenizer
from mneuron.ggio import tensor, load_rows
from mneuron.store import Lens
from mneuron.rlog import Rlog

comptime MUT = get_defined_int["DREAM_MUT", 0]()


def opt(a: List[String], key: String, d: String) -> String:
    for x in a:
        if x.startswith(key + "="):
            return String(x[byte = key.byte_length() + 1 :])
    return d


def read_line(mut eof: Bool) -> String:
    var b = List[UInt8]()
    var c = List[UInt8](length=1, fill=0)
    while True:
        if external_call["read", Int](Int32(0), c.unsafe_ptr(), 1) <= 0:
            eof = True
            break
        if c[0] == 10:
            break
        b.append(c[0])
    return String(unsafe_from_utf8=Span(b))


def send(line: String):
    """One request line to fd 1 with write(2): print() stays in the stdio buffer when stdout is a FIFO and the engine never answers."""
    var s = line + "\n"  # a named String: as_bytes() of the temporary would dangle before write(2)
    _ = external_call["write", Int](Int32(1), s.unsafe_ptr(), s.byte_length())


def num_after(l: String, key: String) -> Int:
    var i = l.find(key)
    if i < 0:
        return -1
    var v = 0
    for x in l[byte = i + key.byte_length() :].as_bytes():
        if x < 48 or x > 57:
            break
        v = v * 10 + Int(x) - 48
    return v


def jesc(s: String) -> String:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")


def stem(model: String) -> String:
    var p = model.split("/")
    return String(p[len(p) - 1]).replace(".gguf", "").lower()


def generate(ids: List[Int], n: Int, layer: Int, row: Int, scale: Float64, bank: String, mut rid: Int) raises -> List[Int]:
    var pl = String()
    for i in range(len(ids)):
        pl += ("," if i > 0 else "") + String(ids[i])
    rid += 1
    var q = '{"id":' + String(rid) + ',"prompt":[' + pl + '],"n":' + String(n)
    if row >= 0:
        q += ',"steer_bank":"' + bank + '","steer_layer":' + String(layer) + ',"steer_row":' + String(row) + ',"steer_scale":' + String(scale)
    send(q + "}")
    var out = List[Int]()
    var eof = False
    while True:
        var l = read_line(eof)
        if eof:
            raise Error("engine closed its output during request " + String(rid))
        if l.startswith('{"id":' + String(rid) + ","):
            if l.find('"error"') >= 0:
                raise Error("engine: " + l)
            if l.find('"done"') >= 0:
                break
            out.append(num_after(l, '"tok":'))
    if len(out) != n:
        raise Error("request " + String(rid) + " returned " + String(len(out)) + " tokens, wanted " + String(n))
    return out^


def dream_one(base: List[Int], steps: Int, ntok: Int, layer: Int, row: Int, scale0: Float64, bank: String, mut rid: Int, mut sc: String) raises -> List[Int]:
    """`steps` requests, each prompt + dream so far, steer at scale0 * (step + 1) (row < 0: unsteered); returns the generated ids."""
    var ids = base.copy()
    var gen = List[Int]()
    for s in range(steps):
        var scale = scale0 * Float64(s + 1)
        var t = generate(ids, ntok, layer, row, scale, bank, rid)
        ids.extend(t.copy())
        gen.extend(t.copy())
        sc += (" " if s > 0 else "") + (String(scale) if row >= 0 else "0")
    return gen^


def line(kind: String, n: Int, owner: Int, layer: Int, m: Int, sc: String, gen: List[Int], tok: Tokenizer) -> String:
    var tx = String()
    for t in gen:
        tx += " " + String(t)
    return '{"kind":"' + kind + '","n":' + String(n) + ',"for":' + String(owner) + ',"m":' + String(m) + ',"layer":' + String(layer) + ',"scales":"' + sc + '","toks":"' + String(tx.strip()) + '","text":"' + jesc(tok.decode(gen)) + '"}\n'


def run(a: List[String]) raises:
    var model = a[2]
    var odir = a[3]
    var layer = Int(opt(a, "layer", "16"))
    var nfeat = Int(opt(a, "nfeat", "8"))
    var nctrl = Int(opt(a, "nctrl", "4"))
    var steps = Int(opt(a, "steps", "8"))
    var ntok = Int(opt(a, "tokens", "16"))
    var rho = Float64(opt(a, "rho", "0"))
    var mult = Float64(opt(a, "mult", "1"))  # exploratory scale multiplier; the prereg run is mult=1
    if rho <= 0:
        raise Error("rho= must be the layer residual norm (tools/rho-cap.py)")
    var lens = Lens(opt(a, "lens", stem(model)), layer)
    var tok = Tokenizer(model)
    var g = Gguf(model)
    var di = tensor(g, "blk." + String(layer) + ".ffn_down.weight")
    var ffn = g.tensors[di].dims[0]
    var h = g.tensors[di].dims[1]
    var feats = List[Int]()  # top lens z, strongest first
    for _ in range(nfeat):
        var best = -1
        for n in range(ffn):
            var seen = False
            for f in feats:
                seen = seen or f == n
            if not seen and (best < 0 or lens.z(n, False, 0) > lens.z(best, False, 0)):
                best = n
        feats.append(best)
    var units = List[Int]()  # neuron per unit, -1 = base
    var kinds = List[String]()
    var owner = List[Int]()  # the feature a unit is scored against
    units.append(-1)
    kinds.append("base")
    owner.append(-1)
    var seed: UInt64 = 0x9E3779B97F4A7C15
    for f in feats:
        units.append(f)
        kinds.append("feat")
        owner.append(f)
    for f in feats:
        for _ in range(nctrl):
            var n = f
            var ok = False
            while not ok:
                seed ^= seed << 13
                seed ^= seed >> 7
                seed ^= seed << 17
                n = Int(seed % UInt64(ffn))
                ok = True
                for u in units:
                    ok = ok and u != n
            units.append(n)
            kinds.append("ctrl")
            owner.append(f)
    var ut = String()  # row, kind, neuron, owner: the units file the engine stand-in of tests/dream.sh reads
    for ui in range(len(units)):
        ut += String(ui) + "\t" + kinds[ui] + "\t" + String(units[ui]) + "\t" + String(owner[ui]) + "\n"
    with open(odir + "/units.tsv", "w") as f:
        f.write(ut)
    var D = load_rows(g, di, 0, h)
    var bank = odir + "/bank.f32"
    var bo = Out(bank)
    for u in units:
        var s: Float64 = 0
        for j in range(h):
            s += Float64(D[j * ffn + max(u, 0)]) ** 2
        var inv = 1.0 / sqrt(s)
        for j in range(h):
            bo.put(Int(bitcast[DType.uint32](Float32(Float64(D[j * ffn + max(u, 0)]) * inv))), 4)
    bo.close()
    var eof = False
    while not read_line(eof).startswith('{"ready"'):  # engine banner until ready
        if eof:
            raise Error("engine closed before ready")
    var base = tok.encode(opt(a, "prompt", "The"))
    var rid = 0
    var js = String()
    var ms = List[Int]()  # F26c sweep multipliers; empty = the single run at mult=
    for w in opt(a, "mults", "").split(","):
        if w.byte_length() > 0:
            ms.append(Int(w))
    for ui in range(len(units)):
        for mi in range(1 if units[ui] < 0 or len(ms) == 0 else len(ms)):
            var m = 0 if units[ui] < 0 else (ms[mi] if len(ms) > 0 else 0)
            var sc = String()
            var gen = dream_one(base, steps, ntok, layer, ui if units[ui] >= 0 else -1, rho * (Float64(m) if len(ms) > 0 else mult) * 0.2, bank, rid, sc)
            js += line(kinds[ui], units[ui], owner[ui], layer, m, sc, gen, tok)
    if len(ms) > 0:  # prompt-only baseline: the lens top-8 token texts in the prompt, no steer
        for f in feats:
            var pt = String("The")
            for i in range(8):
                pt += " " + tok.decode([lens.id(f, False, i)])
            var sc = String()
            js += line("prom", f, f, layer, 0, sc, dream_one(tok.encode(pt), steps, ntok, layer, -1, 0.0, bank, rid, sc), tok)
    with open(odir + "/dream.jsonl", "w") as f:
        f.write(js)


def parse_toks(l: String) raises -> List[Int]:
    var i = l.find('"toks":"') + 8
    var j = l.find('","text"')
    var out = List[Int]()
    for w in l[byte = i : j].split(" "):
        out.append(Int(w))
    return out^


def rate(toks: List[Int], lens: Lens, n: Int) -> Float64:
    var hit = 0
    for t in toks:
        for i in range(lens.k):
            if lens.id(n, False, i) == t:
                hit += 1
                break
    return Float64(hit) / Float64(len(toks))


def score(a: List[String]) raises:
    var odir = a[2]
    var layer = Int(opt(a, "layer", "16"))
    var lens = Lens(opt(a, "lens", ""), layer)
    var lines = List[String]()
    with open(odir + "/dream.jsonl", "r") as f:
        var raw = f.read_bytes()  # the text field may hold lossy-decoded bytes
        for l in String(from_utf8_lossy=Span(raw)).split("\n"):
            if l.byte_length() > 0:
                lines.append(String(l))
    var feats = List[Int]()
    var ftoks = List[List[Int]]()
    var base = List[List[Int]]()
    var ctoks = List[List[Int]]()
    var cfor = List[Int]()
    var bad = String()
    for l in lines:
        var t = parse_toks(l)
        if len(t) != 128:
            bad = "a dream has " + String(len(t)) + " tokens, not 128 (bar 3)"
        var kind = l[byte = 9 : 13]
        if kind == "feat":
            feats.append(num_after(l, '"n":'))
            ftoks.append(t^)
        elif kind == "base":
            base.append(t^)
        else:
            ctoks.append(t^)
            cfor.append(num_after(l, '"for":'))
    var nf = len(feats)
    var fr = List[Float64]()
    var cr = List[Float64]()
    var wins = 0
    var sf: Float64 = 0
    var sc: Float64 = 0
    var sb: Float64 = 0
    var out = String("feature  rate   ctrl-mean  base\n")
    for i in range(nf):
        var tgt = feats[(i + MUT) % nf]  # the mutant scores against the next feature's lens
        var r = rate(ftoks[i], lens, tgt)
        var m: Float64 = 0
        var c = 0
        for j in range(len(ctoks)):
            if cfor[j] == feats[i]:
                var x = rate(ctoks[j], lens, tgt)
                m += x
                c += 1
                cr.append(x)
        m = m / Float64(max(c, 1))
        var b = rate(base[0], lens, tgt)
        fr.append(r)
        sf += r
        sc += m
        sb += b
        wins += 1 if r > m else 0
        out += "N" + String(feats[i]) + "  " + String(r) + "  " + String(m) + "  " + String(b) + "\n"
    sf /= Float64(nf)
    sc /= Float64(nf)
    sb /= Float64(nf)
    var b1 = sf >= max(3 * sc, 0.02)
    var b2 = wins * 4 >= 3 * nf
    var b4 = sb <= sf
    var mind = Int(opt(a, "distinct", "0"))  # prereg F26b bar 7: mean distinct token ids per feature dream, 0 = off
    var sd: Float64 = 0
    for t in ftoks:
        var seen = List[Int]()
        for x in t:
            var f = False
            for y in seen:
                f = f or y == x
            if not f:
                seen.append(x)
        sd += Float64(len(seen))
    sd /= Float64(max(nf, 1))
    var b7 = sd >= Float64(mind)
    var ok = b1 and b2 and b4 and b7 and bad == ""
    out += "mean feature " + String(sf) + " control " + String(sc) + " base " + String(sb) + " wins " + String(wins) + "/" + String(nf) + "\n"
    out += "bar1 (>=3x controls, >=0.02) " + ("ok" if b1 else "MISS") + " | bar2 (>=6/8 beat own controls) " + ("ok" if b2 else "MISS") + " | bar4 (base <= features) " + ("ok" if b4 else "MISS") + "\n"
    if mind > 0:
        out += "bar7 (mean distinct tokens " + String(sd) + " >= " + String(mind) + ") " + ("ok" if b7 else "MISS") + "\n"
    with open(odir + "/score.txt", "w") as f:
        f.write(out)
    print(out)
    var vals = fr.copy()
    vals.extend(cr.copy())
    var rl = Rlog("dream", "{\"dir\":\"" + odir + "\"}", "")
    rl.done("mean_feature=" + String(sf) + " mean_control=" + String(sc) + " wins=" + String(wins) + "/" + String(nf), odir + "/score.txt", vals, 0.0, 1.0)
    if not ok:
        rl.event("anomaly", "prereg F26 bar missed: " + ("" if b1 else "bar1 ") + ("" if b2 else "bar2 ") + ("" if b4 else "bar4 ") + ("" if b7 else "bar7 ") + bad, odir + "/score.txt")
    print("PASS dream" if ok else "FAIL dream: prereg bar missed")


def ndist(t: List[Int]) -> Int:
    var seen = List[Int]()
    for x in t:
        var f = False
        for y in seen:
            f = f or y == x
        if not f:
            seen.append(x)
    return len(seen)


def sweepscore(a: List[String]) raises:
    """Score F26c (docs/prereg-f26c.md): selection set A = features 1,3,5,7 by lens z, held-out B = 2,4,6,8; m* from A with distinct >= 40, bars on B."""
    var odir = a[2]
    var lens = Lens(opt(a, "lens", ""), Int(opt(a, "layer", "16")))
    var raw = open(odir + "/dream.jsonl", "r").read_bytes()
    var feats = List[Int]()
    var ms = List[Int]()
    var k = List[String]()  # kind
    var nn = List[Int]()
    var ow = List[Int]()
    var mm = List[Int]()
    var tk = List[List[Int]]()
    var bad = String()
    for l in String(from_utf8_lossy=Span(raw)).split("\n"):
        if l.byte_length() == 0:
            continue
        var t = parse_toks(String(l))
        if len(t) != 128:
            bad = "a dream has " + String(len(t)) + " tokens, not 128 (bar 4)"
        var kd = String(l[byte = 9 : 13])
        var n = num_after(String(l), '"n":')
        var m = num_after(String(l), '"m":')
        k.append(kd)
        nn.append(n)
        ow.append(num_after(String(l), '"for":'))
        mm.append(m)
        tk.append(t^)
        if kd == "feat":
            var seen = False
            for f in feats:
                seen = seen or f == n
            if not seen:
                feats.append(n)
            var sm = False
            for x in ms:
                sm = sm or x == m
            if not sm:
                ms.append(m)
    var nf = len(feats)
    var out = String("m   setA(gap dist)   setB(feat ctrl wins dist)   prompt-only\n")
    var best = -1
    var bgap: Float64 = -1e9
    var bm = List[Float64]()  # B stats per m: feat rate, ctrl rate, wins, distinct
    for mi in range(len(ms)):
        var ga: Float64 = 0
        var da: Float64 = 0
        var fb: Float64 = 0
        var cb: Float64 = 0
        var wb = 0
        var db: Float64 = 0
        for fi in range(nf):
            var tgt = feats[(fi + MUT) % nf]
            var fr: Float64 = 0
            var cm: Float64 = 0
            var nc = 0
            var di = 0
            for j in range(len(k)):
                if mm[j] != ms[mi] or ow[j] != feats[fi]:
                    continue
                if k[j] == "feat":
                    fr = rate(tk[j], lens, tgt)
                    di = ndist(tk[j])
                elif k[j] == "ctrl":
                    cm += rate(tk[j], lens, tgt)
                    nc += 1
            cm /= Float64(max(nc, 1))
            if fi % 2 == 0:
                ga += (fr - cm) / Float64(max(nf // 2, 1))
                da += Float64(di) / Float64(max(nf // 2, 1))
            else:
                fb += fr / Float64(max(nf // 2, 1))
                cb += cm / Float64(max(nf // 2, 1))
                wb += 1 if fr > cm else 0
                db += Float64(di) / Float64(max(nf // 2, 1))
        bm.append(fb)
        bm.append(cb)
        bm.append(Float64(wb))
        bm.append(db)
        out += String(ms[mi]) + "   " + String(ga) + " " + String(da) + "   " + String(fb) + " " + String(cb) + " " + String(wb) + "/" + String(nf // 2) + " " + String(db) + "\n"
        if da >= 40 and ga > bgap:
            bgap = ga
            best = mi
    var pr: Float64 = 0
    var np = 0
    for j in range(len(k)):
        if k[j] == "prom":
            pr += rate(tk[j], lens, ow[j])
            np += 1
    out += "prompt-only mean rate " + String(pr / Float64(max(np, 1))) + "\n"
    var b1 = best >= 0
    var b2 = False
    var b3 = False
    if b1:
        var fb = bm[best * 4]
        var cb = bm[best * 4 + 1]
        b2 = fb >= max(3 * cb, 0.02) and Int(bm[best * 4 + 2]) * 4 >= 3 * (nf // 2)
        b3 = bm[best * 4 + 3] >= 40
        out += "m* = " + String(ms[best]) + " (A gap " + String(bgap) + ")\n"
    var ok = b1 and b2 and b3 and bad == ""
    out += "bar1 (m* exists) " + ("ok" if b1 else "MISS") + " | bar2 (B rate, 3/4 wins) " + ("ok" if b2 else "MISS") + " | bar3 (B distinct >= 40) " + ("ok" if b3 else "MISS") + "\n"
    with open(odir + "/score.txt", "w") as f:
        f.write(out)
    print(out)
    var vals = List[Float64]()
    for v in bm:
        vals.append(v)
    var rl = Rlog("dream-sweep", "{\"dir\":\"" + odir + "\"}", "")
    rl.done("m*=" + String(ms[best] if b1 else -1), odir + "/score.txt", vals, 0.0, 200.0)
    if not ok:
        rl.event("anomaly", "prereg F26c bar missed " + ("" if b1 else "bar1 ") + ("" if b2 else "bar2 ") + ("" if b3 else "bar3 ") + bad, odir + "/score.txt")
    print("PASS sweep" if ok else "FAIL sweep: prereg bar missed")


def main() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    if len(a) < 3:
        raise Error("usage: dream run MODEL OUTDIR rho=F [...] | dream score OUTDIR lens=STEM")
    if a[1] == "run":
        run(a)
    elif a[1] == "score":
        score(a)
    elif a[1] == "sweepscore":
        sweepscore(a)
    else:
        raise Error("dream: run or score")
