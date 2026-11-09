"""tcsteer: the B2 gate of docs/prereg-b2.md (transcoder features steered through the engine, scored by lens-token overlap).
  tcsteer pick  TC_DIR NF H N OUT           N features by held-out activation sum (not always-on); OUT/pick.txt, OUT/mini/tc.bin (decoder rows only, for `lens --tc OUT/mini N`)
  tcsteer req   TC_DIR NF H LAYER CAP OUT   OUT/req.jsonl (engine requests, BARO_SERVE=1 BARO_STEER_SERVE=1) + OUT/v_<f>_<dose>.f32, doses 2,8,32 x mean |y|
  tcsteer score LENS.bin RESP OUT           reads engine stdout RESP; prints diag/off/base per dose and PASS or BAR MISSED
"""
from std.sys import argv, exit
from std.math import sqrt
from json import parse
from mneuron.ggio import write_file, put_f32, put_u32
from mneuron.store import read_text
from mneuron.rlog import Rlog

comptime P = 6
comptime NDOSE = 3
comptime TOPK = 8


def doses() -> List[Float64]:
    return [2.0, 8.0, 32.0]


def feats_of(path: String) raises -> List[Int]:
    var r = List[Int]()
    for t in read_text(path).split(" "):
        var s = String(String(t).strip())
        if s.byte_length() > 0:
            r.append(Int(s))
    return r^


def f32_at(b: List[UInt8], i: Int) -> Float32:
    return b.unsafe_ptr().unsafe_bitcast[Float32]()[i]


def main() raises:
    var a = argv()
    var mode = String(a[1])
    if mode == "pick":
        var tcd = String(a[2])
        var nf = Int(String(a[3]))
        var h = Int(String(a[4]))
        var n = Int(String(a[5]))
        var out = String(a[6])
        var fb = open(tcd + "/feat.f32", "r").read_bytes()
        var mean_cnt: Float64 = 0
        for f in range(nf):
            mean_cnt += Float64(f32_at(fb, f)) / Float64(nf)
        var chosen = List[Int]()
        for _ in range(n):
            var best = -1
            var bs = Float32(-1)
            for f in range(nf):
                var c = f32_at(fb, f)
                var s = f32_at(fb, nf + f)
                var taken = False
                for q in chosen:
                    if q == f:
                        taken = True
                if not taken and c > 0 and Float64(c) <= 25 * mean_cnt and s > bs:
                    bs = s
                    best = f
            if best < 0:
                raise Error("no candidate feature")
            chosen.append(best)
        var tb = open(tcd + "/tc.bin", "r").read_bytes()
        var mini = List[UInt8]()
        for _ in range(h * n + n):  # We, be: zeros (the lens reads decoder rows only)
            put_f32(mini, 0)
        for f in chosen:
            for j in range(h):
                put_f32(mini, f32_at(tb, h * nf + nf + f * h + j))
        for _ in range(h):
            put_f32(mini, 0)
        var txt = String()
        for f in chosen:
            txt += String(f) + " "
        var tb2 = List[UInt8]()
        for c in txt.as_bytes():
            tb2.append(c)
        write_file(out + "/pick.txt", tb2)
        write_file(out + "/mini/tc.bin", mini)
        print("picked", txt)
    elif mode == "req":
        var tcd = String(a[2])
        var nf = Int(String(a[3]))
        var h = Int(String(a[4]))
        var layer = Int(String(a[5]))
        var cap = String(a[6])
        var out = String(a[7])
        var fs = feats_of(out + "/pick.txt")
        var tb = open(tcd + "/tc.bin", "r").read_bytes()
        var rec = open(cap + "/L" + String(layer) + ".f16", "r").read_bytes()
        var tok = open(cap + "/tokens.i32", "r").read_bytes()
        var ybar: Float64 = 0
        var nrec = 2000
        for r in range(nrec):
            var q: Float64 = 0
            for j in range(h):
                var v = Float64(rec.unsafe_ptr().unsafe_bitcast[Float16]()[r * 2 * h + h + j])
                q += v * v
            ybar += sqrt(q) / Float64(nrec)
        var s = String()
        var id = 1
        for pi in range(P):  # unsteered rows first: id 1..P
            var ps = String()
            for j in range(32):
                ps += String(tok.unsafe_ptr().unsafe_bitcast[Int32]()[pi * 7919 % 90000 + j]) + ("," if j < 31 else "")
            s += '{"id":' + String(id) + ',"prompt":[' + ps + '],"n":1,"spec":false,"logits_topk":20}\n'
            id += 1
        for fi in range(len(fs)):
            var nrm: Float64 = 0
            for j in range(h):
                nrm += Float64(f32_at(tb, h * nf + nf + fs[fi] * h + j)) ** 2
            nrm = sqrt(nrm)
            for di in range(NDOSE):
                var vb = List[UInt8]()
                for j in range(h):
                    put_f32(vb, Float32(Float64(f32_at(tb, h * nf + nf + fs[fi] * h + j)) / nrm * doses()[di] * ybar))
                var vp = out + "/v_" + String(fs[fi]) + "_" + String(di) + ".f32"
                write_file(vp, vb)
                for pi in range(P):
                    var ps = String()
                    for j in range(32):
                        ps += String(tok.unsafe_ptr().unsafe_bitcast[Int32]()[pi * 7919 % 90000 + j]) + ("," if j < 31 else "")
                    s += '{"id":' + String(id) + ',"prompt":[' + ps + '],"n":1,"spec":false,"logits_topk":20,"steer_file":"' + vp + '","steer_layer":' + String(layer) + "}\n"
                    id += 1
        var sb = List[UInt8]()
        for c in s.as_bytes():
            sb.append(c)
        write_file(out + "/req.jsonl", sb)
        print("requests", id - 1, "mean |y|", ybar)
    elif mode == "score":
        var lb = open(String(a[2]), "r").read_bytes()
        var out = String(a[4])
        var fs = feats_of(out + "/pick.txt")
        var k = Int(lb.unsafe_ptr().unsafe_bitcast[Int32]()[4])
        var nfe = len(fs)
        var lens = List[List[Int]]()
        for i in range(nfe):
            var ids = List[Int]()
            for q in range(TOPK):
                ids.append(Int(lb.unsafe_ptr().unsafe_bitcast[Int32]()[5 + i * 4 * k + q]))
            lens.append(ids^)
        var top = List[List[Int]]()  # per request id (1-based) the top-20 ids
        for _ in range(P * (1 + nfe * NDOSE) + 1):
            top.append(List[Int]())
        for line in read_text(String(a[3])).split("\n"):
            if String(line).startswith('{"id":') and String(line).find('"logits_topk"') >= 0:
                var j = parse(String(line))
                var ids = List[Int]()
                var arr = j.get("logits_topk")
                for q in range(arr.len()):
                    ids.append(arr.at(q).get("id").int())
                top[j.get("id").int()] = ids^
        def ov(l: List[Int], t: List[Int]) -> Float64:
            var c = 0
            for x in l:
                for y in t:
                    if x == y:
                        c += 1
            return Float64(c)
        var base: Float64 = 0
        for i in range(nfe):
            for pi in range(P):
                base += ov(lens[i], top[1 + pi]) / Float64(nfe * P)
        var best_ok = False
        var summary = String()
        for di in range(NDOSE):
            var dg: Float64 = 0
            var of: Float64 = 0
            for i in range(nfe):
                for j in range(nfe):
                    var m: Float64 = 0
                    for pi in range(P):
                        m += ov(lens[i], top[1 + P + ((j * NDOSE + di) * P) + pi]) / Float64(P)
                    if i == j:
                        dg += m / Float64(nfe)
                    else:
                        of += m / Float64(nfe * (nfe - 1))
            print("dose", doses()[di], "x|y|  diag", dg, " off", of, " base", base)
            summary += " dose" + String(doses()[di]) + ":diag=" + String(dg) + ",off=" + String(of)
            if dg >= 2 * of and dg - base >= 0.5:
                best_ok = True
        var rl = Rlog("tcsteer", "score " + String(a[2]), "")
        var vals = List[Float64]()
        vals.append(base)
        rl.done("base=" + String(base) + summary, out + "/pick.txt", vals, 0.0, 8.0)
        if not best_ok:
            rl.event("anomaly", "prereg-b2 bar missed:" + summary + " base=" + String(base), out + "/pick.txt")
        print("PASS tcsteer: feature steer moves its lens tokens" if best_ok else "BAR MISSED: no dose with diag >= 2 x off and diag - base >= 0.5")
