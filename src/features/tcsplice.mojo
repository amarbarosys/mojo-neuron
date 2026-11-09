"""tcsplice: the B1 splice gate of docs/prereg-b1-splice.md (transcoder spliced into the engine, KL against the unspliced run).
  tcsplice req   TC NF K CAP LAYER OUT [NREQ=4]   rows files + OUT/req.jsonl for the last NREQ whole requests of the capture, positions 32, 64, .. 480
  tcsplice score RESP                             mean KL and NLL per arm (tc, ident, zero), ratio tc/zero, verdict against the bar 0.25
Arms: tc = transcoder output of the capture's x rows, ident = the capture's true y (floor), zero = splice_zero. Request ids: 1 000 000 * arm + 1000 * r + position (arm 4 = the unspliced base the engine needs first)."""
from std.sys import argv, exit
from json import parse
from mneuron.ggio import write_file, put_f32
from mneuron.store import read_text
from mneuron.fwd import matmul
from mneuron.rlog import Rlog
from tcenc import Enc


def main() raises:
    var a = argv()
    if String(a[1]) == "req":
        var tc = String(a[2])
        var nf = Int(String(a[3]))
        var k = Int(String(a[4]))
        var cap = String(a[5])
        var layer = String(a[6])
        var out = String(a[7])
        var nreq = Int(String(a[8])) if len(a) > 8 else 4
        var tokb = open(cap + "/tokens.i32", "r").read_bytes()
        var reqb = open(cap + "/req.i32", "r").read_bytes()
        var tok = tokb.unsafe_ptr().unsafe_bitcast[Int32]()
        var rq = reqb.unsafe_ptr().unsafe_bitcast[Int32]()
        var n = len(tokb) // 4
        var h = 4096
        var enc = Enc(tc, h, nf, k)
        var wdt = List[Float32](length=h * nf, fill=0)
        for f in range(nf):
            for j in range(h):
                wdt[j * nf + f] = enc.wd[f * h + j]
        var bdb = open(tc + "/tc.bin", "r").read_bytes()
        var bdp = bdb.unsafe_ptr().unsafe_bitcast[Float32]()
        var s = String()
        var last = Int(rq[n - 1])
        for r in range(nreq):
            var rid = last - nreq + 1 + r
            var s0 = 0
            while Int(rq[s0]) != rid:
                s0 += 1
            var len_ = 0
            while s0 + len_ < n and Int(rq[s0 + len_]) == rid:
                len_ += 1
            var rec = open(cap + "/L" + layer + ".f16", "r")
            _ = rec.seek(s0 * 2 * h * 2)
            var rb = rec.read_bytes(len_ * 2 * h * 2)
            var hp = rb.unsafe_ptr().unsafe_bitcast[Float16]()
            var x = List[Float32](length=len_ * h, fill=0)
            for t in range(len_):
                for j in range(h):
                    x[t * h + j] = hp[t * 2 * h + j].cast[DType.float32]()
            var A = enc.acts(x, len_)
            var Y = matmul(wdt, A, h, nf, len_)  # Y[t][j] = sum_f A[t][f] Wd[f][j]
            var tcb = List[UInt8]()
            var yb = List[UInt8]()
            for t in range(len_):
                for j in range(h):
                    put_f32(tcb, Y[t * h + j] + bdp[h * nf + nf + nf * h + j])
                    put_f32(yb, hp[t * 2 * h + h + j].cast[DType.float32]())
            write_file(out + "/rows_tc_" + String(r) + ".f32", tcb)
            write_file(out + "/rows_y_" + String(r) + ".f32", yb)
            _ = len(rb)  # hp must outlive its use above
            for pos in range(32, min(len_, 481), 32):
                var ps = String()
                for t in range(pos):
                    ps += String(Int(tok[s0 + t])) + ("," if t + 1 < pos else "")
                var base = '"prompt":[' + ps + '],"n":1,"gain":"' + layer + ':0:1","slens_ctx":' + String(pos) + ',"slens_k":1,"slens_tgt":' + String(Int(tok[s0 + pos])) + ',"slens_flags":3,"splice_layer":' + layer
                s += '{"id":' + String(4000000 + 1000 * r + pos) + ',"prompt":[' + ps + '],"n":1,"gain":"' + layer + ':0:1","slens_ctx":' + String(pos) + ',"slens_k":0}\n'  # the unspliced base the arms are compared with
                s += '{"id":' + String(1000000 + 1000 * r + pos) + "," + base + ',"splice":"' + out + "/rows_tc_" + String(r) + '.f32"}\n'
                s += '{"id":' + String(2000000 + 1000 * r + pos) + "," + base + ',"splice":"' + out + "/rows_y_" + String(r) + '.f32"}\n'
                s += '{"id":' + String(3000000 + 1000 * r + pos) + "," + base + ',"splice_zero":1}\n'
        var sb = List[UInt8]()
        for c in s.as_bytes():
            sb.append(c)
        _ = len(tokb) + len(reqb) + len(bdb)
        write_file(out + "/req.jsonl", sb)
        print("requests written for", nreq, "held-out requests")
    else:
        var kl = List[Float64](length=3, fill=0)
        var nl = List[Float64](length=3, fill=0)
        var cnt = List[Int](length=3, fill=0)
        for line in read_text(String(a[2])).split("\n"):
            var ln = String(line)
            if ln.startswith('{"id":') and ln.find('"error"') >= 0:
                print("FAIL engine error:", ln[byte=0:200])
                exit(1)
            if ln.startswith('{"id":') and ln.find('"slens"') >= 0:
                var j = parse(ln)
                var arm = j.get("id").int() // 1000000 - 1
                kl[arm] += j.get("slens").get("kl").num()
                nl[arm] += j.get("slens").get("nll").num()
                cnt[arm] += 1
        if cnt[0] == 0 or cnt[0] != cnt[1] or cnt[0] != cnt[2]:
            print("FAIL tcsplice: missing slens lines", cnt[0], cnt[1], cnt[2])
            exit(1)
        var mk = List[Float64]()
        for q in range(3):
            mk.append(kl[q] / Float64(cnt[q]))
        var ratio = mk[0] / mk[2]
        var res = "KL tc=" + String(mk[0]) + " identity=" + String(mk[1]) + " zero=" + String(mk[2]) + " ratio_tc_over_zero=" + String(ratio) + " mean_nll tc=" + String(nl[0] / Float64(cnt[0])) + " identity=" + String(nl[1] / Float64(cnt[1])) + " zero=" + String(nl[2] / Float64(cnt[2])) + " positions=" + String(cnt[0])
        print(res)
        var rl = Rlog("tcsplice", "score " + String(a[2]), "")
        var vals = mk.copy()
        rl.done(res, String(a[2]), vals, 0.0, 10.0)
        var ok = ratio <= 0.25 and mk[1] <= 1e-3
        if not ok:
            rl.event("anomaly", "prereg-b1-splice bar missed: ratio " + String(ratio) + " (bar 0.25) identity KL " + String(mk[1]) + " | " + res, String(a[2]))
        print("PASS splice bars" if ok else "BAR MISSED: tc/zero KL ratio " + String(ratio) + " > 0.25 or identity KL " + String(mk[1]) + " > 1e-3")
