"""tcceil CAP LAYER [DUMP.f32]: closed-form ceiling for a transcoder on a capture (docs/reports): ridge x -> y on the first 90 % of the records (cut at a request
boundary, x and y centred by the train means), held-out FVU = 1 - R2. A dense linear map; a TopK transcoder can only beat it through its non-linearity."""
from std.sys import argv, exit
from mneuron.ggio import write_file, put_f32
from mneuron.rlog import Rlog
from xmap import fit, r2


def main() raises:
    var av = argv()
    var cap = String(av[1])
    var layer = String(av[2])
    var rl = Rlog("tcceil", cap + " " + layer, "")
    var meta = open(cap + "/meta.json", "r").read_bytes()
    var tokb = open(cap + "/req.i32", "r").read_bytes()
    var req = tokb.unsafe_ptr().unsafe_bitcast[Int32]()
    var n = len(tokb) // 4
    var h = 4096
    for q in range(len(meta) - 5):  # "h":NNNN
        if meta[q] == 34 and meta[q + 1] == 104 and meta[q + 2] == 34 and meta[q + 3] == 58:
            h = 0
            var z = q + 4
            while meta[z] >= 48 and meta[z] <= 57:
                h = h * 10 + Int(meta[z]) - 48
                z += 1
    var hs = n * 9 // 10
    while hs < n and req[hs] == req[hs - 1]:
        hs += 1
    if hs < 1 or n - hs < 1:
        print("FAIL tcceil: " + String(n) + " records in " + String(Int(req[n - 1]) - Int(req[0]) + 1) + " request(s) leave " + String(n - hs) + " held-out rows (the split cuts at a request boundary after 90 %)")
        exit(1)
    var f = open(cap + "/L" + layer + ".f16", "r")
    var rb = f.read_bytes(n * 2 * h * 2)
    if len(rb) < n * 2 * h * 2:
        print("FAIL tcceil: L" + layer + ".f16 holds " + String(len(rb)) + " bytes, req.i32 lists " + String(n) + " records of " + String(2 * h * 2) + " bytes")
        exit(1)
    var hp = rb.unsafe_ptr().unsafe_bitcast[Float16]()
    var X = List[Float32](length=n * h, fill=0)
    var Y = List[Float32](length=n * h, fill=0)
    var mx = List[Float64](length=h, fill=0)
    var my = List[Float64](length=h, fill=0)
    for t in range(n):
        for j in range(h):
            X[t * h + j] = hp[t * 2 * h + j].cast[DType.float32]()
            Y[t * h + j] = hp[t * 2 * h + h + j].cast[DType.float32]()
            if t < hs:
                mx[j] += Float64(X[t * h + j]) / Float64(hs)
                my[j] += Float64(Y[t * h + j]) / Float64(hs)
    for t in range(n):
        for j in range(h):
            X[t * h + j] -= Float32(mx[j])
            Y[t * h + j] -= Float32(my[j])
    var ss0: Float64 = 0  # share of the total variance of y that sits in the first record of each request (the attention-sink row)
    var sst: Float64 = 0
    var n0 = 0
    for t in range(n):
        var q: Float64 = 0
        for j in range(h):
            q += Float64(Y[t * h + j]) ** 2
        sst += q
        if t == 0 or req[t] != req[t - 1]:
            ss0 += q
            n0 += 1
    print("first-record rows:", n0, "of", n, "carry", ss0 / sst, "of the centred y sum of squares")
    var M = fit(X, Y, hs, h, h)
    if len(av) > 3:  # the ridge matrix (h x h, row i = output i) for tc_train --init-ridge
        var mb = List[UInt8]()
        for v in M:
            put_f32(mb, v)
        write_file(String(av[3]), mb)
    var heldout = r2(M, X, Y, hs, n, h, h)
    var train = r2(M, X, Y, 0, hs, h, h)
    var res = "linear_ridge heldout_fvu=" + String(1.0 - heldout) + " train_fvu=" + String(1.0 - train) + " train_rows=" + String(hs) + " heldout_rows=" + String(n - hs) + " h=" + String(h)
    _ = len(rb) + len(tokb) + len(meta)  # the raw pointers hp and req do not keep their owners alive
    print(res)
    var vals = List[Float64]()
    vals.append(1.0 - heldout)
    rl.done(res, "", vals, -1.0, 5.0)
