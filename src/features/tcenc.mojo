"""Transcoder encoder on the CPU: z = x We + be, TopK (ties kept, z > 0), as in tc.mojo's tie rule. tc.bin = We (h x nf), be, Wd (nf x h), bd."""
from mneuron.fwd import matmul


struct Enc(Movable):
    var wet: List[Float32]  # nf x h (We transposed)
    var be: List[Float32]
    var wd: List[Float32]  # nf x h
    var nf: Int
    var h: Int
    var k: Int

    def __init__(out self, path: String, h: Int, nf: Int, k: Int) raises:
        var b = open(path + "/tc.bin", "r").read_bytes()
        var p = b.unsafe_ptr().unsafe_bitcast[Float32]()
        self.nf = nf
        self.h = h
        self.k = k
        self.wet = List[Float32](length=nf * h, fill=0)
        self.be = List[Float32](length=nf, fill=0)
        self.wd = List[Float32](length=nf * h, fill=0)
        for j in range(h):
            for f in range(nf):
                self.wet[f * h + j] = p[j * nf + f]
        for f in range(nf):
            self.be[f] = p[h * nf + f]
        for i in range(nf * h):
            self.wd[i] = p[h * nf + nf + i]
        _ = len(b)

    def acts(self, x: List[Float32], T: Int) -> List[Float32]:
        """T x nf TopK activations of the rows of x (T x h)."""
        var z = matmul(self.wet, x, self.nf, self.h, T)
        for t in range(T):
            var vals = List[Float32]()
            for f in range(self.nf):
                z[t * self.nf + f] += self.be[f]
                if z[t * self.nf + f] > 0:
                    vals.append(z[t * self.nf + f])
            var thr = Float32(0)
            if len(vals) > self.k:
                sort(vals)
                thr = vals[len(vals) - self.k]
            for f in range(self.nf):
                if not (z[t * self.nf + f] > 0 and z[t * self.nf + f] >= thr):
                    z[t * self.nf + f] = 0
        return z^
