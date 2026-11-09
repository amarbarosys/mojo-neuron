"""reverse-oracle OUTDIR GGUF: f64 recomputation of reverse.jsonl from contrast.dump.bin (python gguf reader, own bf16 decode).
The tool's top and bottom lists must equal the oracle's: same (layer, neuron), cosine within 2e-3."""
import json, sys
import numpy as np
from gguf import GGUFReader

out, gguf = sys.argv[1:3]
rows = [json.loads(l) for l in open(f"{out}/reverse.jsonl")]
top = len(rows) // 2
r = GGUFReader(gguf)
W = {t.name: t for t in r.tensors if t.name.endswith("ffn_down.weight")}
nl = len(W)
h = W["blk.0.ffn_down.weight"].data.shape[0]
dump = np.fromfile(f"{out}/contrast.dump.bin", dtype=np.float32).reshape(2, -1, nl, h)
swap = len(sys.argv) > 3 and sys.argv[3] == "swap"
d = (dump[1] - dump[0] if swap else dump[0] - dump[1]).astype(np.float64).mean(axis=0)
cs = []
for L in range(nl):
    t = W[f"blk.{L}.ffn_down.weight"]
    a = t.data
    if a.dtype == np.uint8:  # BF16 arrives as raw bytes
        a = np.ascontiguousarray(a).view(np.uint16)
    m = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64) if a.dtype == np.uint16 else a.astype(np.float64)
    c = (m.T @ d[L]) / (np.linalg.norm(m, axis=0) * np.linalg.norm(d[L]))
    cs += [(float(c[n]), L, n) for n in range(c.size)]
cs.sort(reverse=True)
want = cs[:top] + cs[-top:][::-1]
bad = 0
for (c, L, n), g in zip(want, rows):
    if (L, n) != (g["layer"], g["neuron"]) or abs(c - g["cos"]) > 2e-3:
        bad += 1
if bad:
    raise SystemExit(f"FAIL reverse-oracle: {bad} of {2 * top} entries differ (first oracle {want[0]}, tool {rows[0]})")
print(f"PASS reverse-oracle: top {top} and bottom {top} of {len(cs)} neurons equal the f64 answer; best {want[0]}")
