"""Oracle for lens.bin v1 (docs/format.md): header, size, ordering, id range. Usage: check_lens_bin.py FILE.lens.bin VOCAB"""
import struct, sys, math
p, vocab = sys.argv[1], int(sys.argv[2])
b = open(p, 'rb').read()
if b[:4] != b'MNLN': sys.exit(f"FAIL lens.bin: bad magic in {p}")
ver, layer, ffn, h, k = struct.unpack_from('<5I', b, 4)
if ver != 1: sys.exit(f"FAIL lens.bin: version {ver}")
if len(b) != 24 + ffn * k * 16: sys.exit(f"FAIL lens.bin: size {len(b)} != {24 + ffn * k * 16}")
off = 24
for n in range(ffn):
    ti = struct.unpack_from(f'<{k}i', b, off); tz = struct.unpack_from(f'<{k}f', b, off + 4 * k)
    bi = struct.unpack_from(f'<{k}i', b, off + 8 * k); bz = struct.unpack_from(f'<{k}f', b, off + 12 * k)
    off += 16 * k
    for ids in (ti, bi):
        if any(not (0 <= i < vocab) for i in ids): sys.exit(f"FAIL lens.bin: neuron {n} id out of range")
    for z in tz + bz:
        if not math.isfinite(z): sys.exit(f"FAIL lens.bin: neuron {n} non-finite z")
    if any(tz[i] < tz[i + 1] for i in range(k - 1)): sys.exit(f"FAIL lens.bin: neuron {n} top not descending")
    if any(bz[i] > bz[i + 1] for i in range(k - 1)): sys.exit(f"FAIL lens.bin: neuron {n} bottom not ascending")
    if tz[0] < bz[0]: sys.exit(f"FAIL lens.bin: neuron {n} top below bottom")
print(f"ok lens.bin layer {layer} ffn {ffn} hidden {h} k {k}")
