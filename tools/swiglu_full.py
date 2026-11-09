"""Full-tensor oracle for fires: an independent numpy f32 forward of a bf16/f16/f32 dense GGUF (qwen2, llama, granite) that writes
EVERY activation silu(gate)*up of every layer, because llama.cpp's eval-callback prints only six values per token.
Usage: swiglu_full.py MODEL ORACLE_TXT OUT.f32     (ORACLE_TXT from swiglu_oracle.py: its `ids` line and llama.cpp edge values)
OUT.f32: for each layer listed in ORACLE_TXT, T x ffn float32 (token-major). The script first checks itself against llama.cpp's
printed edge values (same tolerance as fires_check), so the full oracle is itself anchored to llama.cpp."""
import struct, sys
import numpy as np

SZ = {0: 4, 1: 2, 30: 2}


def read_gguf(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver, nt, nkv = struct.unpack('<IQQ', f.read(20))

    def rstr():
        n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode()

    def rval(t):
        fm = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if t in fm:
            return struct.unpack('<' + fm[t], f.read(struct.calcsize(fm[t])))[0]
        if t == 8: return rstr()
        if t == 9:
            et, n = struct.unpack('<IQ', f.read(12)); return [rval(et) for _ in range(n)]
        raise ValueError(t)
    kv = {}
    for _ in range(nkv):
        k = rstr(); t, = struct.unpack('<I', f.read(4)); kv[k] = rval(t)
    ts = {}
    for _ in range(nt):
        n = rstr(); nd, = struct.unpack('<I', f.read(4)); d = struct.unpack(f'<{nd}Q', f.read(8 * nd)); ty, off = struct.unpack('<IQ', f.read(12))
        ts[n] = (d, ty, off)
    al = kv.get('general.alignment', 32)
    base = (f.tell() + al - 1) // al * al
    mm = np.memmap(path, dtype=np.uint8, mode='r')

    def get(name):
        d, ty, off = ts[name]
        n = int(np.prod(d))
        if ty not in SZ: raise SystemExit(f"swiglu_full: tensor {name} has ggml type {ty}; use a bf16, f16 or f32 file")
        raw = mm[base + off: base + off + n * SZ[ty]]
        if ty == 0: a = raw.view(np.float32)
        elif ty == 1: a = raw.view(np.float16).astype(np.float32)
        else: a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return a.reshape(d[::-1]).astype(np.float32)  # ggml dims are reversed: [out][in]
    return kv, ts, get


def rmsnorm(x, w, eps):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps).astype(np.float32) * w


def rope(x, base, rd, neox):  # x: T, heads, hd
    T, nh, hd = x.shape
    pos = np.arange(T)[:, None]
    fr = base ** (-2.0 * np.arange(rd // 2) / rd)
    a = pos * fr[None, :]
    c, s = np.cos(a)[:, None, :].astype(np.float32), np.sin(a)[:, None, :].astype(np.float32)
    y = x.copy()
    if neox:
        x0, x1 = x[..., :rd // 2], x[..., rd // 2:rd]
        y[..., :rd // 2] = x0 * c - x1 * s; y[..., rd // 2:rd] = x0 * s + x1 * c
    else:
        x0, x1 = x[..., 0:rd:2], x[..., 1:rd:2]
        y[..., 0:rd:2] = x0 * c - x1 * s; y[..., 1:rd:2] = x0 * s + x1 * c
    return y


def main():
    model, otxt, outp = sys.argv[1:4]
    ids, layers, edge = [], [], {}
    for line in open(otxt):
        w = line.split()
        if not w: continue
        if w[0] == 'ids': ids = [int(i) for i in w[1].split(',')]
        elif w[1] != 'sum':
            L = int(w[0][1:]); t = int(w[1][1:])
            if L not in layers: layers.append(L)
            edge[(L, t)] = [float(v) for v in w[2:8]]
    kv, ts, get = read_gguf(model)
    arch = kv['general.architecture']
    assert arch in ('qwen2', 'llama', 'granite'), arch
    assert 'rope_freqs.weight' not in ts, 'rope_freqs.weight is not modelled here'
    nh = kv[f'{arch}.attention.head_count']; nkv = kv.get(f'{arch}.attention.head_count_kv', nh)
    eps = kv.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5); base = kv.get(f'{arch}.rope.freq_base', 10000.0)
    emb = get('token_embd.weight'); h = emb.shape[1]
    hd = get('blk.0.attn_q.weight').shape[0] // nh
    rd = kv.get(f'{arch}.rope.dimension_count', hd)
    es, rs = kv.get(f'{arch}.embedding_scale', 1.0), kv.get(f'{arch}.residual_scale', 1.0)
    asc = kv.get(f'{arch}.attention.scale', 1.0 / np.sqrt(hd))
    x = emb[ids] * np.float32(es); T = len(ids)
    out = open(outp, 'wb'); bad = 0; tot = 0
    for l in range(max(layers) + 1):
        p = f'blk.{l}.'
        xn = rmsnorm(x, get(p + 'attn_norm.weight'), eps)
        q = xn @ get(p + 'attn_q.weight').T; k = xn @ get(p + 'attn_k.weight').T; v = xn @ get(p + 'attn_v.weight').T
        if p + 'attn_q.bias' in ts:
            q += get(p + 'attn_q.bias'); k += get(p + 'attn_k.bias'); v += get(p + 'attn_v.bias')
        q = rope(q.reshape(T, nh, hd), base, rd, arch == 'qwen2'); k = rope(k.reshape(T, nkv, hd), base, rd, arch == 'qwen2'); v = v.reshape(T, nkv, hd)
        kk = np.repeat(k, nh // nkv, axis=1); vv = np.repeat(v, nh // nkv, axis=1)
        sc = np.einsum('thd,uhd->htu', q, kk) * np.float32(asc)
        sc = np.where(np.tril(np.ones((T, T), bool))[None], sc, -np.inf)
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        att = np.einsum('htu,uhd->thd', sc.astype(np.float32), vv).reshape(T, nh * hd)
        x = x + (att @ get(p + 'attn_output.weight').T) * np.float32(rs)
        xn2 = rmsnorm(x, get(p + 'ffn_norm.weight'), eps)
        ga = xn2 @ get(p + 'ffn_gate.weight').T; up = xn2 @ get(p + 'ffn_up.weight').T
        a = (ga / (1.0 + np.exp(-ga.astype(np.float64))).astype(np.float32)) * up
        if l in layers:
            out.write(a.astype(np.float32).tobytes())
            ffn = a.shape[1]
            for t in range(T):  # anchor: llama.cpp's printed first and last three values
                w = edge[(l, t)]; g = [a[t, 0], a[t, 1], a[t, 2], a[t, ffn - 3], a[t, ffn - 2], a[t, ffn - 1]]
                for gv, wv in zip(g, w):
                    tot += 1; bad += abs(gv - wv) > 0.01 + 0.05 * abs(wv)
        if l < max(layers):
            x = x + (a @ get(p + 'ffn_down.weight').T) * np.float32(rs)
    out.close()
    print(f"swiglu_full: {len(layers)} layers x {T} tokens x {ffn}; vs llama.cpp edge values {bad} of {tot} outside tolerance")
    if bad * 100 > tot * 3: sys.exit("FAIL swiglu_full: numpy oracle disagrees with llama.cpp")


main()
