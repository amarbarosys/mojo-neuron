"""A1 oracle (docs/prereg-a1.md): compare an engine capture with the engine's own BARO_DUMP at the last prompt position."""
import json, sys
import numpy as np

d, dump, n_prompt, mut = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "mutant"
m = json.load(open(f"{d}/meta.json")); H, NL, nf, F = m["h"], m["n_layers"], len(m["fields"]), m["fields"]
tok = np.fromfile(f"{d}/tokens.i32", dtype=np.int32)
dm = np.fromfile(dump, dtype=np.float32).reshape(-1, 2 * NL, H)  # [token, slot, H]: slot 2l after attn, 2l+1 after ffn
for l in m["layers"]:
    a = np.fromfile(f"{d}/L{l}.f16", dtype=np.float16)
    if a.size != tok.size * nf * H: raise SystemExit(f"FAIL records L{l}: {a.size} values for {tok.size} tokens x {nf} x {H}")
    a = a.reshape(-1, nf, H); row = lambda f: a[n_prompt - 1, F.index(f)]
    ll = (l + 1 if l + 1 < NL else l - 1) if mut else l
    want = dm[0, 2 * ll + 1].astype(np.float16)
    same = row("r").view(np.uint16) == want.view(np.uint16)
    if mut:
        if same.all(): raise SystemExit(f"FAIL mutant L{l}: capture equals the dump of layer {ll}")
        continue
    if not same.all(): raise SystemExit(f"FAIL bit-equal L{l}: {int((~same).sum())} of {H} values differ")
    if "y" in F:
        yr = row("r").astype(np.float64) - dm[0, 2 * l].astype(np.float64)
        err = np.abs(yr - row("y").astype(np.float64)).max() / max(np.abs(yr).max(), 1e-9)
        if err > 1e-2: raise SystemExit(f"FAIL y L{l}: r - mid vs y relative {err:.4f}")
    if "x" in F:
        x = row("x").astype(np.float64); rms = float(np.sqrt((x * x).mean()))
        if not np.isfinite(x).all() or not 0.1 < rms < 100: raise SystemExit(f"FAIL x L{l}: rms {rms}")
    print(f"L{l}: r bit-equal ({H} values), y ok, x rms ok")
if not mut: print(f"{tok.size} tokens, {len(m['layers'])} layers")
