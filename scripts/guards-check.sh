#!/usr/bin/env bash
# Admission guards of the neuron2b engine ops (FSM audit report-B.md, P1 to P3): requests that must be refused with an error line, plus a request that must still be accepted.
# usage: guards-check.sh ENGINE_BINARY [TAG]     cases: P1 splice rows vs prompt+n, P2 implants_* without BARO_IMPLANTS and lo > k, P3 slens_foil without slens_tgt.
# Exit 0 only when every bad request is refused with an error line that names its cause and every good request is served. P4 (capture with pf_chunk > PF_CAP_ROWS) cannot be reached (pf_chunk = min(BARO_PREFILL_C, CP), CP = 1024 = PF_CAP_ROWS): no runtime test, a comptime assert holds it.
set -euo pipefail
cd "$(dirname "$0")/.."
ENGINE=${1:?usage: guards-check.sh ENGINE_BINARY [TAG]}; TAG=${2:-run}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; PYB=${PYB:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python3}
O=$PWD/.work/guards-$TAG; rm -rf "$O"; mkdir -p "$O"
"$PYB" -I - "$O" <<'PYEOF'
import sys, json, numpy as np
O = sys.argv[1]; H = 4096; ids = [760, 6511, 314, 9338, 369, 264, 3128, 3415, 11, 323, 892, 1054, 2631, 303, 17, 18, 19, 20, 21, 22]  # 20 tokens
np.zeros((16, H), np.float32).tofile(f"{O}/spl16.f32")
k = 2; keys = np.random.default_rng(0).standard_normal((k, H)).astype(np.float32); vals = np.random.default_rng(1).standard_normal((k, H)).astype(np.float32)
with open(f"{O}/imp.bin", "wb") as f:
    f.write(b"NIMP0001" + np.array([H, 1], np.uint32).tobytes() + np.array([16, k], np.uint32).tobytes() + np.float32(0.95).tobytes() + keys.tobytes() + np.full(k, 1.0, np.float32).tobytes() + vals.tobytes())
A = [  # engine A: BARO_STEER_SERVE=1, no implants file
    {"id": 1, "prompt": ids, "n": 4, "gain": "16:0:1", "splice_layer": 16, "splice": f"{O}/spl16.f32"},        # P1 bad: needs 24 rows, file has 16
    {"id": 2, "prompt": ids[:8], "n": 2, "gain": "16:0:1", "splice_layer": 16, "splice": f"{O}/spl16.f32"},    # P1 good: needs 10 rows
    {"id": 3, "prompt": ids[:8], "n": 1, "implants_scale": 2.0},                                              # P2 bad: no implants loaded
    {"id": 4, "prompt": ids[:8], "n": 1, "slens_ctx": 0, "slens_k": 0, "gain": "16:0:1"},                     # P3 setup: a base context
    {"id": 5, "prompt": ids[:8], "n": 1, "slens_ctx": 0, "slens_k": 1, "slens_foil": 5, "gain": "16:0:1"},    # P3 bad: foil without target
    {"id": 6, "prompt": ids[:8], "n": 1, "gain": "16:0:1"},                                                   # sanity: plain request still served
]
B = [  # engine B: BARO_IMPLANTS loaded
    {"id": 1, "prompt": ids[:8], "n": 1, "implants": 1, "implants_lo": 5, "implants_k": 2},                    # P2 bad: lo > k
    {"id": 2, "prompt": ids[:8], "n": 1, "implants": 1, "implants_lo": 0, "implants_k": 2},                    # P2 good
]
for n, rs in (("A", A), ("B", B)): open(f"{O}/{n}.req", "w").write("".join(json.dumps(r) + "\n" for r in rs))
PYEOF
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK' BARO_TMAX=512
env BARO_STEER_SERVE=1 '$ENGINE' < '$O/A.req' > '$O/A.out' 2> '$O/A.err'
env BARO_STEER_SERVE=1 BARO_IMPLANTS='$O/imp.bin' '$ENGINE' < '$O/B.req' > '$O/B.out' 2> '$O/B.err'
EOS
chmod +x "$O/run.sh"; gpu-wait run --timeout 900 -- "$O/run.sh" || true
"$PYB" -I - "$O" <<'PYEOF'
import sys, json
O = sys.argv[1]; bad = []
def lines(n):
    r = {}
    for l in open(f"{O}/{n}.out"):
        if l.startswith('{"id":'):
            j = json.loads(l); r.setdefault(j["id"], []).append(j)
    return r
A, B = lines("A"), lines("B")
err = lambda r, i: next((j["error"] for j in r.get(i, []) if "error" in j), None); done = lambda r, i: any(j.get("done") for j in r.get(i, []))
checks = [("P1 splice file shorter than prompt+n is refused", err(A, 1) is not None and "splice" in (err(A, 1) or "")), ("P1 splice file long enough is served", done(A, 2) and err(A, 2) is None),
          ("P2 implants_scale without BARO_IMPLANTS is refused", err(A, 3) is not None and "implants" in (err(A, 3) or "")), ("P2 implants_lo > implants_k is refused", err(B, 1) is not None and "implants" in (err(B, 1) or "")),
          ("P2 a valid implants range is served", done(B, 2) and err(B, 2) is None), ("P3 slens_foil without slens_tgt is refused", err(A, 5) is not None and "slens" in (err(A, 5) or "")), ("plain request still served", done(A, 6) and err(A, 6) is None)]
for name, ok in checks:
    print(("ok      " if ok else "MISSING ") + name)
    if not ok: bad.append(name)
sys.exit(1 if bad else 0)
PYEOF
echo "PASS guards-check ($TAG): every bad request refused with a named error, every good request served"
