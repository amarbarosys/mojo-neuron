#!/usr/bin/env bash
# Gate G1: rank-k implants in engine.mojo. (1) k=0 file and "implants":0 give the logits of the unmodified layer-loop engine bit for bit, (2) an unrelated
# prompt is untouched by an implant keyed on prompt p, (3) p's own logits move, (4) a random-key implant leaves p untouched (the key matters).
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b-imp}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; L=${LAYER:-16}
O=$PWD/.work/implants-check; mkdir -p "$O/cap"; rm -f "$O"/*.req "$O"/*.out
[ -x "$ENGINE" ] || { echo "FAIL setup: no engine at $ENGINE"; exit 1; }
NP=$(wc -w < "$PACK/prompt-tokens.txt"); P=$(tr -s ' \n' ',' < "$PACK/prompt-tokens.txt" | sed 's/,$//')
Q=$(tr -s ' \n' '\n' < "$PACK/prompt-tokens.txt" | sed '/^$/d' | tac | paste -sd, -)   # the same tokens reversed: an unrelated prompt of equal length
cat > "$O/cap.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME'
env BARO_SPEC=0 BARO_PACK='$PACK' BARO_CAPTURE='$O/cap' BARO_CAPTURE_LAYERS=$L '$ENGINE' > '$O/cap.log' 2>&1
EOS
chmod +x "$O/cap.sh"; [ -s "$O/cap/L$L.f16" ] || gpu-wait run --timeout 300 -- "$O/cap.sh" || true  # a capture of this prompt from an earlier run is reused (same engine, same prompt)
[ -s "$O/cap/L$L.f16" ] || { echo "FAIL capture: $(tail -n 2 "$O/cap.log" | cut -c1-200)"; exit 1; }
R=$((NP - 1))
python3 -I tools/implants-write.py "$O/cap" $L $R "$O/k0.bin" k=0
python3 -I tools/implants-write.py "$O/cap" $L $R "$O/k1.bin" k=1 > /dev/null
python3 -I tools/implants-write.py "$O/cap" $L $R "$O/rnd.bin" k=1 key=random > /dev/null
cat > "$O/rows.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME'
cd '$O'
run() { # run NAME IMPLANTS(or none) FLAG
  { echo "{\"id\":1,\"prompt\":[$P],\"n\":1,\"logits_topk\":20,\"gain\":\"$L:0:1\"\$3}"; echo "{\"id\":2,\"prompt\":[$Q],\"n\":1,\"logits_topk\":20,\"gain\":\"$L:0:1\"\$3}"; } > "\$1.req"
  env BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK' \${2:+BARO_IMPLANTS=\$2} '$ENGINE' < "\$1.req" > "\$1.out" 2>&1
}
run base ""
run k0 '$O/k0.bin'
run off '$O/k1.bin' ',"implants":0'
run k1 '$O/k1.bin'
run rnd '$O/rnd.bin'
EOS
chmod +x "$O/rows.sh"; gpu-wait run --timeout 900 -- "$O/rows.sh" || true
python3 -I - "$O" <<'PYEOF'
import sys, json
O = sys.argv[1]
def rows(n):
    r = {}
    for l in open(f"{O}/{n}.out"):
        if l.startswith('{"id":') and '"logits_topk"' in l:
            d = json.loads(l); r[d["id"]] = {e["id"]: e["logit"] for e in d["logits_topk"]}
    if len(r) != 2: raise SystemExit(f"FAIL rows {n}: {len(r)} logits_topk lines: {open(f'{O}/{n}.out').read()[-300:]}")
    return r
b, k0, off, k1, rnd = (rows(n) for n in ("base", "k0", "off", "k1", "rnd"))
same = lambda x, y: all(x[i] == y[i] for i in x if i in y) and len(set(x) & set(y)) >= 15
if not (same(b[1], k0[1]) and same(b[2], k0[2])): raise SystemExit("FAIL k=0: logits differ from the unmodified engine")
if not (same(b[1], off[1]) and same(b[2], off[2])): raise SystemExit('FAIL "implants":0: logits differ from the unmodified engine')
if not same(b[2], k1[2]): raise SystemExit("FAIL unrelated prompt: implant keyed on p moved q's logits")
d = max(abs(k1[1][i] - b[1][i]) for i in b[1] if i in k1[1])
if d <= 0.05: raise SystemExit(f"FAIL implant: p's own logits moved by only {d:.3e}")
if not same(b[1], rnd[1]): raise SystemExit("FAIL mutant: a random-key implant moved p's logits")
print(f"k=0 and implants:0 bit-equal, q untouched, p moved by {d:.3f} (max over its top-20 logits), random key leaves p untouched")
PYEOF
echo "PASS implants-check: engine rank-k implants are off bit-for-bit, fire only on the keyed prompt, and the key matters"
