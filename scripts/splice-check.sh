#!/usr/bin/env bash
# Gate for the engine splice op (docs/prereg-splice.md): capture y of layer 16 on a 48-token text, then in ONE engine process score positions 8..47 with no splice,
# an identity splice (the true y), splice_zero, a zeros file and a shifted-rows mutant. One more engine process (the engine without the new fields) gives the bar-1 row.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2b}; ENGINE=${ENGINE:-$BARO/.work/engine-9b-splice}; OLD=${OLD:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2/.work/engine-9b-imp}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; TOKJ=${TOKJ:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf/tokenizer.json}
PYB=${PYB:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python3}; L=${LAYER:-16}
O=$PWD/.work/splice-check; rm -rf "$O"; mkdir -p "$O/cap"
[ -x "$ENGINE" ] && [ -x "$OLD" ] || { echo "FAIL setup: engine $ENGINE or $OLD missing"; exit 1; }
"$PYB" -I - "$O" "$TOKJ" <<'PYEOF'
import sys, json
from tokenizers import Tokenizer
O, tj = sys.argv[1], sys.argv[2]
ids = Tokenizer.from_file(tj).encode(" ".join(open("tests/data/eval64.txt").read().split("\n")[:6])).ids[:48]
assert len(ids) == 48, len(ids)
json.dump(ids, open(f"{O}/text.json", "w"))
open(f"{O}/cap.req", "w").write(json.dumps({"id": 1, "prompt": ids, "n": 1}) + "\n")
PYEOF
cat > "$O/cap.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK' BARO_CAPTURE='$O/cap' BARO_CAPTURE_LAYERS=$L BARO_CAPTURE_FIELDS=y
exec '$ENGINE' < '$O/cap.req' > '$O/cap.out' 2> '$O/cap.err'
EOS
chmod +x "$O/cap.sh"; gpu-wait run --timeout 300 -- "$O/cap.sh" || true
[ -s "$O/cap/L$L.f16" ] || { echo "FAIL capture: $(tail -n 2 "$O/cap.err" | cut -c1-200)"; exit 1; }
"$PYB" -I - "$O" $L <<'PYEOF'
import sys, json, numpy as np
O, L = sys.argv[1], sys.argv[2]
ids = json.load(open(f"{O}/text.json")); H = json.load(open(f"{O}/cap/meta.json"))["h"]
y = np.fromfile(f"{O}/cap/L{L}.f16", dtype=np.float16).reshape(-1, H).astype(np.float32)
assert y.shape[0] == 48, y.shape
y.tofile(f"{O}/rows.f32"); np.zeros_like(y).tofile(f"{O}/zeros.f32"); np.roll(y, 7, axis=0).tofile(f"{O}/shift.f32")
reqs, nid = [], 1
common = lambda i: {"prompt": ids[:i], "n": 1, "gain": f"{L}:0:1"}
for i in range(8, 48): reqs.append({"id": nid, **common(i), "slens_ctx": i, "slens_k": 0}); nid += 1   # bases, no splice
cfgs = {"none": {}, "ident": {"splice_layer": int(L), "splice": f"{O}/rows.f32"}, "zero": {"splice_layer": int(L), "splice_zero": 1},
        "zfile": {"splice_layer": int(L), "splice": f"{O}/zeros.f32"}, "shift": {"splice_layer": int(L), "splice": f"{O}/shift.f32"}}
meta = {}
for ci, (name, extra) in enumerate(cfgs.items()):
    meta[name] = []
    for i in range(8, 48):
        reqs.append({"id": 1000 * (ci + 1) + i, **common(i), **extra, "slens_ctx": i, "slens_k": 1, "slens_tgt": ids[i], "slens_flags": 3}); meta[name].append(1000 * (ci + 1) + i)
open(f"{O}/main.req", "w").write("".join(json.dumps(r) + "\n" for r in reqs)); json.dump(meta, open(f"{O}/meta.json", "w"))
row = {"id": 1, "prompt": ids[:20], "n": 1, "logits_topk": 20, "gain": f"{L}:0:1"}
open(f"{O}/row.req", "w").write(json.dumps(row) + "\n")
PYEOF
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK'
'$ENGINE' < '$O/main.req' > '$O/main.out' 2> '$O/main.err'
'$ENGINE' < '$O/row.req' > '$O/row-new.out' 2> '$O/row-new.err'
'$OLD' < '$O/row.req' > '$O/row-old.out' 2> '$O/row-old.err'
EOS
chmod +x "$O/run.sh"; gpu-wait run --timeout 900 -- "$O/run.sh" || true
[ -s "$O/main.out" ] || { echo "FAIL run: $(tail -n 2 "$O/main.err" | cut -c1-200)"; exit 1; }
export NX_RLOG=$O/rl
"$PYB" -I tools/splice-verify.py "$O"
echo "PASS splice-check: no-field request unchanged, identity splice keeps NLL, zero and shifted rows raise it, zeros file equals splice_zero"
