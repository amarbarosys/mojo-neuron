#!/usr/bin/env bash
# 400k-token layer-16 capture through the batched prefill path (fields xy, f16, same layout as .work/cap/qwythos-9b = 195 x 512 tokens of Brain notes), held apart from that set:
# windows of 512 tokens from other Brain notes, any window sharing a 32-token run with the 100k prompts is dropped. One gpu-wait job. Output: main checkout .work/capture/l16-400k (disk budget 8 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2b}; ENGINE=${ENGINE:-$BARO/.work/engine-9b-pfcap}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
TOKJ=${TOKJ:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf/tokenizer.json}; PYB=${PYB:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python3}
OLD=${OLD:-$HOME/Projects/mojo/mojo-neuron-lanes/features/.work/qw-req.jsonl}; OUT=${OUT:-$HOME/Projects/mojo/mojo-neuron/.work/capture/l16-400k}; NREQ=${NREQ:-782}
J=$PWD/.work/pfcap-400k; mkdir -p "$J"; [ -e "$OUT" ] && { echo "FAIL setup: $OUT exists (the engine appends; move it first)"; exit 1; }; mkdir -p "$OUT"
[ -x "$ENGINE" ] && [ -f "$OLD" ] || { echo "FAIL setup: engine or $OLD missing"; exit 1; }
"$PYB" -I - "$J" "$TOKJ" "$OLD" "$NREQ" <<'PYEOF'
import sys, json, os, random, glob
from tokenizers import Tokenizer
J, tj, old, nreq = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); tok = Tokenizer.from_file(tj)
seen = set()
for l in open(old):
    p = json.loads(l)["prompt"]
    for i in range(0, len(p) - 31): seen.add(tuple(p[i:i + 32]))
files = sorted(f for f in glob.glob(os.path.expanduser("(private notes)"), recursive=True) if "/.archive/" not in f and "/.git/" not in f and os.path.getsize(f) > 4000)
random.Random(5).shuffle(files); reqs = []; skipped = 0
for f in files:
    try: ids = tok.encode(open(f, errors="ignore").read(), add_special_tokens=False).ids
    except Exception: continue
    for i in range(0, len(ids) - 511, 512):
        w = ids[i:i + 512]
        if any(tuple(w[j:j + 32]) in seen for j in range(0, 481)): skipped += 1; continue
        reqs.append({"id": len(reqs) + 1, "prompt": w, "n": 1, "spec": False})
        if len(reqs) == nreq: break
    if len(reqs) == nreq: break
assert len(reqs) == nreq, f"only {len(reqs)} windows from {len(files)} files"
open(f"{J}/req.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in reqs)); print(f"{nreq} windows of 512 tokens ({nreq * 512} tokens) from {len({1})} corpus pass; dropped {skipped} windows that shared a 32-token run with the 100k set")
PYEOF
cat > "$J/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK' BARO_TMAX=1024 BARO_CAPTURE='$OUT' BARO_CAPTURE_LAYERS=16 BARO_CAPTURE_FIELDS=xy BARO_CAPTURE_MODEL=qwythos-9b
exec '$ENGINE' < '$J/req.jsonl' > '$J/out.jsonl' 2> '$J/err.txt'
EOS
chmod +x "$J/run.sh"; gpu-wait run --timeout 1800 -- "$J/run.sh" || true
"$PYB" -I - "$J" "$OUT" "$NREQ" <<'PYEOF'
import sys, json, os, numpy as np
J, OUT, nreq = sys.argv[1], sys.argv[2], int(sys.argv[3])
dones = [json.loads(l) for l in open(f"{J}/out.jsonl") if l.startswith('{"id":') and '"done"' in l]
if len(dones) != nreq: raise SystemExit(f"FAIL run: {len(dones)} done lines for {nreq} requests, see {J}/err.txt")
secs = sum(d["prefill_s"] for d in dones); ntok = nreq * 512
size = os.path.getsize(f"{OUT}/L16.f16"); tok = np.fromfile(f"{OUT}/tokens.i32", dtype=np.int32)
want = np.concatenate([np.array(json.loads(l)["prompt"], dtype=np.int32) for l in open(f"{J}/req.jsonl")])
if size != ntok * 2 * 4096 * 2 or not np.array_equal(tok, want): raise SystemExit(f"FAIL capture: size {size} (want {ntok * 2 * 4096 * 2}) or tokens differ from the prompts")
if size > 8e9: raise SystemExit(f"FAIL disk: {size} bytes over the 8 GB budget")
print(f"records {ntok}, {size / 1e9:.2f} GB, prefill rate {ntok / secs:.0f} tok/s (sum of the engine's prefill_s over {nreq} requests, file writes included)")
PYEOF
sha=$(sha256sum "$OUT/L16.f16" | cut -c1-16); echo "sha256(L16.f16)[:16] $sha"; echo "PASS pfcap-400k: $OUT"
