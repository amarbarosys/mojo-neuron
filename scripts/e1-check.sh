#!/usr/bin/env bash
# Gate E1 (docs/prereg-e1.md): baro-serve steer + capture passthrough against the engine's own direct runs.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b-e1}; SERVE=${SERVE:-$BARO/serve/target/release/baro-serve}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}; L=${LAYER:-16}; PORT=${PORT:-18471}
N=${NEURON:-$(sed -n 2p .work/scan-check/batch/L16.scan.tsv 2>/dev/null | cut -f1 || true)}; N=${N:-3}
O=$PWD/.work/e1-check; rm -rf "$O"; mkdir -p "$O/cap" "$O/direct-cap" "$O/row"
[ -x "$ENGINE" ] && [ -x "$SERVE" ] || { echo "FAIL setup: engine $ENGINE or server $SERVE missing"; exit 1; }
NP=$(wc -w < "$PACK/prompt-tokens.txt"); P=$(tr -s ' \n' ',' < "$PACK/prompt-tokens.txt" | sed 's/,$//')
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
set -uo pipefail
export PATH='$PATH' HOME='$HOME'
cd '$BARO'
# direct engine: capture of the prompt (A1 path) and the steered raw logits row
env BARO_SPEC=0 BARO_PACK='$PACK' BARO_CAPTURE='$O/direct-cap' BARO_CAPTURE_LAYERS=$L '$ENGINE' > '$O/direct-cap.log' 2>&1
echo '{"id":1,"prompt":[$P],"n":1,"top_logprobs":1,"gain":"$L:$N:0"}' > '$O/row.req'
env BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK' BARO_DUMP_LOGITS_DIR='$O/row' '$ENGINE' < '$O/row.req' > '$O/row.log' 2>&1
# server
env BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_PACK='$PACK' BARO_CAPTURE='$O/cap' BARO_CAPTURE_LAYERS=$L '$SERVE' --engine '$ENGINE' --pack '$PACK' --port $PORT > '$O/serve.log' 2>&1 &
SP=\$!
for i in \$(seq 1 120); do curl -sf localhost:$PORT/health > /dev/null && break; sleep 1; done
req() { curl -s -m 120 localhost:$PORT/v1/completions -H 'content-type: application/json' -d "\$1"; }
req '{"prompt":[$P],"max_tokens":1,"temperature":0,"logits_topk":20}' > '$O/plain.json'
req '{"prompt":[$P],"max_tokens":1,"temperature":0,"logits_topk":20,"steer":[{"layer":$L,"neuron":$N,"scale":0}]}' > '$O/steer.json'
req '{"prompt":[$P],"max_tokens":4,"temperature":0,"capture":"t1"}' > '$O/cap.json'
kill \$SP; wait \$SP 2>/dev/null
exit 0
EOS
chmod +x "$O/run.sh"
gpu-wait run --timeout 900 -- "$O/run.sh" || true
for f in plain steer cap; do [ -s "$O/$f.json" ] || { echo "FAIL http $f: no response: $(tail -2 "$O/serve.log" | cut -c1-200)"; exit 1; }; done
python3 -I - "$O" "$L" "$NP" <<'PYEOF'
import sys, json, numpy as np
O, L, NP = sys.argv[1], sys.argv[2], int(sys.argv[3])
top = lambda n: {d["id"]: d["logit"] for d in json.load(open(f"{O}/{n}.json"))["logits_topk"][0]}
plain, steer = top("plain"), top("steer")
row = np.fromfile(f"{O}/row/row-0.bin", dtype=np.float32)
bad = max(abs(row[i] - v) for i, v in steer.items())
if bad > 1e-3: raise SystemExit(f"FAIL steer: HTTP steered logits differ from the engine's direct gain run by {bad:.3e}")
dd = max(abs(steer[i] - plain[i]) for i in steer if i in plain) if any(i in plain for i in steer) else 1.0
if dd <= 1e-3: raise SystemExit(f"FAIL mutant: steered and plain responses differ by only {dd:.3e}")
c = json.load(open(f"{O}/cap.json")); path = c["capture"]["path"]
H = 4096; a = np.fromfile(f"{path}/L{L}.f16", dtype=np.uint16); d = np.fromfile(f"{O}/direct-cap/L{L}.f16", dtype=np.uint16)
ntok = np.fromfile(f"{path}/tokens.i32", dtype=np.int32).size
want = NP + 4 - 1
if ntok != want or a.size != ntok * 3 * H: raise SystemExit(f"FAIL capture: {ntok} records (want {want}), {a.size} values")
if not np.array_equal(a[: NP * 3 * H], d[: NP * 3 * H]): raise SystemExit("FAIL capture: first prompt records differ from the direct engine capture")
print(f"steer: HTTP vs direct max |logit diff| {bad:.3e}; steered vs plain {dd:.3e}; capture {ntok} records at {path}, first {NP} byte-equal to the direct run")
PYEOF
echo "PASS e1-check: steer over HTTP == engine direct gain run, steer acts, capture path returned and prompt records byte-equal"
