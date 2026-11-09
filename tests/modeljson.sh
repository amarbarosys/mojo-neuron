#!/usr/bin/env bash
# Gate: `lens --out` must merge model.json, never overwrite it: engine fields (rho, rho_src, anything unknown) survive, the lens
# fields refresh, and gguf is stored absolute even when the model path was given relative. Defects it catches (codex A4, A8):
# rho wiped to [], relative gguf path. Mutant: the gate itself must see a wiped rho.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}
R=.work/neurons-mj; rm -rf "$R"; D=$R/qwen2.5-0.5b-instruct-q4_k_m; mkdir -p "$D"
LENS=$PWD/.work/lens; OUT=$PWD/$R
RHO='[1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5]'
printf '{"v":1,"model":"qwen2.5-0.5b-instruct-q4_k_m","gguf":"/stale","arch":"qwen2","n_layer":24,"hidden":896,"ffn":4864,"vocab":151936,"rho":%s,"rho_src":"engine median over 8 prompts","calib":{"n":8}}\n' "$RHO" > "$D/model.json"
(cd "$(dirname "$Q")" && "$LENS" "./$(basename "$Q")" 12 8 --out "$OUT" >/dev/null) || { echo "FAIL modeljson: lens --out"; exit 1; }
check() { python3 -I - "$D/model.json" "$Q" "$RHO" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); q = sys.argv[2]; rho = json.loads(sys.argv[3])
bad = []
if m.get("rho") != rho: bad.append(f"rho {m.get('rho')!r:.40} != engine rho")
if m.get("rho_src") != "engine median over 8 prompts": bad.append("rho_src lost")
if m.get("calib") != {"n": 8}: bad.append("unknown engine field lost")
if m.get("gguf") != q: bad.append(f"gguf {m.get('gguf')!r} is not the absolute path {q!r}")
if (m.get("hidden"), m.get("ffn"), m.get("n_layer")) != (896, 4864, 24): bad.append("lens fields not refreshed")
print("; ".join(bad)); sys.exit(1 if bad else 0)
PY
}
res=$(check) || { echo "FAIL modeljson: $res"; exit 1; }
# mutant: a wiped rho must be seen by the check above
sed -i 's/"rho":\[[^]]*\]/"rho":[]/' "$D/model.json"
if check >/dev/null; then echo "FAIL modeljson: mutant (rho wiped) passed the check"; exit 1; fi
# a model.json that is not JSON must stop the writer, not be replaced
printf 'not json' > "$D/model.json"
if "$LENS" "$Q" 12 8 --out "$OUT" >/dev/null 2>&1; then echo "FAIL modeljson: lens overwrote a corrupt model.json"; exit 1; fi
[ "$(cat "$D/model.json")" = "not json" ] || { echo "FAIL modeljson: corrupt model.json changed"; exit 1; }
echo "PASS modeljson: rho, rho_src and unknown fields survive, gguf absolute, wiped-rho mutant seen, corrupt file kept"
