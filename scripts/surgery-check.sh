#!/usr/bin/env bash
# Gate A3 (docs/prereg-a3.md): surgery zero of a neuron == engine gain 0, same logits; llama.cpp loads the file.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
SRC=${SRC:-$(python3 -c "import json;print(json.load(open('$PACK/identity.json'))['source'])")}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}; SP=${SUBPROC_SRC:-$HOME/Projects/mojo/mojo-subprocess/src}
L=${LAYER:-16}; N=${NEURON:-$(sed -n 2p .work/scan-check/batch/L16.scan.tsv 2>/dev/null | cut -f1 || true)}; N=${N:-3}
LLAMA=${LLAMA:-$HOME/Models/llama.cpp/build/bin/llama-completion}; [ -x "$LLAMA" ] || LLAMA=$HOME/Models/llama.cpp/build/bin/llama-cli
O=$PWD/.work/surgery-check; rm -rf "$O"; mkdir -p "$O"
[ -x "$ENGINE" ] && [ -f "$SRC" ] || { echo "FAIL setup: engine $ENGINE or source $SRC missing"; exit 1; }
timeout 300 "$M" build src/causal/surgery.mojo -I src -I "$GG" -I "$SP" -I "$LG" -o "$O/surgery" 2> "$O/build.log" || { grep -A5 -m3 error "$O/build.log" | cut -c1-240; echo "FAIL build: surgery"; exit 1; }
NX_RLOG=off "$O/surgery" "$SRC" "$O/z.gguf" layer=$L neuron=$N op=zero > "$O/surgery.txt"; cat "$O/surgery.txt"
ub=$(grep -o 'bytes=[0-9]*' "$O/surgery.txt" | cut -d= -f2); nd=$(cmp -l "$SRC" "$O/z.gguf" | wc -l)
[ "$nd" -gt 0 ] && [ "$nd" -le "$ub" ] && cmp -l "$SRC" "$O/z.gguf" | awk '$3!=0{exit 1}' > /dev/null || { echo "FAIL bytes: $nd differ (row has $ub), or a written byte is nonzero"; exit 1; }
# bake the pack from the surgery file (CPU)
cat > "$O/bake.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
export PATH='$PATH' HOME='$HOME'
cd '$BARO'; ./.venv/bin/python3 -I tools/engine-pack.py '$O/z.gguf' '$O/pack-z' --q4 > '$O/bake.log' 2>&1
EOS
chmod +x "$O/bake.sh"
gpu-wait run --room cpu --cores 8 --mem 24 --timeout 900 -- "$O/bake.sh" || true
[ -s "$O/pack-z/pack.bin" ] || { echo "FAIL bake: $(tail -2 "$O/bake.log" | cut -c1-200)"; exit 1; }
cp "$PACK/prompt-tokens.txt" "$O/pack-z/" 2>/dev/null || true
P=$(tr -s ' \n' ',' < "$PACK/prompt-tokens.txt" | sed 's/,$//')
cat > "$O/rows.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
export PATH='$PATH' HOME='$HOME'
row() { # row DIR PACK GAIN
  mkdir -p "\$1"; echo "{\"id\":1,\"prompt\":[$P],\"n\":1,\"top_logprobs\":1,\"gain\":\"\$3\"}" > "\$1.req"
  env BARO_SERVE=1 BARO_STEER_SERVE=1 BARO_SPEC=0 BARO_PACK="\$2" BARO_DUMP_LOGITS_DIR="\$1" '$ENGINE' < "\$1.req" > "\$1.log" 2>&1
}
row '$O/abl' '$PACK' $L:$N:0
row '$O/surg' '$O/pack-z' $L:0:1
row '$O/base' '$PACK' $L:0:1
EOS
chmod +x "$O/rows.sh"
gpu-wait run --timeout 600 -- "$O/rows.sh" || true
for d in abl surg base; do [ -f "$O/$d/row-0.bin" ] || { echo "FAIL rows $d: $(grep -iE 'rror' "$O/$d.log" | tail -1 | cut -c1-200)"; exit 1; }; done
python3 -I - "$O" <<'PYEOF'
import sys, numpy as np
O = sys.argv[1]; f = lambda n: np.fromfile(f"{O}/{n}/row-0.bin", dtype=np.float32).astype(np.float64)
a, s, b = f("abl"), f("surg"), f("base")
d1 = np.abs(a - s).max(); d2 = np.abs(a - b).max()
if d1 > 1e-3: raise SystemExit(f"FAIL surgery==engine ablation: max |logit diff| {d1:.3e} > 1e-3")
if not (d2 > 1e-3 and d2 > 10 * d1): raise SystemExit(f"FAIL mutant: ablation vs unablated differs {d2:.3e} (surgery diff {d1:.3e})")
print(f"engine ablation vs surgery pack: max |logit diff| {d1:.3e}; ablated vs unablated {d2:.3e}; argmax equal {a.argmax() == s.argmax()}")
PYEOF
cat > "$O/llama.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME'
exec '$LLAMA' -m '$O/z.gguf' -p "The capital of France is" -n 4 -t 8 --no-warmup -no-cnv > '$O/llama.out' 2> '$O/llama.err'
EOS
chmod +x "$O/llama.sh"
gpu-wait run --room cpu --cores 8 --mem 12 --timeout 900 -- "$O/llama.sh" || true
grep -q "The capital of France is" "$O/llama.out" || { echo "FAIL llama.cpp: $(tail -2 "$O/llama.err" | cut -c1-200)"; exit 1; }
echo "PASS surgery-check: layer $L neuron $N, via=up zero == engine gain 0 on the baked pack, unablated differs, llama.cpp loads and generates"
