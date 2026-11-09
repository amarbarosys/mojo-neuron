#!/usr/bin/env bash
# Gate: scripted run of the window app (needs a display and the GPU for the UI compose): map -> bucket -> list -> neuron -> fires -> recipe.
# Checks the final STATE line, that every shot exists, and by OCR that the neuron view really shows its panels. A human look at the shots is still the final word.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/shots
scripts/build-app.sh >/dev/null
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}; MOD=qwen2.5-0.5b-instruct-q4_k_m
[ "$(MNEURON_CLIPTEST=1 .work/app)" = "CLIP 日本語~ abc é€😀~" ] || { echo "FAIL app: clip splits or miscounts UTF-8"; exit 1; }
export MNEURON_DIR=$PWD/.work/neurons-app; rm -rf "$MNEURON_DIR"
for l in 4 8 12 16 20; do .work/lens "$Q" $l 8 --out "$MNEURON_DIR" >/dev/null; done
rm -f .work/shots/*.png
out=$(MNEURON_SCRIPT=tests/app_script.txt timeout 180 .work/app 2>&1 | tail -1)
echo "$out" | grep -q '^STATE view=2 layer=12 ' || { echo "FAIL app: end state wrong: $out"; exit 1; }
echo "$out" | grep -q 'dropped=0' || { echo "FAIL app: draw commands dropped: $out"; exit 1; }
echo "$out" | grep -qE 'recipe=12:[0-9]+:\+1\.0,12:[0-9]+:-1\.0 ' || { echo "FAIL app: recipe buttons: $out"; exit 1; }
echo "$out" | grep -q 'status=recipe ok' || { echo "FAIL app: recipe check: $out"; exit 1; }
# steered chat: seed the engine's rows for the neuron the script ends on (with unsteered base rows), rerun, the three columns must show them
sel=$(echo "$out" | sed -n 's/.* sel=\([0-9]*\) .*/\1/p')
python3 -I - "$MNEURON_DIR/$MOD/12.sweep.jsonl" "$sel" <<'PY'
import json, sys
n = int(sys.argv[2])
rows = [("base", 0.0, "Who are you?", "BASELINEWORD says hello"), (n, -0.5, "Who are you?", "NEGATIVEWORD answer"), (n, 0.5, "Who are you?", "POSITIVEWORD answer"),
        ("base", 0.0, "Other prompt", "wrongbase")]
open(sys.argv[1], "w").write("".join(json.dumps({"model": "m", "layer": 12, "neuron": a, "prompt": p, "dose": d, "text": t, "tokens": [1]}) + "\n" for a, d, p, t in rows))
PY
out2=$(MNEURON_SCRIPT=tests/app_script.txt timeout 180 .work/app 2>&1 | tail -1)
echo "$out2" | grep -q ' chat=neg:19,base:23,pos:19 ' || { echo "FAIL app: steered chat columns (want neg:19,base:23,pos:19): $out2"; exit 1; }
for s in map list neuron fires recipe; do [ "$(stat -c %s .work/shots/$s.png 2>/dev/null || echo 0)" -gt 20000 ] || { echo "FAIL app: shot $s missing or empty"; exit 1; }; done
ocr=$(img-read .work/shots/fires.png 2>&1)  # "romotes": the glass theme makes tesseract drop the first glyph of the green label (the glyph is drawn: checked by pixel columns)
for w in "connections" "fires on your text" "romotes" "steered chat" "snake" "BASELINEWORD" "NEGATIVEWORD" "POSITIVEWORD"; do echo "$ocr" | grep -qi "$w" || { echo "FAIL app: OCR of the neuron view lacks '$w'"; exit 1; }; done
ocr=$(img-read .work/shots/map.png 2>&1); echo "$ocr" | grep -qi "map" || { echo "FAIL app: OCR of the map view"; exit 1; }
# live generate: the button runs the engine's sweep for the selected neuron; the three columns must change and show the prompt
E=${ENGINE:-$HOME/Projects/mojo/mojo-neuron-lanes/engine}; BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/ablate}
for f in "$E/.work/sweep" "$BARO/.work/steer/spark-q05" "$E/.work/steer/q05/rho.f32"; do [ -e "$f" ] || { echo "FAIL app: $f missing (live stage needs the engine lane build)"; exit 1; }; done
printf '{"sweep":"%s","gguf":"%s","spark":"%s","pack":"%s","baro":"%s","rho":"%s"}\n' "$E/.work/sweep" "$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf" \
  "$BARO/.work/steer/spark-q05" "$BARO/.work/abl/q/pack" "$BARO" "$E/.work/steer/q05/rho.f32" > "$MNEURON_DIR/$MOD/live.json"
export MNEURON_LEDGER=${LEDGER:-$HOME/Projects/mojo/mojo-neuron/.work/gpu-ledger.md}
out3=$(MNEURON_SCRIPT=tests/app_live_script.txt timeout 180 .work/app 2>&1 | tail -1)
echo "$out3" | grep -q 'status=live sweep done' || { echo "FAIL app: generate button did not run the sweep: $out3"; exit 1; }
echo "$out3" | grep -qE ' chat=neg:[1-9][0-9]*,base:[1-9][0-9]*,pos:[1-9][0-9]* ' || { echo "FAIL app: live columns empty: $out3"; exit 1; }
[ "$(stat -c %s .work/shots/live.png 2>/dev/null || echo 0)" -gt 20000 ] || { echo "FAIL app: live shot missing"; exit 1; }
img-read .work/shots/live.png 2>&1 | grep -qi "who are you" || { echo "FAIL app: OCR of the live shot lacks the prompt"; exit 1; }
echo "PASS app: 5 shots + live generate, end state $(echo "$out" | cut -d' ' -f2-4)"
