#!/usr/bin/env bash
# Gate: the R-style ridge lens. (1) rlens map = numpy ridge on the same data, beats the identity on held-out windows, 100x-ridge mutant is far;
# (2) lens --rmap passes the f64 brute-force golden with the transported directions; (3) the R readout differs from the logit lens for most neurons;
# (4) the lens.bin-format file loads through the nx lens call and its top token equals the text tool's.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
for t in rlens lens nx; do rm -f .work/$t; "$M" build src/$t.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/$t 2>&1 | grep -E "error" -A4 && { echo "FAIL build: $t"; exit 1; } || true; done
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}; MOD=qwen2.5-0.5b-instruct-q4_k_m
D=$PWD/.work/neurons-rg; rm -rf "$D" .work/ridge
.work/rlens "$Q" tests/corpus.txt --layers 4,12 --windows 36 --holdout 4 --out "$D" --dump .work/ridge | tail -1 | grep -q "^wrote" || { echo "FAIL rlens: fit"; exit 1; }
python3 -I tools/check_ridge.py .work/ridge || exit 1
R=$D/$MOD/rlens.bin
.work/lens "$Q" 12 8 6 --rmap "$R" | tail -1 | grep -q "^PASS" || { echo "FAIL lens --rmap: golden"; exit 1; }
echo "ok   lens --rmap golden + mutant"
.work/lens "$Q" 12 8 > .work/l12.tsv; .work/lens "$Q" 12 8 --rmap "$R" > .work/r12.tsv
diff=$(paste -d'|' <(cut -f2 .work/l12.tsv | cut -d$'\xe2' -f1 | head -400) <(cut -f2 .work/r12.tsv | cut -d$'\xe2' -f1 | head -400) | awk -F'|' '$1!=$2' | wc -l)
[ "$diff" -ge 240 ] || { echo "FAIL rlens: R readout equals the logit lens for $((400 - diff)) of 400 neurons"; exit 1; }
echo "ok   R readout differs from logit lens for $diff of 400 neurons"
.work/lens "$Q" 12 8 --out "$D" --rmap "$R" >/dev/null; .work/lens "$Q" 12 8 --out "$D" >/dev/null
python3 -I tools/check_lens_bin.py "$D/$MOD/12.rlens.bin" 151936 >/dev/null
for n in 5 2000; do
  want=$(awk -F'\t' -v n=$n '$1==n{split($2,a,"\xe2\x90\x9f"); print a[1]}' .work/r12.tsv)
  got=$(MNEURON_DIR=$D .work/nx lens "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":$n,\"kind\":\"r\"}" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["top"][0][0])')
  [ "$want" = "$got" ] || { echo "FAIL lens call kind r: neuron $n '$got' != '$want'"; exit 1; }
done
echo "ok   lens call kind r = text tool"
echo PASS rlens
