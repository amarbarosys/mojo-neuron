#!/usr/bin/env bash
# Capture the residual (field r) of every line of TEXTS through the baro engine (serve mode, one request per line, n=1) into CAP.
# Request i of the capture = line i of TEXTS. Work files go to WORK. Env: JUDGE=1 wraps each text as a true/false question and asks for logits_topk 20
# instead of a capture (NOCAP, no capture dir written). usage: cap-texts.sh TEXTS CAP WORK [LAYERS=3,7,11,15,19,23,27,31]
set -euo pipefail
TEXTS=${1:?usage: cap-texts.sh TEXTS CAP WORK [LAYERS]}; CAP=${2:?}; OUT=${3:?}; LAYERS=${4:-${LAYERS:-3,7,11,15,19,23,27,31}}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}
PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
TOK=$HOME/Models/llama.cpp/build/bin/llama-tokenize; export LD_LIBRARY_PATH=$HOME/Models/llama.cpp/build/bin GG TOK
for f in "$ENGINE" "$PACK/pack.bin" "$GG" "$TOK" "$TEXTS"; do [ -e "$f" ] || { echo "FAIL setup: $f missing"; exit 1; }; done
mkdir -p "$OUT"; rm -rf "$CAP"; mkdir -p "$CAP"
tok1() { local t="$1"; [ "${JUDGE:-0}" = 1 ] && t="Statement: $1"$'\n'"Is this statement true or false?"$'\n'"Answer:"; local ids; ids=$("$TOK" -m "$GG" --ids --log-disable -p "$t" 2>/dev/null | tail -1 | tr -d '[] ') || true; printf '%s\n' "$ids"; }
export -f tok1
# xargs -P does not keep order: number the lines, tokenize in parallel, sort back
nl -ba -w1 -s$'\t' "$TEXTS" | xargs -d '\n' -P 8 -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r i t <<< "$1"; printf "%s\t%s\n" "$i" "$(tok1 "$t")"' _ {} | sort -n > "$OUT/ids.tsv"
[ "$(wc -l < "$OUT/ids.tsv")" = "$(wc -l < "$TEXTS")" ] || { echo "FAIL tokenize: $(wc -l < "$OUT/ids.tsv") of $(wc -l < "$TEXTS") texts tokenized"; exit 1; }
awk -F'\t' '$2=="" { bad++ } END { exit bad>0 }' "$OUT/ids.tsv" || { echo "FAIL tokenize: an empty line in $OUT/ids.tsv (a text the tokenizer rejected)"; exit 1; }
awk -F'\t' -v extra="$([ "${JUDGE:-0}" = 1 ] && echo ',"logits_topk":20')" '{ printf "{\"id\":%d,\"prompt\":[%s],\"n\":1%s}\n", $1, $2, extra }' "$OUT/ids.tsv" > "$OUT/req.jsonl"
cat > "$OUT/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SERVE=1 BARO_SPEC=0 $([ "${JUDGE:-0}" = 1 ] || echo "BARO_CAPTURE='$CAP' BARO_CAPTURE_LAYERS='$LAYERS' BARO_CAPTURE_FIELDS=r BARO_CAPTURE_MODEL=qwythos-9b")
exec '$ENGINE' < '$OUT/req.jsonl' > '$OUT/engine.out' 2> '$OUT/engine.err'
EOS
chmod +x "$OUT/run.sh"
echo "ok   $(wc -l < "$OUT/req.jsonl") requests; engine under gpu-wait (capture -> $CAP)"
gpu-wait run --label "${LABEL:-cap-texts}" --timeout "${TIMEOUT:-1500}" -- "$OUT/run.sh" || true
n=$(( $(stat -c %s "$CAP/req.i32" 2>/dev/null || echo 0) / 4 ))
if [ "${JUDGE:-0}" = 1 ]; then n=$(grep -c '"logits_topk"' "$OUT/engine.out" || true); fi
[ "$n" -gt 0 ] || { echo "FAIL run: no capture; $(grep -iE 'error|abort' "$OUT/engine.err" | tail -1 | cut -c1-200)"; exit 1; }
d=$(grep -c '"done":true' "$OUT/engine.out" || true); e=$(grep -c '"error"' "$OUT/engine.out" || true); N=$(wc -l < "$OUT/req.jsonl")
[ "$e" = 0 ] && [ "$d" = "$N" ] || { echo "FAIL run: $d of $N requests done, $e errors: $(grep -m1 '"error"' "$OUT/engine.out" | cut -c1-200)"; exit 1; }
echo "ok   capture: $n records, $d requests done"
