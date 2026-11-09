# F24b S1: paraphrase every statement of TEST_PAIRS (pair, true, false, answer, wrong) with the local llama-server (port 8088, scripts/probe-server.sh).
# Keeps a rewrite only when it holds the statement's key entity (answer for the true one, wrong for the false one, case-insensitive). Writes OUT/S1.tsv:
# label, pair, text. usage: probe-para.sh TEST_PAIRS OUT
set -euo pipefail
IN=${1:?usage: probe-para.sh TEST_PAIRS OUT}; OUT=${2:?}; PORT=${PORT:-8088}; mkdir -p "$OUT"
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL server: nothing on port $PORT"; exit 1; }
# one statement list: label, pair, key, text
awk -F'\t' '{ printf "1\t%s\t%s\t%s\n0\t%s\t%s\t%s\n", $1, $4, $2, $1, $5, $3 }' "$IN" > "$OUT/S1.src"
: > "$OUT/S1.raw"; N=$(wc -l < "$OUT/S1.src"); B=20; PAR=${PAR:-8}
one() { # one START
  local s=$1 txt; txt=$(sed -n "$s,$((s + B - 1))p" "$OUT/S1.src" | cut -f4)
  local p="Rewrite each sentence below with different wording and sentence structure. Keep the meaning, and keep every name, place, number and key word exactly as written. Output exactly one rewritten sentence per input line, in the same order, no numbering, no other text.
$txt"
  jq -n --arg p "$p" '{messages:[{role:"user",content:$p}],temperature:0.3,max_tokens:1500}' |
    curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @- | jq -r '.choices[0].message.content' > "$OUT/S1.part.$s" || echo "WARN batch $s failed" >&2
}
for s in $(seq 1 $B "$N"); do one "$s" & while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do sleep 0.2; done; done; wait
: > "$OUT/S1.tsv"; kept=0; dropped=0
for s in $(seq 1 $B "$N"); do
  want=$(sed -n "$s,$((s + B - 1))p" "$OUT/S1.src" | wc -l); got=$(grep -c . "$OUT/S1.part.$s" || true)
  if [ "$want" != "$got" ]; then dropped=$((dropped + want)); continue; fi   # line count differs: alignment is unknowable, drop the batch
  paste <(sed -n "$s,$((s + B - 1))p" "$OUT/S1.src") <(grep . "$OUT/S1.part.$s") | while IFS=$'\t' read -r l p k t r; do
    if grep -qiF -- "$k" <<< "$r"; then printf '%s\t%s\t%s\n' "$l" "$p" "$r" >> "$OUT/S1.tsv"; else echo drop >> "$OUT/S1.drop"; fi; done
done
echo "ok   S1 $(wc -l < "$OUT/S1.tsv") of $N paraphrases kept ($dropped dropped with a misaligned batch, $( [ -f "$OUT/S1.drop" ] && wc -l < "$OUT/S1.drop" || echo 0) without the key entity)"
