#!/usr/bin/env bash
# F24 step 1: candidate statements from a local llama.cpp model. Output: $OUT/cand.tsv = true<TAB>answer<TAB>wrong (one per line, deduped).
# The false twin is built here (true with `answer` replaced by `wrong`), never by the model, so twin and statement differ in one span.
# usage: probe-gen.sh OUT [CALLS=240]    needs a llama-server on $PORT (scripts/probe-server.sh starts one under gpu-wait)
set -euo pipefail
OUT=${1:?usage: probe-gen.sh OUT [CALLS]}; CALLS=${2:-240}; PORT=${PORT:-8088}; PAR=${PAR:-8}
mkdir -p "$OUT"; : > "$OUT/cand.raw"
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL server: nothing on port $PORT (run scripts/probe-server.sh)"; exit 1; }
CATS=(rivers mountains "national capitals" "chemical elements" painters composers novels "scientific discoveries" "wild animals" plants planets battles monarchs
  physicists "world cities" countries languages "olympic sports" films "human anatomy" diseases minerals currencies universities bridges islands lakes
  philosophers "programming languages" companies treaties architects poets mathematicians "musical instruments" deserts seas "famous ships" dinosaurs vitamins)
LET=(A B C D E F G H I K L M N P R S T V W)
one() { # one CALLIDX
  local i=$1 c="${CATS[$(( $1 % ${#CATS[@]} ))]}" l="${LET[$(( ($1 / ${#CATS[@]} + $1 * 7) % ${#LET[@]} ))]}"
  local p="Write 25 different short, well-known, true encyclopedic statements about $c (prefer subjects starting with the letter $l). Each statement is one sentence of 6 to 14 words ending in a period. Its ANSWER is the last key fact: a proper name, place, year, number or one common noun of at most 3 words, never the subject and never the whole sentence. WRONG is a plausible but incorrect replacement of the same kind. Output one line per statement in the form STATEMENT | ANSWER | WRONG, no numbering, no other text. Examples:
The capital of Australia is Canberra. | Canberra | Sydney
Marie Curie won her first Nobel Prize in 1903. | 1903 | 1921
Mount Kilimanjaro is located in Tanzania. | Tanzania | Kenya
Water boils at 100 degrees Celsius at sea level. | 100 | 90"
  jq -n --arg p "$p" --argjson s "$i" '{messages:[{role:"user",content:$p}],temperature:0.8,top_p:0.95,max_tokens:1500,seed:$s}' |
    curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @- | jq -r '.choices[0].message.content' >> "$OUT/cand.raw" || echo "WARN call $i failed" >&2
}
export -f one; export OUT PORT; export -a CATS LET 2>/dev/null || true
for i in $(seq 1 "$CALLS"); do one "$i" & while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do sleep 0.2; done; done; wait
# keep well-formed lines: 3 fields, answer occurs exactly once in the statement, wrong differs from answer, statement ends with a period
awk -F' \\| ' 'NF==3 { s=$1; a=$2; w=$3; gsub(/^[ \t]+|[ \t]+$/,"",s); gsub(/^[ \t]+|[ \t]+$/,"",a); gsub(/^[ \t]+|[ \t]+$/,"",w);
  if (s ~ /\.$/ && length(a) <= 28 && split(a, aw, " ") <= 3 && a != s && index(s, a) > 6 && a != "" && w != "" && tolower(a) != tolower(w) && !(tolower(s) in seen)) {
    n=0; t=s; while ((i=index(t,a))>0) { n++; t=substr(t,i+length(a)) }
    if (n==1) { seen[tolower(s)]=1; print s "\t" a "\t" w } } }' "$OUT/cand.raw" > "$OUT/cand.tsv"
echo "ok   $(wc -l < "$OUT/cand.tsv") candidates from $(wc -l < "$OUT/cand.raw") raw lines -> $OUT/cand.tsv"
