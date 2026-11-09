#!/usr/bin/env bash
# F24b (docs/prereg-f24b.md): build the six test sets, capture them (one engine job, layers 15,19,23) and run the output baseline (one engine job).
# Needs: .work/f24b/test-pairs.tsv, .work/f24b/S1.tsv (probe-para.sh). Writes $D/all.tsv (set, label, text), $D/cap (capture), $D/judge/engine.out, $D/tokids.txt.
# usage: probe-sets.sh [D=.work/f24b]
set -euo pipefail
cd "$(dirname "$0")/.."
D=${1:-.work/f24b}; GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
CAPD=${CAPD:-$HOME/Projects/mojo/mojo-neuron/.work/cap/qwythos-lie-sets}
for f in "$D/test-pairs.tsv" "$D/S1.tsv"; do [ -s "$f" ] || { echo "FAIL setup: $f missing"; exit 1; }; done
awk -F'\t' -v OFS='\t' '
  { s0 = "S0"; print s0, 1, $2; print s0, 0, $3 }' "$D/test-pairs.tsv" > "$D/all.tsv"
awk -F'\t' -v OFS='\t' '{ print "S1", $1, $3 }' "$D/S1.tsv" >> "$D/all.tsv"
awk -F'\t' -v OFS='\t' '{ print "S2", 1, "As far as I know, " $2; print "S2", 0, "As far as I know, " $3 }' "$D/test-pairs.tsv" >> "$D/all.tsv"
awk -F'\t' -v OFS='\t' '{ print "S3", 1, "It is true that " $2; print "S3", 0, "It is true that " $3 }' "$D/test-pairs.tsv" >> "$D/all.tsv"
awk -F'\t' -v OFS='\t' '{ print "S4", 0, "It is false that " $2; print "S4", 1, "It is false that " $3 }' "$D/test-pairs.tsv" >> "$D/all.tsv"   # inverted: label flipped
awk -F'\t' -v OFS='\t' '{ print "S5", 1, "I strongly believe that " $2; print "S5", 0, "I strongly believe that " $3 }' "$D/test-pairs.tsv" >> "$D/all.tsv"
cut -f3 "$D/all.tsv" > "$D/texts.txt"
awk -F'\t' '{ n[$1]++ } END { for (k in n) print k, n[k] }' "$D/all.tsv" | sort | tr '\n' ' '; echo
# token ids of the answer words (single tokens, no BOS)
TOK=$HOME/Models/llama.cpp/build/bin/llama-tokenize; export LD_LIBRARY_PATH=$HOME/Models/llama.cpp/build/bin
: > "$D/tokids.txt"
for w in " True" " true" " False" " false"; do ids=$("$TOK" -m "$GG" --ids --no-bos --log-disable -p "$w" 2>/dev/null | tail -1 | tr -d '[] ')
  case "$ids" in *,*|"") echo "FAIL tokenize: '$w' is not one token ($ids)"; exit 1;; esac; echo "${w// /_} $ids" >> "$D/tokids.txt"; done
cat "$D/tokids.txt" | tr '\n' ' '; echo
LABEL=probe-sets TIMEOUT=900 scripts/cap-texts.sh "$D/texts.txt" "$CAPD" "$D/capwork" 15,19,23
JUDGE=1 LABEL=probe-judge TIMEOUT=900 scripts/cap-texts.sh "$D/texts.txt" "$D/nocap" "$D/judge"
echo "ok   sets captured in $CAPD, judge output $D/judge/engine.out"
