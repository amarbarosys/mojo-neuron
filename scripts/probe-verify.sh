#!/usr/bin/env bash
# F24 step 2: keep a candidate only when the Wikipedia network backs it (docs/prereg-f24.md): `answer` occurs in the top 3 hits of
# `brain-ask --net wiki "<true statement>"` and `wrong` occurs in none. In: $OUT/cand.tsv (true, answer, wrong). Out: $OUT/pairs.tsv
# (true, false, answer, wrong). Works in chunks and stops at TARGET kept pairs. usage: probe-verify.sh OUT [TARGET=2000]
set -euo pipefail
OUT=${1:?usage: probe-verify.sh OUT [TARGET]}; TARGET=${2:-2000}; PAR=${PAR:-12}; CH=${CHUNK:-240}
[ -s "$OUT/cand.tsv" ] || { echo "FAIL verify: no $OUT/cand.tsv (run probe-gen.sh)"; exit 1; }
: > "$OUT/pairs.tsv"; : > "$OUT/seen.txt"
one() { # one 'true<TAB>answer<TAB>wrong' -> pair line on stdout when backed
  local t a w; IFS=$'\t' read -r t a w <<< "$1"
  local hits; hits=$(brain-ask --net wiki "$t" 2>/dev/null | awk '/^wiki\//{n++} n>=1 && n<=3') || return 0
  grep -qiF -- "$a" <<< "$hits" || return 0
  ! grep -qiF -- "$w" <<< "$hits" || return 0
  printf '%s\t%s\t%s\t%s\n' "$t" "${t/"$a"/"$w"}" "$a" "$w"
}
export -f one
total=$(wc -l < "$OUT/cand.tsv"); i=0
while [ "$i" -lt "$total" ] && [ "$(wc -l < "$OUT/pairs.tsv")" -lt "$TARGET" ]; do
  sed -n "$((i + 1)),$((i + CH))p" "$OUT/cand.tsv" | xargs -d '\n' -P "$PAR" -I{} bash -c 'one "$1"' _ {} >> "$OUT/pairs.tsv" || true
  i=$((i + CH)); echo "verified $((i < total ? i : total))/$total kept $(wc -l < "$OUT/pairs.tsv")" >&2
done
head -n "$TARGET" "$OUT/pairs.tsv" > "$OUT/pairs.tmp" && mv "$OUT/pairs.tmp" "$OUT/pairs.tsv"
echo "ok   kept $(wc -l < "$OUT/pairs.tsv") of $((i < total ? i : total)) candidates -> $OUT/pairs.tsv"
