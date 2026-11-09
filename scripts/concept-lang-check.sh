#!/usr/bin/env bash
# F3 known pair (docs/prereg-f3.md): mean(French) - mean(English) lens at layers 23 and 27 of Qwythos-9B holds French function words on top and
# English at the bottom; the swapped expression flips both. Needs the capture of tests/data/en-fr.tsv (cap-texts.sh, 20 FR then 20 EN, layers 15,23,27).
set -euo pipefail
cd "$(dirname "$0")/.."
CAP=${CAP:-$HOME/Projects/mojo/mojo-neuron/.work/cap/qwythos-enfr}
GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
for f in .work/nx "$CAP/meta.json" "$GG"; do [ -e "$f" ] || { echo "FAIL setup: $f missing"; exit 1; }; done
export NX_RLOG=${NX_RLOG:-}; D=.work/concept-lang; mkdir -p $D
FR=" le la les de des du et est un une que pour dans qui pas sur avec au aux je il nous vous"; EN=" the of and is to that with for in are was on at by"
count() { # count FILE POLE LIST -> number of tokens of the pole list that are markers (display text with its leading space)
  jq -r ".layers[0].lens.$2[][0]" "$1" | awk -v list="$3" 'BEGIN { n = split(list, a, " "); for (i = 1; i <= n; i++) m[" " a[i]] = 1 } ($0 in m) { c++ } END { print c + 0 }'; }
nx() { .work/nx concept "{\"gguf\":\"$GG\",\"expr\":\"$2\",\"k\":50,\"out\":\"$D/$1\"}"; }
pass=0
for L in 23 27; do
  nx fe$L "mean($CAP,$L,1-20) - mean($CAP,$L,21-40)" > $D/fe$L.json; nx ef$L "mean($CAP,$L,21-40) - mean($CAP,$L,1-20)" > $D/ef$L.json
  ft=$(count $D/fe$L.json top "$FR"); et=$(count $D/fe$L.json top "$EN"); fb=$(count $D/fe$L.json bottom "$FR"); eb=$(count $D/fe$L.json bottom "$EN")
  mft=$(count $D/ef$L.json top "$FR"); met=$(count $D/ef$L.json top "$EN")
  echo "L$L FR-EN top: fr $ft en $et | bottom: fr $fb en $eb | swapped top: fr $mft en $met"
  [ "$ft" -ge 4 ] && [ "$et" -le 1 ] && [ "$eb" -ge 4 ] && [ "$fb" -le 1 ] && [ "$met" -ge 4 ] && [ "$mft" -le 1 ] && pass=1
done
[ "$pass" = 1 ] || { echo "FAIL lang: no layer of 23, 27 meets the prereg counts"; exit 1; }
echo "PASS concept-lang: French minus English lens holds French function words on top and English at the bottom, the swapped expression flips both"
