#!/usr/bin/env bash
# Gate lens-diff. Mode a = docs/prereg-lensdiff.md (qwythos-enfr, 20 pairs, lowercase F3 markers); mode b = docs/prereg-lensdiffb.md (qwythos-enfr60 pairs 21..61,
# case-insensitive wider markers). Bars: antisymmetry, A vs A zero reply, known-pair counts at layer 23 or 27, centred-sign-flip null, consistency; MUT=1 builds the
# summed mutant, which must FAIL the known-pair bar. CPU, the GGUF is read for W_U. usage: tests/lensdiff.sh [a|b] [MUT]
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
MODE=${1:-b}; MUT=${2:-0}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
SUB=${SUB:-$HOME/Projects/mojo/mojo-subprocess/src}; B=$HOME/Projects/mojo/mojo-baro
Q=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
CAPS=$HOME/Projects/mojo/mojo-neuron/.work/cap
if [ "$MODE" = a ]; then
  CAP=$CAPS/qwythos-enfr; RA=1-20; RB=21-40; LAYERS="15 23 27"
  FR='[" le"," la"," les"," de"," des"," du"," et"," est"," un"," une"," que"," pour"," dans"," qui"," pas"," sur"," avec"," au"," aux"," je"," il"," nous"," vous"]'
  EN='[" the"," of"," and"," is"," to"," that"," with"," for"," in"," are"," was"," on"," at"," by"]'; NORM='.'
else
  CAP=$CAPS/qwythos-enfr60; RA=21-61; RB=82-122; LAYERS="23 27"
  FR='["le","la","les","de","des","du","et","est","un","une","que","pour","dans","qui","pas","sur","avec","au","aux","je","il","elle","nous","vous","ce","cette","ces","mais","ou","où","donc","car","ne","se","sa","son","ses","leur","nos","mes","mon","ma","ton","ta"]'
  EN='["the","of","and","is","to","that","with","for","in","are","was","on","at","by","it","this","these","those","while","although","though","whose","either","i","we","you","he","she","they","could","would","can","will","not","but","or","if","when","as","from","there","their","his","her"]'; NORM='ascii_downcase | ltrimstr(" ")'
fi
for f in "$CAP/meta.json" "$Q"; do [ -e "$f" ] || { echo "FAIL setup: $f missing"; exit 1; }; done
BIN=.work/nx_ld$MUT; D=""; [ "$MUT" = 1 ] && D="-D LENSDIFF_MUT=1"
timeout 290 "$M" build src/nx.mojo $D -I src -I "$GG" -I "$JS" -I "$LG" -I "$SUB" -I "$B/serve" -I "$B" -o "$BIN" 2> "$BIN-build.log" || { grep -A5 -m3 error "$BIN-build.log" | cut -c1-240; echo "FAIL build: nx"; exit 1; }
export NX_RLOG=${NX_RLOG:-off}; O=.work/lensdiff; [ "$MODE" = b ] && O=.work/lensdiff/b; mkdir -p $O
ld() { "$BIN" lens-diff "{\"gguf\":\"$Q\",\"cap\":\"$CAP\",\"layer\":$1,\"a\":\"$2\",\"b\":\"$3\",\"k\":50,\"null\":${4:-20}}"; }
cnt() { jq -r "$2" "$1" | jq -Rs --argjson l "$3" '[split("\n")[] | select(. != "") | ('"$NORM"') | select(. as $t | $l | index($t))] | length'; }  # markers among the lines of a jq selection
ld ${LAYERS%% *} $RA $RA 0 > $O/aa.json  # the summed mutant breaks bars 1 on purpose: skip them there so bar 2 is the one that fails (prereg bar 5)
[ "$MUT" = 1 ] || { [ "$(jq '.zero' $O/aa.json)" = true ] && [ "$(jq '.norm == 0' $O/aa.json)" = true ] && [ "$(jq 'has("lens")' $O/aa.json)" = false ] || { echo "FAIL zero: A vs A is not an exact zero reply without a lens: $(head -c 200 $O/aa.json)"; exit 1; }; }
pass=0
for L in $LAYERS; do
  ld $L $RA $RB > $O/fe$L.json; ld $L $RB $RA 0 > $O/ef$L.json
  jq -r '.lens.top[][0]' $O/fe$L.json > $O/fe$L.top; jq -r '.lens.bottom[][0]' $O/ef$L.json > $O/ef$L.bot
  [ "$MUT" = 1 ] || cmp -s $O/fe$L.top $O/ef$L.bot || { echo "FAIL antisymmetry: L$L top of FR-EN differs from bottom of EN-FR"; exit 1; }
  ft=$(cnt $O/fe$L.json '.lens.top[][0]' "$FR"); et=$(cnt $O/fe$L.json '.lens.top[][0]' "$EN")
  fb=$(cnt $O/fe$L.json '.lens.bottom[][0]' "$FR"); eb=$(cnt $O/fe$L.json '.lens.bottom[][0]' "$EN")
  cons=$(jq -r '.consistency' $O/fe$L.json); nrm=$(jq -r '.norm' $O/fe$L.json)
  nn=$(jq '.null_top | length' $O/fe$L.json); [ "$nn" = 20 ] || { echo "FAIL null: $nn null lenses, want 20"; exit 1; }
  nc=$(for i in $(seq 0 19); do cnt $O/fe$L.json ".null_top[$i][]" "$FR"; done | awk '{ s += $1; if ($1 > mx) mx = $1 } END { printf "%.2f %d", s / NR, mx }')
  echo "L$L FR-EN top: fr $ft en $et | bottom: fr $fb en $eb | consistency $cons norm $nrm | null fr-markers mean/max: $nc"
  if [ "$ft" -ge 4 ] && [ "$et" -le 1 ] && [ "$eb" -ge 4 ] && [ "$fb" -le 1 ] && [ "$L" != 15 ]; then
    [ "$(awk -v m="${nc% *}" -v x="${nc#* }" -v c="$cons" 'BEGIN { print (m <= 1.0 && x <= 3 && c >= 0.9) ? 1 : 0 }')" = 1 ] && pass=1
  fi
done
if [ "$MUT" = 1 ]; then
  [ "$pass" = 0 ] || { echo "FAIL mutant: the summed difference still meets the known-pair bar"; exit 1; }
  echo "PASS lensdiff mutant: the sum of the two means fails the known-pair bar, as it must"; exit 0
fi
[ "$pass" = 1 ] || { echo "FAIL lensdiff($MODE): no layer of $LAYERS meets the prereg counts with null <= 1.0 mean / 3 max and consistency >= 0.9"; exit 1; }
echo "PASS lensdiff($MODE): antisymmetric, A vs A zero, FR - EN holds French markers on top and English at the bottom, null and consistency hold"
