#!/usr/bin/env bash
# Gate CAST (docs/prereg-cast.md). Needs the capture of tests/data/en-fr-60.tsv (French lines 1..61 then English lines 62..122, layers 3,7,...,31) in $CAP and
# the lie capture $LIE (English statements, same layers) for the out-of-domain bar. usage: tests/cast.sh [MUT]  (MUT=1: inverted comparison, bars 1 and 3 must FAIL)
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
MUT=${1:-0}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; P=$HOME/Projects/mojo; B=$P/mojo-baro
INC="-I src -I $P/mojo-gguf/src -I $P/mojo-json/src -I $P/mojo-log/src -I $P/mojo-subprocess/src -I $B/serve -I $B"
CAP=${CAP:-$P/mojo-neuron/.work/cap/qwythos-enfr60}; LIE=${LIE:-$P/mojo-neuron/.work/cap/qwythos-lie}
GGUF=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
for f in "$CAP/meta.json" "$LIE/meta.json" "$GGUF"; do [ -e "$f" ] || { echo "FAIL setup: $f missing"; exit 1; }; done
BIN=.work/nx_cast$MUT; D=""; [ "$MUT" = 1 ] && D="-D CAST_MUT=1"
timeout 290 "$M" build src/nx.mojo $D $INC -o "$BIN" 2> "$BIN-build.log" || { grep -A5 -m3 error "$BIN-build.log" | cut -c1-240; echo "FAIL build: nx"; exit 1; }
export NX_RLOG=${NX_RLOG:-off}; O=.work/cast/run$MUT; rm -rf "$O"; mkdir -p "$O"
"$BIN" cast "{\"op\":\"fit\",\"cap\":\"$CAP\",\"out\":\"$O\"}" > "$O/fit.out"
f=$O/fit.json; L=$(jq .layer $f); TAU=$(jq .tau $f)
echo "chosen layer $L tau $TAU; per layer (cv, test): $(jq -r '[.per_layer[] | "L\(.layer) \(.cv|.*100|round/100) \(.test)"] | join(" | ")' $f)"
miss=0
chk() { # chk NAME CONDITION(jq expr on fit.json) DESCRIPTION
  if jq -e "$2" $f > /dev/null; then echo "ok   bar $1: $3"; else echo "MISS bar $1: $3"; miss=$((miss + 1)); fi; }
chk 1 '.test_accuracy >= 0.95' "held-out accuracy $(jq .test_accuracy $f) (>= 0.95)"
chk 2 '.control_mean <= 0.60' "shuffled-label control mean $(jq .control_mean $f) over $(jq '.control_accuracies|length' $f) seeds (<= 0.60)"
chk 4 '.width_min_accuracy >= 0.90 and .test_auc >= 0.98' "min accuracy over tau +-0.25 range $(jq .width_min_accuracy $f) (>= 0.90), AUC $(jq .test_auc $f) (>= 0.98)"
# bar 3: gate output byte-equal to zeros below tau and to the persona vector at or above it, every test prompt (pairs 42..61: French requests 42..61, English 103..122)
"$BIN" concept "{\"gguf\":\"$GGUF\",\"expr\":\"mean($CAP,$L,1-41) - mean($CAP,$L,62-102)\",\"k\":2,\"out\":\"$O/persona\"}" > /dev/null
PER=$O/persona/concept.f32; head -c "$(stat -c %s $PER)" /dev/zero > $O/zeros.f32
cmp -s $PER $O/zeros.f32 && { echo "FAIL setup: the persona vector is all zeros"; exit 1; }
bad=0; nf=0; nz=0
for r in $(seq 42 61) $(seq 103 122); do
  "$BIN" cast "{\"op\":\"gate\",\"fit\":\"$O\",\"cap\":\"$CAP\",\"req\":$r,\"persona\":\"$PER\",\"out\":\"$O/g.f32\"}" > $O/g.json
  s=$(jq .score $O/g.json)
  if awk -v s="$s" -v t="$TAU" 'BEGIN { exit !(s >= t) }'; then want=$PER; nf=$((nf + 1)); else want=$O/zeros.f32; nz=$((nz + 1)); fi
  cmp -s $O/g.f32 $want || bad=$((bad + 1))
done
if [ "$bad" = 0 ]; then echo "ok   bar 3: all 40 gate outputs byte-equal (persona on $nf, zeros on $nz)"; else echo "MISS bar 3: $bad of 40 gate outputs differ from the decision score >= tau"; miss=$((miss + 1)); fi
# bar 5: English statements of the lie capture must not fire
r=$("$BIN" cast "{\"op\":\"score\",\"fit\":\"$O\",\"cap\":\"$LIE\",\"reqs\":\"1-300\"}"); echo "$r" > $O/ood.json
if [ "$(jq '.fire_rate <= 0.05' $O/ood.json)" = true ]; then echo "ok   bar 5: false-fire rate $(jq .fire_rate $O/ood.json) on 300 English statements (<= 0.05)"; else echo "MISS bar 5: false-fire rate $(jq .fire_rate $O/ood.json) on 300 English statements (<= 0.05)"; miss=$((miss + 1)); fi
if [ "$MUT" = 1 ]; then
  b1=$(jq '.test_accuracy >= 0.95' $f)
  [ "$b1" = false ] && [ "$bad" != 0 ] && { echo "PASS cast mutant: the inverted comparison fails bars 1 and 3, as it must"; exit 0; }
  echo "FAIL cast mutant: inverted comparison still meets bar 1 ($b1) or bar 3 (bad=$bad)"; exit 1
fi
[ "$miss" = 0 ] && echo "PASS cast: bars 1-5 hold" || { echo "FAIL cast: $miss bar(s) missed (lines above)"; exit 1; }
