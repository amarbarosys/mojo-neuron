#!/usr/bin/env bash
# Gate tuned lens (docs/prereg-tlensb.md). Bar 4 + its mutant: the gradient oracle passes and the build with a broken rmsnorm backward (-D TCORE_MUT=1) fails it.
# Bars 0-3 from RUN/result.json of `nx tlens` (default .work/tlens/r1, fit by: nx tlens '{"gguf":..,"cap":".../qwythos-lie","out":".work/tlens/r1"}').
# Bar 5: a fit against shuffled teacher rows (-D TLENS_MUT=1, same lr) must fail bar 2; run once into .work/tlens/mut if missing. usage: tests/tlens.sh [RUN]
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
RUN=${1:-.work/tlens/r1}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; P=$HOME/Projects/mojo; B=$P/mojo-baro
INC="-I src -I $P/mojo-gguf/src -I $P/mojo-json/src -I $P/mojo-log/src -I $P/mojo-subprocess/src -I $B/serve -I $B"
GGUF=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
CAP=${CAP:-$P/mojo-neuron/.work/cap/qwythos-lie}
timeout 250 "$M" build src/lens2/tcore_check.mojo $INC -o .work/tcore_check 2> .work/tcore_check.log || { grep -A5 -m3 error .work/tcore_check.log | cut -c1-240; echo "FAIL build: tcore_check"; exit 1; }
timeout 250 "$M" build src/lens2/tcore_check.mojo -D TCORE_MUT=1 $INC -o .work/tcore_check_m 2> .work/tcore_check_m.log || { echo "FAIL build: tcore_check mutant"; exit 1; }
out=$(.work/tcore_check) || { echo "FAIL bar 4: gradient oracle: $(echo "$out" | tail -2 | tr '\n' ' ')"; exit 1; }
if mo=$(.work/tcore_check_m 2>&1); then echo "FAIL bar 4 mutant: the oracle passes with a broken rmsnorm backward"; exit 1; fi
echo "ok   bar 4: $(echo "$out" | tail -2 | head -1); mutant (rmsnorm mean term dropped) fails it"
[ -s "$RUN/result.json" ] || { echo "FAIL setup: $RUN/result.json missing: fit first (12 min CPU): nx tlens '{\"gguf\":\"$GGUF\",\"cap\":\"$CAP\",\"out\":\"$RUN\"}', or pass the path of an existing fit"; exit 1; }
jq -e '.layers | length == 6' "$RUN/result.json" >/dev/null || { echo "FAIL setup: $RUN does not hold the 6 prereg layers"; exit 1; }
bars() { # bars RESULT -> prints the per-bar lines and exits 0 when bars 0-3 hold
  jq -r '
    . as $r | [$r.layers[] | select(.kl_tuned <= 0.8 * .kl_lens)] as $b2 | [$r.layers[] | select(.top1_tuned >= .top1_lens)] as $t2 | [$r.layers[] | select(.kl_tuned <= 0.9 * .kl_bonly)] as $b3 |
    "bar 0 teacher next-token accuracy \($r.teacher_top1_text_acc) (>= 0.25): \(if $r.teacher_top1_text_acc >= 0.25 then "ok" else "MISS" end)",
    "bar 1 identity KL at layer 31 \($r.identity_kl_l31) (< 1e-4): \(if ($r.identity_kl_l31|fabs) < 1e-4 then "ok" else "MISS" end)",
    "bar 2 KL tuned <= 0.8 x lens at \($b2|length) of 6 (>= 4): \(if ($b2|length) >= 4 then "ok" else "MISS" end); top-1 not lower at \($t2|length) of 6 (>= 5): \(if ($t2|length) >= 5 then "ok" else "MISS" end)",
    "bar 3 KL tuned <= 0.9 x b-only at \($b3|length) of 6 (>= 4): \(if ($b3|length) >= 4 then "ok" else "MISS" end)",
    ($r.layers[] | "  L\(.layer): lens \(.kl_lens|.*1000|round/1000) b-only \(.kl_bonly|.*1000|round/1000) tuned \(.kl_tuned|.*1000|round/1000) | top-1 \(.top1_lens|.*1000|round/1000) \(.top1_bonly|.*1000|round/1000) \(.top1_tuned|.*1000|round/1000) | train \(.train_kl_tuned|.*1000|round/1000) val \(.val_kl_tuned|.*1000|round/1000) epoch \(.epoch_tuned)")' "$1"
  jq -e '.teacher_top1_text_acc >= 0.25 and ((.identity_kl_l31|fabs) < 1e-4)
         and ([.layers[] | select(.kl_tuned <= 0.8 * .kl_lens)] | length) >= 4 and ([.layers[] | select(.top1_tuned >= .top1_lens)] | length) >= 5
         and ([.layers[] | select(.kl_tuned <= 0.9 * .kl_bonly)] | length) >= 4' "$1" >/dev/null
}
rc=0; bars "$RUN/result.json" || rc=1
echo "S coverage $(jq -r .s_coverage "$RUN/result.json"), lr_a $(jq -r .lr_a "$RUN/result.json"), wall $(jq -r '.wall_s|round' "$RUN/result.json") s"
# bar 5: shuffled-teacher mutant (needs the mutant nx build and one fit)
mkdir -p .work/tlens  # a clean worktree has no .work/tlens (the redirect below failed there)
if [ ! -s .work/tlens/mut/result.json ]; then
  timeout 280 "$M" build src/nx.mojo -D TLENS_MUT=1 $INC -o .work/nx_tl_m 2> .work/nx_tl_m.log || { grep -A5 -m3 error .work/nx_tl_m.log | cut -c1-240; echo "FAIL build: nx tlens mutant"; exit 1; }
  NX_RLOG=off .work/nx_tl_m tlens "{\"gguf\":\"$GGUF\",\"cap\":\"$CAP\",\"out\":\".work/tlens/mut\",\"lr\":$(jq -r .lr_a "$RUN/result.json")}" > .work/tlens/mut.out
fi
if bars .work/tlens/mut/result.json > .work/tlens/mut.bars 2>&1; then echo "FAIL bar 5: the shuffled-teacher fit still meets bars 0-3"; exit 1; fi
n=$(jq '[.layers[] | select(.kl_tuned <= 0.8 * .kl_lens)] | length' .work/tlens/mut/result.json)
[ "$n" -lt 4 ] && echo "ok   bar 5: shuffled-teacher fit beats the logit lens by 0.8 at $n of 6 layers (< 4), so bar 2 fails for it" || { echo "FAIL bar 5: shuffled fit meets bar 2 at $n layers"; exit 1; }
[ "$rc" = 0 ] && echo "PASS tlens: bars 0-5 hold" || { echo "FAIL tlens: a prereg bar is missed on $RUN (lines above)"; exit 1; }
