#!/usr/bin/env bash
# F3 engine gate (docs/prereg-f3.md checks 6 and 7): on Qwythos-9B, `A - A` steers like no steer (top-20 first-step logits equal within one bf16 step,
# same order) and a real concept (French minus English at L23, scale 8) changes them. One engine process (BARO_STEER_SERVE=1), 3 prompts x 3 conditions.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}; CAP=${CAP:-$HOME/Projects/mojo/mojo-neuron/.work/cap/qwythos-enfr}
TOK=$HOME/Models/llama.cpp/build/bin/llama-tokenize; export LD_LIBRARY_PATH=$HOME/Models/llama.cpp/build/bin
for f in "$ENGINE" .work/nx "$CAP/meta.json" "$GG" "$TOK"; do [ -e "$f" ] || { echo "FAIL setup: $f missing"; exit 1; }; done
D=$PWD/.work/concept-engine; export NX_RLOG=${NX_RLOG:-}
if [ "${REUSE:-0}" != 1 ]; then   # REUSE=1 replays the checks on the last engine.out (no card)
rm -rf "$D"; mkdir -p "$D"
.work/nx concept "{\"gguf\":\"$GG\",\"expr\":\"tok:_Paris@15 - tok:_Paris@15\",\"out\":\"$D/zero\",\"k\":3}" > /dev/null
.work/nx concept "{\"gguf\":\"$GG\",\"expr\":\"mean($CAP,23,1-20) - mean($CAP,23,21-40)\",\"scale\":8,\"out\":\"$D/real\",\"k\":3}" > /dev/null
: > "$D/req.jsonl"; p=0
for t in "The capital of France is" "Yesterday I went to the" "def fibonacci(n):"; do
  ids=$("$TOK" -m "$GG" --ids --log-disable -p "$t" 2>/dev/null | tail -1 | tr -d '[] '); [ -n "$ids" ] || { echo "FAIL tokenize: $t"; exit 1; }
  echo "{\"id\":$((p*10+1)),\"prompt\":[$ids],\"n\":1,\"logits_topk\":20}" >> "$D/req.jsonl"
  echo "{\"id\":$((p*10+2)),\"prompt\":[$ids],\"n\":1,\"logits_topk\":20,\"steer_file\":\"$D/zero/concept.f32\"}" >> "$D/req.jsonl"
  echo "{\"id\":$((p*10+3)),\"prompt\":[$ids],\"n\":1,\"logits_topk\":20,\"steer_file\":\"$D/real/concept.f32\"}" >> "$D/req.jsonl"; p=$((p+1)); done
cat > "$D/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SERVE=1 BARO_SPEC=0 BARO_STEER_SERVE=1
exec '$ENGINE' < '$D/req.jsonl' > '$D/engine.out' 2> '$D/engine.err'
EOS
chmod +x "$D/run.sh"; gpu-wait run --label concept-engine --timeout 600 -- "$D/run.sh" || true
fi
n=$(grep -c '"logits_topk"' "$D/engine.out" || true); [ "$n" = 9 ] || { echo "FAIL run: $n of 9 logits_topk lines: $(grep -m1 '"error"' "$D/engine.out" | cut -c1-200) $(grep -iE 'error|abort' "$D/engine.err" | tail -1 | cut -c1-160)"; exit 1; }
top() { grep "^{\"id\":$1," "$D/engine.out" | grep '"logits_topk"' | jq -c '[.logits_topk[] | [.id, .logit]]'; }
for p in 0 1 2; do
  a=$(top $((p*10+1))); z=$(top $((p*10+2))); r=$(top $((p*10+3)))
  jq -n --argjson a "$a" --argjson z "$z" '($a | map(.[0])) == ($z | map(.[0])) and ([range(0; $a | length)] | all(. as $i | (($a[$i][1] - $z[$i][1]) | fabs) <= 0.00390625 * ([($a[$i][1] | fabs), ($z[$i][1] | fabs)] | max)))' | grep -q true \
    || { echo "FAIL zero: prompt $p, A - A moved the top-20 logits beyond one bf16 step or reordered them"; exit 1; }
  jq -n --argjson a "$a" --argjson r "$r" '($a | map(.[0])) != ($r | map(.[0]))' | grep -q true || { echo "FAIL mutant: prompt $p, the real concept left the top-20 order unchanged, check 6 cannot fail"; exit 1; }
done
echo "PASS concept-engine: A - A steers like no steer (top-20 logits within one bf16 step, same order) on 3 prompts, the French-minus-English concept reorders the top 20 on all 3"
