#!/usr/bin/env bash
# B4 label gate (docs/prereg-b4.md). Run inside the local labeller:
#   gpu-wait run --vram 8 --timeout 2400 -- with-llama --ctx 8192 ~/Models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf -- scripts/b4.sh TC NF K LAYER CAP DIR
# DIR holds pick.txt, ctx_<f>.txt, mid_<f>.txt, p99.tsv (tclab ctx). Writes DIR/summary.tsv and DIR/verdict.txt.
set -euo pipefail
export LC_ALL=C  # this box prints decimals with a comma otherwise and printf rejects awk numbers
cd "$(dirname "$0")/.."
TC=$1; NF=$2; K=$3; L=$4; CAP=$5; D=$6
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}
[ -n "${LLAMA_URL:-}" ] || [ -n "${B4_STUB:-}" ] || { echo "FAIL b4: run under with-llama (LLAMA_URL unset)"; exit 1; }
ask() { if [ -n "${B4_STUB:-}" ]; then case "$1" in *"single lowercase word"*) echo thing;; *"Does the snippet"*) echo yes;; *"mean the same"*) echo no;; *"Write 10"*) for n in 1 2 3 4 5 6 7 8 9 10; do echo "The thing number $n appears here in a sentence."; done;; *) echo "stub label";; esac; return; fi; jq -n --arg p "$1" --argjson n "${2:-60}" --argjson t "${3:-0}" '{messages:[{role:"user",content:$p}],max_tokens:$n,temperature:$t}' |
  curl -s "$LLAMA_URL/chat/completions" -H 'Content-Type: application/json' -d @- | jq -r '.choices[0].message.content' | tr -d '\r'; }
read -ra FS < "$D/pick.txt" || true; N=${#FS[@]}
: > "$D/words.txt"; : > "$D/texts.txt"
declare -a LAB WORD
for i in $(seq 0 $((N-1))); do f=${FS[$i]}
  if [ -s "$D/label_$i.txt" ] && [ -s "$D/word_$i.txt" ] && [ -s "$D/sent_$i.txt" ]; then LAB[$i]=$(cat "$D/label_$i.txt"); WORD[$i]=$(cat "$D/word_$i.txt")  # resume: a finished stage is reused
    echo "${WORD[$i]}" >> "$D/words.txt"; cat "$D/sent_$i.txt" >> "$D/texts.txt"; echo "label $f (cached): ${LAB[$i]} (word ${WORD[$i]})"; continue; fi
  LAB[$i]=$(ask "Below are text snippets in which a neuron of a language model fires on the token between << and >>. Give a short label (at most 6 words) for what the neuron detects. Output only the label.

$(cat "$D/ctx_$f.txt")" 24 | head -1 | tr -d '"')
  WORD[$i]=$(ask "Give ONE single lowercase word, no punctuation, that best names this concept: ${LAB[$i]}" 6 | head -1 | tr -dc 'a-z')
  echo "${WORD[$i]}" >> "$D/words.txt"; echo "${LAB[$i]}" > "$D/label_$i.txt"; echo "${WORD[$i]}" > "$D/word_$i.txt"
  ask "Write 10 short, varied sentences, one per line, no numbering, that clearly involve: ${LAB[$i]}" 400 | grep -v '^$' | head -10 | sed 's/^[0-9]*[.)] *//' > "$D/sent_$i.txt"
  while [ "$(wc -l < "$D/sent_$i.txt")" -lt 10 ]; do echo "${LAB[$i]}" >> "$D/sent_$i.txt"; done
  cat "$D/sent_$i.txt" >> "$D/texts.txt"
  echo "label $f: ${LAB[$i]} (word ${WORD[$i]})"
done
# activations of every picked feature on every label sentence (rows 10*j..10*j+9 = sentences of label j)
.work/tclab act "$TC" "$NF" "$K" "$Q" "$L" "$D/pick.txt" "$D/texts.txt" "$D/act.tsv" >/dev/null
# precision on non-top contexts (ranks 50-500) and on the top contexts: the labeller judges "does the snippet match the label"
yes_frac() { local lab=$1 file=$2 y=0 n=0; while IFS= read -r line; do
    r=$(ask "Label: $lab
Snippet (the key token is between << and >>): $line
Does the snippet match the label? Answer only yes or no." 3 | tr 'A-Z' 'a-z'); n=$((n+1)); case "$r" in yes*) y=$((y+1));; esac; done < "$file"; echo "$y $n"; }
# steer-first
printf '%s\n' "The weather today is" "She walked into the room and" "In the beginning there was" "The meeting is scheduled for" "He looked at the picture and" "My favourite part of the day is" > "$D/prompts.txt"
.work/tclab steer "$TC" "$NF" "$K" "$Q" "$L" "$CAP" "$D/pick.txt" "$D/prompts.txt" "$D/words.txt" "${DOSE:-8}" "$D/steer.tsv" | tail -1
: > "$D/summary.tsv"
auc() { awk -v col="$1" -v pos="$2" -v N="$N" 'BEGIN{FS="\t"} {v[NR]=$col} END{np=0;nn=0;wins=0; for(a=1;a<=10;a++){x=v[pos*10+a]; for(b=1;b<=NR;b++){ if(int((b-1)/10)==pos) continue; y=v[b]; nn_=1; wins+=(x>y)+0.5*(x==y)}} print wins/(10*(NR-10))}' "$D/act.tsv"; }
recall() { awk -v col="$1" -v pos="$2" -v thr="$3" 'BEGIN{FS="\t"} {if(int((NR-1)/10)==pos && $col>thr)c++} END{print c/10+0}' "$D/act.tsv"; }
kept=0; sa=0; ss=0
for i in $(seq 0 $((N-1))); do f=${FS[$i]}
  A=$(auc $((i+1)) $i); S=$(auc $((i+1)) $(( (i+1) % N )))   # real label sentences vs shuffled labels (the next feature's sentences)
  thr=$(awk -v f="$f" 'BEGIN{FS="\t"} $1==f{print $2}' "$D/p99.tsv"); R=$(recall $((i+1)) $i "$thr")
  read -r ym nm < <(yes_frac "${LAB[$i]}" "$D/mid_$f.txt"); read -r yt nt < <(yes_frac "${LAB[$i]}" "$D/ctx_$f.txt")
  P=$(awk -v y=$ym -v n=$nm 'BEGIN{print (n>0)?y/n:0}'); PT=$(awk -v y=$yt -v n=$nt 'BEGIN{print (n>0)?y/n:0}')
  ST=$(awk -v f="$f" 'BEGIN{FS="\t"} $1==f{print $5}' "$D/steer.tsv")
  ok=$(awk -v a=$A -v p=$P -v r=$R -v s=$ST 'BEGIN{print (a>=0.8 && p>=0.5 && r>=0.5 && s=="pass")?1:0}')
  kept=$((kept+ok)); sa=$(awk -v a=$sa -v b=$A 'BEGIN{print a+b}'); ss=$(awk -v a=$ss -v b=$S 'BEGIN{print a+b}')
  printf '%s\t%s\tauroc=%.2f\tshuffled_auroc=%.2f\tprecision_nontop=%.2f\tprecision_top=%.2f\trecall=%.2f\tsteer=%s\tkept=%s\n' "$f" "${LAB[$i]}" "$A" "$S" "$P" "$PT" "$R" "$ST" "$ok" >> "$D/summary.tsv"
done
MA=$(awk -v s=$sa -v n=$N 'BEGIN{printf "%.3f", s/n}'); MS=$(awk -v s=$ss -v n=$N 'BEGIN{printf "%.3f", s/n}')
# flip rate: relabel at temperature 0.7 and let the labeller compare
flip=0; for i in $(seq 0 $((N-1))); do f=${FS[$i]}
  l2=$(ask "Below are text snippets in which a neuron of a language model fires on the token between << and >>. Give a short label (at most 6 words) for what the neuron detects. Output only the label.

$(cat "$D/ctx_$f.txt")" 24 0.7 | head -1 | tr -d '"')
  r=$(ask "Do these two labels mean the same thing? A: ${LAB[$i]} B: $l2. Answer only yes or no." 3 | tr 'A-Z' 'a-z'); case "$r" in yes*) ;; *) flip=$((flip+1));; esac; done
BLIND=$(awk -v a=$MA -v b=$MS 'BEGIN{print (b<=0.60 && a-b>=0.20)?"PASS":"MISSED"}')
{ echo "kept $kept of $N; mean AUROC real $MA shuffled $MS (blind check $BLIND: shuffled <= 0.60 and gap >= 0.20); flip rate $flip of $N; synthetic-unit test: DEFERRED"; } | tee "$D/verdict.txt"
.work/rlogcli b4 normal "$(cat "$D/verdict.txt")" "$D/summary.tsv" "$Q"
[ "$BLIND" = PASS ] || .work/rlogcli b4 anomaly "prereg-b4 blind/shuffle bar missed: real $MA shuffled $MS" "$D/summary.tsv" "$Q"
echo "PASS b4 ran (verdicts above; bars judged in the report)"
