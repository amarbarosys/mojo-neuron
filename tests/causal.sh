#!/usr/bin/env bash
# Lane causal CPU tests: A2 scan (canned engine response: a normal line, then a forced non-finite KL -> anomaly) and A3 surgery on
# the 0.5B qwen (zero leaves the column within one quantization step of 0, identity keeps it, untouched bytes equal, rlog lines).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/causal-test
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro}; SP=${SUBPROC_SRC:-$HOME/Projects/mojo/mojo-subprocess/src}
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}
T=.work/causal-test; I="-I src -I $GG -I $BARO -I $BARO/serve -I $SP -I $LG"
for t in scan surgery; do timeout 300 "$M" build src/causal/$t.mojo $I -o $T/$t 2> $T/$t.log || { grep -A5 -m3 error $T/$t.log | cut -c1-240; echo "FAIL build: $t"; exit 1; }; done
# A2: 2 prompts, 2 neurons -> requests 1-2 base, 3-6 neuron requests; slens lines on ids 4 and 6
printf 'The cat sat on the mat today.\nRain fell on the quiet town.\n' > $T/p.txt
resp() { { for i in 1 2 3 4 5 6; do echo "{\"id\":$i,\"done\":true}"; done; echo "{\"id\":4,\"slens\":{\"up\":[[1,0.1]],\"dn\":[[2,0.1]],\"kl\":$1,\"dnll\":0.02}}"; echo "{\"id\":6,\"slens\":{\"up\":[[1,0.1]],\"dn\":[[2,0.1]],\"kl\":$2,\"dnll\":-0.01}}"; } > $3; }
resp 0.5 0.01 $T/ok.resp; resp 1e999 0.01 $T/nan.resp
rm -rf $T/rl; export NX_RLOG=$T/rl
$T/scan $Q $T layer=3 neurons=5,9 prompts=$T/p.txt resp=$T/ok.resp > /dev/null
[ "$(sed -n 2p $T/L3.scan.tsv | cut -f1)" = 5 ] || { echo "FAIL scan: ranking (top neuron should be 5)"; exit 1; }
n=$(grep -c '"kind":"normal"' $T/rl/2*.jsonl)
$T/scan $Q $T layer=3 neurons=5,9 prompts=$T/p.txt resp=$T/nan.resp > /dev/null
a=$(grep -c '"kind":"anomaly"' $T/rl/anomalies.jsonl)
[ "$n" = 1 ] && [ "$a" = 1 ] || { echo "FAIL scan rlog: normal=$n anomaly=$a (want 1 1)"; exit 1; }
{ for i in $(seq 1 10); do echo "{\"id\":$i,\"done\":true}"; done; for r in "4 0.5" "6 0.1" "8 0.49" "10 0.01"; do set -- $r; echo "{\"id\":$1,\"slens\":{\"up\":[[1,0.1]],\"dn\":[[2,0.1]],\"kl\":$2,\"dnll\":0.0}}"; done; } > $T/ctl.resp
$T/scan $Q $T layer=3 neurons=5,9 prompts=$T/p.txt ctl=0.99 resp=$T/ctl.resp > /dev/null
[ "$(sed -n 2p $T/L3.scan.tsv | cut -f1)" = 9 ] || { echo "FAIL scan ctl: ranking by excess over the noise-floor pass (top neuron should be 9)"; exit 1; }
if $T/scan $Q $T layer=3 neurons=5,9 prompts=$T/p.txt resp=/dev/null >/dev/null 2>&1; then echo "FAIL scan: an empty engine response was accepted"; exit 1; fi
# A3: zero / identity on qwen 0.5B layer 3 neuron 7
$T/surgery $Q $T/zu.gguf layer=3 neuron=7 op=zero > $T/zu.txt
$T/surgery $Q $T/z.gguf layer=3 neuron=7 op=zero via=down > $T/z.txt; $T/surgery $Q $T/i.gguf layer=3 neuron=7 op=scale k=1 > $T/i.txt
om=$(grep -o 'old_max=[0-9.e-]*' $T/z.txt | cut -d= -f2); nm=$(grep -o 'new_max=[0-9.e-]*' $T/z.txt | cut -d= -f2)
python3 -c "import sys; om,nm=float('$om'),float('$nm'); sys.exit(0 if nm < 0.5*om else 1)" || { echo "FAIL surgery: down-zeroed column max $nm vs original $om"; exit 1; }
d=$(cmp -l $Q $T/z.gguf | wc -l); [ "$d" -gt 0 ] && [ "$d" -lt 200000 ] || { echo "FAIL surgery: $d bytes differ (want a few blocks)"; exit 1; }
ub=$(grep -o 'bytes=[0-9]*' $T/zu.txt | cut -d= -f2); du=$(cmp -l $Q $T/zu.gguf | wc -l); [ "$du" -le "$ub" ] && [ "$du" -gt 0 ] || { echo "FAIL surgery: via=up changed $du bytes, row has $ub"; exit 1; }
cmp -l $Q $T/zu.gguf | awk -v ub=$ub 'NR==1{f=$1} END{if (NR>0 && $1-f >= ub) exit 1}' || { echo "FAIL surgery: via=up bytes outside one row"; exit 1; }
cmp -l $Q $T/zu.gguf | awk '$3!=0{exit 1}' || { echo "FAIL surgery: via=up wrote a nonzero byte"; exit 1; }
grep -q 'col_abs_change=0 ' $T/i.txt || grep -q 'col_abs_change=0.0 ' $T/i.txt || { echo "FAIL surgery: identity scale changed the column: $(cat $T/i.txt | cut -c1-200)"; exit 1; }
[ "$(grep -c '"tool":"surgery"' $T/rl/2*.jsonl)" = 3 ] || { echo "FAIL surgery rlog lines"; exit 1; }
# nx surgery / nx scan calls shell out to the tool binaries
J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}
timeout 600 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -I "$SP" -o $T/nx 2> $T/nx.log || { grep -A5 -m3 error $T/nx.log | cut -c1-240; echo "FAIL build: nx"; exit 1; }
MNEURON_BIN=$T $T/nx surgery "{\"model\":\"$Q\",\"out\":\"$T/n.gguf\",\"layer\":3,\"neuron\":7,\"op\":\"zero\"}" | grep -q "op=zero via=up" || { echo "FAIL nx surgery: no tool output"; exit 1; }
[ -f $T/n.gguf ] && cmp -s $T/n.gguf $T/zu.gguf || { echo "FAIL nx surgery: output differs from the direct tool run"; exit 1; }
mkdir -p $T/stub && printf '#!/bin/sh\necho ABLATE-SCAN-STUB\n' > $T/stub/scan && chmod +x $T/stub/scan
MNEURON_BIN=$T/stub $T/nx ablate-scan "{\"model\":\"$Q\",\"out\":\"$T/s.tsv\",\"layer\":3}" | grep -q ABLATE-SCAN-STUB || { echo "FAIL nx ablate-scan: did not reach the ablation scan tool"; exit 1; }
echo "PASS causal: scan ranking + normal/anomaly rlog + empty response refused; surgery via=up zeroes one ffn_up row ($du bytes, all 0); via=down column max $nm (was $om, K-quant residual), $d bytes differ, identity keeps the column, rlog lines, nx surgery call equals the tool run"
