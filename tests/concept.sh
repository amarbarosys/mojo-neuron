#!/usr/bin/env bash
# Gate F3 (CPU half): nx concept on qwen2.5-0.5b. Paris - Berlin puts " Paris" at the top and " Berlin" at the bottom of the lens,
# the swapped expression swaps the poles (the check can fail), A - A is the exact zero vector, A - B + B has the lens of A, and bad
# expressions fail with a message. The engine half (A - A steers like no steer) is scripts/concept-engine-check.sh.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
Q=${QWEN:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-q4_K_M.gguf}
[ -f "$Q" ] || { echo "FAIL model: $Q missing"; exit 1; }
timeout 280 "$M" build src/nx.mojo -I src -I "$GG" -I "$JS" -I "$LG" -o .work/nx 2> .work/nx-build.log || { grep -A5 -m3 error .work/nx-build.log | cut -c1-240; echo "FAIL build: nx"; exit 1; }
export NX_RLOG=off; D=.work/concept-gate; rm -rf $D; mkdir -p $D
c() { .work/nx concept "{\"gguf\":\"$Q\",\"expr\":\"$2\",\"k\":4,\"out\":\"$D/$1\"}"; }
c pb 'tok:_Paris@12 - tok:_Berlin@12' > $D/pb.json; c bp 'tok:_Berlin@12 - tok:_Paris@12' > $D/bp.json
[ "$(jq -r '.layers[0].lens.top[0:3][][0] | select(. == " Paris")' $D/pb.json)" = " Paris" ] || { echo "FAIL lens: ' Paris' not in the top 3 of Paris - Berlin"; exit 1; }
[ "$(jq -r '.layers[0].lens.bottom[0:3][][0] | select(. == " Berlin")' $D/pb.json)" = " Berlin" ] || { echo "FAIL lens: ' Berlin' not in the bottom 3 of Paris - Berlin"; exit 1; }
[ "$(jq -r '.layers[0].lens.top[0:3][][0] | select(. == " Berlin")' $D/bp.json)" = " Berlin" ] || { echo "FAIL mutant: swapped expression did not swap the poles"; exit 1; }
[ "$(jq -r '.layers[0].lens.top[0][0]' $D/pb.json)" != "$(jq -r '.layers[0].lens.top[0][0]' $D/bp.json)" ] || { echo "FAIL mutant: both orders have the same top token"; exit 1; }
c aa 'tok:_Paris@12 - tok:_Paris@12' > $D/aa.json
cmp <(head -c $((24 * 896 * 4)) /dev/zero) $D/aa/concept.f32 || { echo "FAIL zero: A - A is not the exact zero vector"; exit 1; }
[ "$(jq '.layers | length' $D/aa.json)" = 0 ] || { echo "FAIL zero: A - A reports a lens layer"; exit 1; }
c a 'tok:_Paris@12' > $D/a.json; c abb 'tok:_Paris@12 - tok:_Berlin@12 + tok:_Berlin@12' > $D/abb.json
[ "$(jq -c '.layers[0].lens.top[0:3] | map(.[0])' $D/a.json)" = "$(jq -c '.layers[0].lens.top[0:3] | map(.[0])' $D/abb.json)" ] || { echo "FAIL algebra: A - B + B has another lens than A"; exit 1; }
n=$(jq '.layers[0].norm' $D/pb.json); [ "$(jq -n "($n > 1.0) and ($n < 1.5)")" = true ] || { echo "FAIL norm: unit(A) - unit(B) has norm $n, want 1.0..1.5"; exit 1; }
c mix 'tok:_Paris@12 - 2*tok:_Berlin@8' > $D/mix.json
[ "$(jq '.layers | length' $D/mix.json)" = 2 ] || { echo "FAIL layers: one term per layer should touch two layers"; exit 1; }
for bad in 'tok:_Paris@12 -' 'tok:_Paris@12 tok:_Berlin@12' 'tok:zzzqxjvnotatoken@12' 'tok:_Paris@99'; do
  if .work/nx concept "{\"gguf\":\"$Q\",\"expr\":\"$bad\",\"out\":\"$D/bad\"}" > $D/bad.out 2>&1; then echo "FAIL input: '$bad' was accepted"; exit 1; fi
  grep -q '^FAIL nx: concept' $D/bad.out || { echo "FAIL input: '$bad' failed without a concept message: $(head -c 120 $D/bad.out)"; exit 1; }
done
# Qwythos has an MTP head (block 32): the steer file must stay f32[32 * 4096] (the engine refuses 33 rows). CPU lens only, a few seconds.
QW=${QWYTHOS:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}
[ -f "$QW" ] || { echo "FAIL model: $QW missing"; exit 1; }
.work/nx concept "{\"gguf\":\"$QW\",\"expr\":\"tok:_Paris@15 - tok:_Berlin@15\",\"k\":3,\"out\":\"$D/qw\"}" > $D/qw.json
[ "$(stat -c %s $D/qw/concept.f32)" = $((32 * 4096 * 4)) ] && [ "$(jq '.n_layer' $D/qw.json)" = 32 ] || { echo "FAIL layers: Qwythos steer file is $(stat -c %s $D/qw/concept.f32) bytes, want $((32 * 4096 * 4)) (MTP head counted as a layer?)"; exit 1; }
echo "PASS concept: Paris - Berlin lens poles, swapped mutant swaps them, A - A exactly zero, A - B + B = A, two-layer mix, 4 bad expressions rejected, Qwythos steer file is 32 layers"
