#!/usr/bin/env bash
# Gate F10 data side (docs/prereg-f10.md): record one request of a 3-request capture, every frame k equals record k of the original files
# (tools/replay-oracle.py reads the originals); only that request's tokens are in the replay; the shifted-frame mutant fails.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
PY=${PY:-python3}
b() { timeout 280 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" "${@:2}" -o ".work/nx$1" 2> ".work/nx$1.log" || { grep -A5 -m3 error ".work/nx$1.log" | cut -c1-240; echo "FAIL build: nx$1"; exit 1; }; }
b _replay & b _replay_m -D REPLAY_MUT=1 & wait
D=.work/replay-test; rm -rf "$D"; mkdir -p "$D"; export REPLAY_DIR=$PWD/$D/replay NX_RLOG=$PWD/$D/rlog
$PY -I tools/replay-seed.py "$D/cap"
.work/nx_replay replay record "{\"cap\":\"$D/cap\",\"req\":1,\"id\":\"t\"}" | grep -q '"tokens": 7' || { echo "FAIL replay: record should hold 7 tokens"; exit 1; }
[ "$(stat -c %s "$REPLAY_DIR/t/tokens.i32")" = 28 ] || { echo "FAIL replay: tokens.i32 holds other requests"; exit 1; }
for l in 3 5; do $PY -I tools/replay-oracle.py "$PWD/.work/nx_replay" "$D/cap" 1 t $l || exit 1; done
if .work/nx_replay replay frame '{"id":"t","k":7}' >/dev/null 2>&1; then echo "FAIL replay: frame past the end accepted"; exit 1; fi
if $PY -I tools/replay-oracle.py "$PWD/.work/nx_replay_m" "$D/cap" 1 t 3 >/dev/null 2>&1; then echo "FAIL replay: shifted-frame mutant passed the gate"; exit 1; fi
[ "$(grep -c '"tool":"replay"' "$D"/rlog/2*-*.jsonl)" -ge 15 ] || { echo "FAIL replay: Rlog lines missing"; exit 1; }
echo "PASS replay: 7 recorded frames x 2 layers equal the original capture records, only request 1 kept, shifted-frame mutant caught"
