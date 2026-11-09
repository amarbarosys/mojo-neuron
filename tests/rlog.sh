#!/usr/bin/env bash
# Gate B0: a normal op writes a normal line, a forced non-finite value writes an anomaly line to both files, NX_RLOG=off writes nothing,
# and a build with the verdict blinded (-D RLOG_MUT=1) fails this gate.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-baro/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
b() { timeout 250 "$M" build src/rlog_check.mojo -I src -I "$GG" -I "$LG" "${@:2}" -o ".work/rlog_check$1" 2> ".work/rlog_check$1.log" || { grep -A5 -m3 error ".work/rlog_check$1.log" | cut -c1-240; echo "FAIL build: rlog_check$1"; exit 1; }; }
b "" & b _m -D RLOG_MUT=1 & wait
t() { local d=.work/rlog-test-$1; rm -rf "$d"; NX_RLOG=$d ".work/rlog_check$1" >/dev/null; echo "$d"; }
d=$(t ""); n=$(grep -c '"kind":"normal"' "$d"/*-*.jsonl); a=$(grep -c '"kind":"anomaly"' "$d"/anomalies.jsonl); da=$(grep -c '"kind":"anomaly"' "$d"/2*.jsonl)
[ "$n" = 1 ] && [ "$a" = 1 ] && [ "$da" = 1 ] || { echo "FAIL rlog: normal=$n anomalies.jsonl=$a dated-anomaly=$da (want 1 1 1)"; exit 1; }
grep -q '"tool":"rlog_check"' "$d"/anomalies.jsonl && grep -q '"git":"[0-9a-f]' "$d"/anomalies.jsonl && grep -q '"run":"2' "$d"/anomalies.jsonl || { echo "FAIL rlog: line lacks tool/git/run"; exit 1; }
rm -rf off; NX_RLOG=off .work/rlog_check >/dev/null; [ ! -e off ] || { rm -rf off; echo "FAIL rlog: NX_RLOG=off wrote a dir"; exit 1; }
d=$(t _m); a=0; [ -f "$d"/anomalies.jsonl ] && a=$(wc -l < "$d"/anomalies.jsonl)
[ "$a" = 0 ] || { echo "FAIL rlog: blinded mutant still wrote an anomaly"; exit 1; }
echo "PASS rlog: normal line, forced non-finite -> anomaly in both files, off writes nothing, blinded verdict build misses it"
