#!/usr/bin/env bash
# Gate F9 (docs/prereg-f9.md): seeded anomalies.jsonl with 3 planted clusters -> nx radar reports those 3 (tool kind unit n), the singleton is
# isolated, the other-week event excluded, digest + Brain note written, an Rlog line appended; the tool-only mutant fails the gate.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
b() { timeout 280 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" "${@:2}" -o ".work/nx$1" 2> ".work/nx$1.log" || { grep -A5 -m3 error ".work/nx$1.log" | cut -c1-240; echo "FAIL build: nx$1"; exit 1; }; }
b _radar & b _radar_m -D RADAR_MUT=1 & wait
D=.work/radar-test; rm -rf "$D"; mkdir -p "$D"
python3 -I tools/radar-seed.py "$D/anomalies.jsonl" | sort > "$D/want.txt"
run() { NX_RLOG="$D/rlog" .work/nx_radar$1 radar "{\"file\":\"$D/anomalies.jsonl\",\"week\":\"2026-W46\",\"report\":\"$D/radar-2026-W46.md\",\"brain\":\"$D/brain\"}"; }
got() { python3 -I tools/radar-got.py; }
run "" | got > "$D/got.txt" || { echo "FAIL radar: output shape (events 14, isolated 2)"; exit 1; }
diff "$D/want.txt" "$D/got.txt" || { echo "FAIL radar: clusters differ from planted"; exit 1; }
[ "$(grep -c '^|' "$D/radar-2026-W46.md")" = 5 ] && [ "$(grep -c 'nan \\| non-finite' "$D/radar-2026-W46.md")" = 2 ] || { echo "FAIL radar: digest should hold header + sep + 3 rows"; exit 1; }
ls "$D"/brain/mojo-neuron/*-radar-2026-W46.md >/dev/null || { echo "FAIL radar: no Brain note"; exit 1; }
grep -q '"tool":"radar"' "$D"/rlog/*-*.jsonl || { echo "FAIL radar: no Rlog line"; exit 1; }
# default week = the latest event's week (W46), not W45
NX_RLOG="$D/rlog" .work/nx_radar radar "{\"file\":\"$D/anomalies.jsonl\"}" | grep -q "\"week\": \"2026-W46\"" || { echo "FAIL radar: default week"; exit 1; }
# the mutant must miss
if run _m | got 2>/dev/null | diff -q "$D/want.txt" - >/dev/null 2>&1; then echo "FAIL radar: tool-only mutant passed the gate"; exit 1; fi
echo "PASS radar: 3 planted clusters (time split, singleton isolated, other week excluded), digest + Brain note + Rlog, tool-only mutant caught"
