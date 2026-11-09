#!/usr/bin/env bash
# Gate F30 (docs/prereg-f30.md): planted notes -> `nx wiki` forward writes links only for the known topic (not the nonsense mutant, not the
# real junk label that fooled a 4.0 threshold), is idempotent, and the reverse query finds the known unit and neither junk unit.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
timeout 280 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_wiki 2> .work/nx_wiki.log || { grep -A5 -m3 error .work/nx_wiki.log | cut -c1-240; echo "FAIL build: nx_wiki"; exit 1; }
D=.work/wiki-test; rm -rf "$D"; mkdir -p "$D"; MOD=wikitest
export MNEURON_BRAIN=$PWD/$D/brain MNEURON_DIR=$PWD/$D/neurons NX_RLOG=$PWD/$D/rlog
python3 -I tools/wiki-seed.py "$MNEURON_BRAIN" $MOD
nx() { .work/nx_wiki "$@"; }
for i in 1 2; do nx wiki "{\"model\":\"$MOD\"}" > "$D/fwd$i.json"; done
N=$MNEURON_BRAIN/$MOD
grep -q '"units": 3' "$D/fwd2.json" || { echo "FAIL wiki: expected 3 units"; exit 1; }
[ "$(grep -c '^## Wikipedia' "$N/L30-N1.md")" = 1 ] || { echo "FAIL wiki: known unit wants exactly one Wikipedia section after two runs"; exit 1; }
grep -E -q '^- \[\[wiki/(Photosynthesis|Chlorophyll)\]\]' "$N/L30-N1.md" || { echo "FAIL wiki: known unit has no Photosynthesis/Chlorophyll link"; cat "$N/L30-N1.md" | tail -6; exit 1; }
for f in L30-N2 L12-N1933; do grep -q 'Wikipedia' "$N/$f.md" && { echo "FAIL wiki: $f got a link"; cat "$N/$f.md" | tail -5; exit 1; }; done
python3 -I tools/wiki-check.py "$MNEURON_DIR/$MOD/wiki.jsonl" || exit 1
nx wiki '{"query":"photosynthesis"}' > "$D/rev.json"
python3 -I tools/wiki-check.py "$D/rev.json" reverse || exit 1
[ "$(grep -c '"tool":"wiki"' "$D"/rlog/2*-*.jsonl)" = 3 ] || { echo "FAIL wiki: want 3 Rlog lines (2 forward, 1 reverse)"; exit 1; }
echo "PASS wiki: known unit linked (one section after 2 runs), nonsense and the 4.18 junk label unlinked below 4.5, reverse finds the known unit only"
