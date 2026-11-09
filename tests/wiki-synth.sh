#!/usr/bin/env bash
# Gate F30b section A (docs/prereg-f30b.md): the labeller on planted units with known labels: recall, null units, flip rate under perturbation,
# injected note text; a whole-note labeller (-D WIKI_MUT=1) must fail the injection check.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
b() { timeout 280 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" "${@:2}" -o ".work/nx$1" 2> ".work/nx$1.log" || { grep -A5 -m3 error ".work/nx$1.log" | cut -c1-240; echo "FAIL build: nx$1"; exit 1; }; }
b _wiki & b _wiki_m -D WIKI_MUT=1 & wait
D=$PWD/.work/wiki-synth; rm -rf "$D"; mkdir -p "$D"; export MNEURON_DIR=$D/neurons NX_RLOG=$D/rlog
python3 -I tools/wiki-synth.py seed "$D/notes"
for s in clean pert inj; do .work/nx_wiki wiki "{\"dir\":\"$D/notes/$s\",\"model\":\"$s\"}" >/dev/null & done; wait
.work/nx_wiki_m wiki "{\"dir\":\"$D/notes/injm\",\"model\":\"injm\"}" >/dev/null
python3 -I tools/wiki-synth.py check "$D"
echo "PASS wiki-synth: labeller recall, null units, perturbation flip rate and injection bars held, whole-note mutant caught"
