#!/usr/bin/env bash
# Gate F2 (pin half, CPU): a pin validates, the verdict passes at the pinned effect and at the fraction, fails below it, and malformed pins are refused.
# The engine half (re-ablate the top units on a model file, surgery-broken model fails) needs A4 and A3.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
timeout 280 "$M" build src/probe/circuit_check.mojo -I src -I "$GG" -I "$JS" -I "$LG" -o .work/circuit_check 2> .work/circuit_check.log || { grep -A5 -m3 error .work/circuit_check.log | cut -c1-240; echo "FAIL build: circuit_check"; exit 1; }
D=.work/circuit-gate; rm -rf $D; mkdir -p $D
pin() { printf '{"name":"t","model":"m.gguf","prompts":[{"text":"The capital of France is","expect":%s}],"units":%s,"effect":%s,"fraction":0.5}\n' "$1" "$2" "$3" > $D/$4.json; }
U='[{"layer":12,"neuron":7}]'
pin 100 "$U" 2.0 ok; pin 100 '[]' 2.0 nounit; pin 999999 "$U" 2.0 badtok; pin 100 "$U" 0 noeffect
c() { .work/circuit_check $D/$1.json 150000 "$2"; }
[ "$(c ok 2.0)" = PASS ] && [ "$(c ok 1.0)" = PASS ] || { echo "FAIL verdict: the pinned effect or exactly the fraction did not pass"; exit 1; }
[ "$(c ok 0.99)" = FAIL ] && [ "$(c ok 0)" = FAIL ] && [ "$(c ok -1)" = FAIL ] || { echo "FAIL verdict: an effect below the fraction passed (the broken-model case)"; exit 1; }
for b in nounit badtok noeffect; do c $b 2.0 | grep -q '^REFUSED: circuit' || { echo "FAIL refuse: pin '$b' was accepted"; exit 1; }; done
echo "PASS circuit: pin loads, verdict passes at E0 and at the fraction, fails below it and at zero, three malformed pins refused"
