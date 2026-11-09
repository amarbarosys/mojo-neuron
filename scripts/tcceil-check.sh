#!/usr/bin/env bash
# Gate tcceil: an empty held-out split (one request) and a capture file shorter than req.i32 must FAIL with a message and a non-zero exit, never print nan; a capture with 20 requests gives a finite FVU.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
M=${MOJO:-$HOME/Projects/mojo/mojo-baro/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}; JS=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}
[ "${SKIP_BUILD:-0}" = 1 ] || timeout 250 "$M" build src/features/tcceil.mojo -I src -I src/features -I "$GG" -I "$LG" -I "$JS" -o .work/tcceil 2> .work/tcceil-build.log || { grep -A5 -m3 error .work/tcceil-build.log | cut -c1-240; echo "FAIL build: tcceil"; exit 1; }
SRC=${CAP:-$HOME/Projects/mojo/mojo-neuron/.work/cap/qwen2.5-0.5b-b1}; [ -s "$SRC/L12.f16" ] || { echo "FAIL setup: no capture at $SRC"; exit 1; }
mk() { local d=.work/tcceil-test-$1; rm -rf "$d"; mkdir -p "$d"; cp "$SRC/meta.json" "$d/"; head -c $(( $2 * 3584 )) "$SRC/L12.f16" > "$d/L12.f16"; head -c $(( $2 * 4 )) "$SRC/req.i32" > "$d/req.i32"; head -c $(( $2 * 4 )) "$SRC/tokens.i32" > "$d/tokens.i32"; echo "$d"; }
d=$(mk one 128)
set +e; out=$(NX_RLOG=off .work/tcceil "$d" 12 2>&1); rc=$?; set -e
{ [ "$rc" != 0 ] && echo "$out" | grep -q "^FAIL tcceil" && ! echo "$out" | grep -qi "nan"; } || { echo "FAIL tcceil-check: empty held-out split gave rc=$rc, output: $(echo "$out" | tail -1 | cut -c1-120)"; exit 1; }
d=$(mk short 128); head -c 100000 "$SRC/L12.f16" > "$d/L12.f16"  # file shorter than req.i32 says
set +e; out=$(NX_RLOG=off .work/tcceil "$d" 12 2>&1); rc=$?; set -e
{ [ "$rc" != 0 ] && echo "$out" | grep -q "^FAIL tcceil"; } || { echo "FAIL tcceil-check: truncated capture file gave rc=$rc, output: $(echo "$out" | tail -1 | cut -c1-120)"; exit 1; }
d=$(mk many 2560)
out=$(NX_RLOG=off .work/tcceil "$d" 12 2>&1 | tail -1)
echo "$out" | grep -q "heldout_fvu=[0-9]" && ! echo "$out" | grep -qi "nan" || { echo "FAIL tcceil-check: 20 requests gave: $out"; exit 1; }
echo "PASS tcceil-check: empty held-out split and a truncated file fail loudly, 20 requests give a finite FVU"
