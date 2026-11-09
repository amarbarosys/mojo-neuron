#!/usr/bin/env bash
# C1 on a real engine capture: Qwythos-9B (engine-9b, BARO_CAPTURE layer 16) writes a capture, nx scan's strip equals numpy on it, and the same capture
# renders to a WAV for listening (.work/scan-real/real.wav). Needs the neuron2 engine build (A1); one short GPU job through gpu-wait.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
O=$PWD/.work/scan-real; rm -rf "$O"; mkdir -p "$O/cap"; fail() { echo "FAIL scan-real: $1"; exit 1; }
[ -x "$ENGINE" ] || fail "no engine at $ENGINE"
rm -f .work/nx_real; timeout 400 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_real 2>&1 | grep -E "error" -A4 && fail "build nx" || true
cat > "$O/run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SPEC=0 BARO_CAPTURE='$O/cap' BARO_CAPTURE_LAYERS=16
exec '$ENGINE' > '$O/engine.log' 2>&1
EOS
chmod +x "$O/run.sh"; gpu-wait run --timeout 300 -- "$O/run.sh" || true
[ -s "$O/cap/tokens.i32" ] || fail "no capture: $(grep -iE 'error|abort' "$O/engine.log" | tail -1 | cut -c1-200)"
n=$(( $(stat -c %s "$O/cap/tokens.i32") / 4 ))
want=$(python3 -I tools/scan-oracle.py "$O/cap" "$n"); export NX_RLOG=$O/rlog
got=$(.work/nx_real scan "{\"cap\":\"$O/cap\"}" | sed -n 's/.*"sig": "\([^"]*\)".*/\1/p')
[ "$got" = "$want" ] || fail "strip differs from numpy on the engine capture: $got vs $want"
.work/nx_real sound "{\"cap\":\"$O/cap\",\"out\":\"$O/real.wav\"}" > /dev/null
echo "PASS scan-real: $got on an engine capture of $n tokens (Qwythos-9B L16), WAV $O/real.wav"
