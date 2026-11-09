#!/usr/bin/env bash
# Gate C1: the live scan tails a growing capture and its strip equals the capture file. Fixture = synthetic capture in the engine's format
# (tools/mkcap.py); 30 tokens exist when the app starts, 10 more are appended while it runs (after the app wrote its mid state, as the engine does:
# layer file first, then tokens.i32). Checks: nx scan == numpy on 30 and 40 tokens; the app's STATE signature mid-run == numpy(30), at the end == numpy(40),
# the clicked cell opens the right channel; the shot shows both panels. Needs a display and the GPU (UI compose only).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/shots .work/scan
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
PY=${PY:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python}; D=$PWD/.work/scan; fail() { echo "FAIL scan: $1"; exit 1; }
scripts/build-app.sh >/dev/null; cp .work/app $D/app.bin
rm -f .work/nx_scan; timeout 400 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_scan 2>&1 | grep -E "error" -A4 && fail "build nx" || true
[ -x .work/nx_scan ] || fail "no .work/nx_scan"
rm -rf $D/full $D/live $D/rlog $D/*.txt $D/*.flag; $PY tools/mkcap.py $D/full 6 64 40 7
# live dir: the first 30 records
$PY - $D <<'PY'
import sys, numpy as np, shutil, os
D = sys.argv[1]; os.makedirs(D + "/live"); h = 64
for f, sz in (("L6.f16", 30 * 2 * h * 2), ("tokens.i32", 120), ("req.i32", 120)):
    open(D + "/live/" + f, "wb").write(open(D + "/full/" + f, "rb").read()[:sz])
shutil.copy(D + "/full/meta.json", D + "/live/meta.json")
PY
o30=$($PY tools/scan-oracle.py $D/full 30); o40=$($PY tools/scan-oracle.py $D/full 40); ch5=$($PY tools/scan-oracle.py $D/full 40 5)
[ "$o30" != "$o40" ] || fail "oracle cannot tell 30 from 40 tokens"
export NX_RLOG=$D/rlog
[ "$(.work/nx_scan scan "{\"cap\":\"$D/full\"}" | sed -n 's/.*"sig": "\([^"]*\)".*/\1/p')" = "$o40" ] || fail "nx scan differs from numpy on 40 tokens"
[ "$(.work/nx_scan scan "{\"cap\":\"$D/live\"}" | sed -n 's/.*"sig": "\([^"]*\)".*/\1/p')" = "$o30" ] || fail "nx scan differs from numpy on 30 tokens"
echo "ok   nx scan == numpy: $o30 and $o40"
cat > $D/script.txt <<SCR
frame
frame
click 530 24
wait 12
state $D/mid.txt
waitfile $D/grown.flag
wait 30
click 350 130
wait 3
shot $PWD/.work/shots/scan.png
SCR
cat > $D/app.sh <<APP
#!/usr/bin/env bash
export PATH=$PATH HOME=$HOME WAYLAND_DISPLAY=\${WAYLAND_DISPLAY:-wayland-1} DISPLAY=\${DISPLAY:-:0} XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/run/user/1000} MNEURON_DIR=$D/neurons MNEURON_CAP=$D/live NX_RLOG=$D/rlog
cd $PWD; MNEURON_SCRIPT=$D/script.txt timeout 200 $D/app.bin 2>&1 | tail -1
APP
chmod +x $D/app.sh; mkdir -p $D/neurons
# the writer: waits for the app's mid state, then appends records 30..39 like the engine (layer file, then tokens)
( for _ in $(seq 1 900); do [ -e $D/mid.txt ] && break; sleep 0.5; done
  $PY - $D <<'PY'
import sys
D = sys.argv[1]; h = 64; rec = 2 * h * 2
src = open(D + "/full/L6.f16", "rb").read(); tok = open(D + "/full/tokens.i32", "rb").read(); rq = open(D + "/full/req.i32", "rb").read()
for t in range(30, 40):
    open(D + "/live/L6.f16", "ab").write(src[t * rec:(t + 1) * rec]); open(D + "/live/tokens.i32", "ab").write(tok[t * 4:t * 4 + 4]); open(D + "/live/req.i32", "ab").write(rq[t * 4:t * 4 + 4])
open(D + "/grown.flag", "w").write("1")
PY
) &
out=$(gpu-wait run --shared --vram 2 --timeout 400 -- $D/app.sh 2>&1 | grep '^STATE' | tail -1); wait
mid=$(cat $D/mid.txt 2>/dev/null || true)
echo "$mid" | grep -q " scan=$o30 " || fail "mid-run strip is not the 30-token capture ($o30): $mid"
echo "$out" | grep -q " scan=$o40 " || fail "final strip is not the 40-token capture ($o40): $out"
echo "$out" | grep -q " ch=$ch5 " || fail "clicked cell did not open channel $ch5 (the strongest of token 5): $out"
echo "ok   app strip == numpy: 30 tokens mid-run, 40 after the append, clicked cell -> channel $ch5"
grep -q '"tool":"scan"' $D/rlog/2*.jsonl || fail "no rlog line"
[ "$(stat -c %s .work/shots/scan.png 2>/dev/null || echo 0)" -gt 20000 ] || fail "shot missing"
ocr=$(img-read .work/shots/scan.png 2>&1); for w in "live scan" "channel dossier" "strongest tokens"; do echo "$ocr" | grep -qi "$w" || fail "OCR of the scan view lacks '$w'"; done
echo "PASS scan: $o30 -> $o40, channel $ch5, rlog"
