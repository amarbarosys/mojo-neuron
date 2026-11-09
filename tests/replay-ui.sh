#!/usr/bin/env bash
# Gate F10 scrubber (docs/prereg-f10.md): record request 1 of a 3-request capture, open the replay dir in the app's scan view, scrub to token 3 and 5:
# the strip the app shows equals record k of the ORIGINAL capture (token, channels, f16 values), "play" renders that frame's WAV and its bytes equal
# `nx sound` of the original capture at that record (start=i, tokens=1); frames 3 and 5 differ. Needs a display and the GPU (UI compose only).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/shots
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
D=$PWD/.work/replay-ui; fail() { echo "FAIL replay-ui: $1"; exit 1; }
scripts/build-app.sh >/dev/null; rm -rf "$D"; mkdir -p "$D/neurons"; cp .work/app "$D/app.bin"
timeout 400 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_rui 2>&1 | grep -E "error" -A4 && fail "build nx" || true
[ -x .work/nx_rui ] || fail "no .work/nx_rui"
export NX_RLOG=$D/rlog REPLAY_DIR=$D/replay
python3 -I tools/replay-seed.py "$D/cap"
.work/nx_rui replay record "{\"cap\":\"$D/cap\",\"req\":1,\"id\":\"t\"}" >/dev/null
cat > $D/script.txt <<SCR
frame
frame
click 530 24
wait 12
scrub 3
state $D/s3.txt
play
scrub 5
state $D/s5.txt
play
wait 3
shot $PWD/.work/shots/replay.png
SCR
cat > $D/app.sh <<APP
#!/usr/bin/env bash
export PATH=$PATH HOME=$HOME WAYLAND_DISPLAY=\${WAYLAND_DISPLAY:-wayland-1} DISPLAY=\${DISPLAY:-:0} XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/run/user/1000} MNEURON_DIR=$D/neurons MNEURON_CAP=$D/replay/t NX_RLOG=$D/rlog
cd $PWD; MNEURON_SCRIPT=$D/script.txt timeout 200 $D/app.bin 2>&1 | tail -1
APP
chmod +x $D/app.sh
out=$(gpu-wait run --shared --vram 2 --timeout 400 -- $D/app.sh 2>&1 | grep '^STATE' | tail -1)
echo "$out" | grep -q " rp=5:" || fail "final state has no scrubbed frame 5: $out"
i3=$(python3 -I tools/replay-ui-oracle.py "$D/cap" 1 3 "$D/s3.txt"); i5=$(python3 -I tools/replay-ui-oracle.py "$D/cap" 1 5 "$D/s5.txt")
[ "$i3" != "$i5" ] || fail "oracle cannot tell frame 3 from 5"
echo "ok   app strip == original capture records $i3 and $i5 (frames 3 and 5 of request 1)"
for p in "3 $i3" "5 $i5"; do set -- $p
  .work/nx_rui sound "{\"cap\":\"$D/cap\",\"out\":\"$D/orig$1.wav\",\"start\":$2,\"tokens\":1}" >/dev/null
  cmp -s "$D/replay/t.frame$1.wav" "$D/orig$1.wav" || fail "frame $1 WAV differs from the C3 sound of original record $2"
done
cmp -s "$D/replay/t.frame3.wav" "$D/replay/t.frame5.wav" && fail "frames 3 and 5 sound the same"
echo "ok   frame WAV == C3 sound of the original record (bytes), frames differ"
[ "$(stat -c %s .work/shots/replay.png 2>/dev/null || echo 0)" -gt 20000 ] || fail "shot missing"
ocr=$(img-read .work/shots/replay.png 2>&1); echo "$ocr" | grep -qi "live scan" || fail "OCR of the scan view lacks 'live scan'"
echo "PASS replay-ui: scrubbed frames 3 and 5 equal original records $i3 and $i5 (strip and WAV), shot looked at by OCR"
