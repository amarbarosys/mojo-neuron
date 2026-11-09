#!/usr/bin/env bash
# Gate C3: sonification of a fixed capture is deterministic (golden sha256), pitch follows the layer (FFT against 110*2^(3L/24) harmonics),
# the timbre follows the data (other data, other bytes), and the WAV is a valid 24 kHz mono 16-bit file of 0.12 s per token. CPU only.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/snd
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
PY=${PY:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python}; D=$PWD/.work/snd; fail() { echo "FAIL sound: $1"; exit 1; }
rm -f .work/nx_sound; timeout 400 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_sound 2>&1 | grep -E "error" -A4 && fail "build nx" || true
[ -x .work/nx_sound ] || fail "no .work/nx_sound"
export NX_RLOG=$D/rlog; rm -rf "$NX_RLOG" "$D"/c* ; snd() { .work/nx_sound sound "{\"cap\":\"$1\",\"out\":\"$2\"${3:+,$3}}" >/dev/null; }
$PY tools/mkcap.py $D/cg 4,12 64 30 7; snd $D/cg $D/g1.wav; snd $D/cg $D/g2.wav
h1=$(sha256sum < $D/g1.wav | cut -c1-64); [ "$h1" = "$(sha256sum < $D/g2.wav | cut -c1-64)" ] || fail "two renders of one capture differ"
[ "$h1" = "$(cut -c1-64 tests/sound_golden.sha256)" ] || fail "golden mismatch: $h1 (tests/sound_golden.sha256)"
echo "ok   golden: $h1 (twice)"
# the data decides: another seed and a shifted window change the bytes
$PY tools/mkcap.py $D/cs 4,12 64 30 8; snd $D/cs $D/s.wav; snd $D/cg $D/w.wav '"start":1'
[ "$(sha256sum < $D/s.wav | cut -c1-64)" != "$h1" ] && [ "$(sha256sum < $D/w.wav | cut -c1-64)" != "$h1" ] || fail "mutant (other data or window) gives the golden bytes"
$PY - $D <<'PYEOF' || exit 1
import sys, wave, subprocess, numpy as np
D = sys.argv[1] + "/"
def tool(layer):
    subprocess.run([sys.executable, "tools/mkcap.py", D + "c%d" % layer, str(layer), "64", "30", "7"], check=True)
    subprocess.run([".work/nx_sound", "sound", '{"cap":"%sc%d","out":"%sp%d.wav"}' % (D, layer, D, layer)], check=True, capture_output=True)
    w = wave.open(D + "p%d.wav" % layer)
    assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (24000, 1, 2, 30 * 2880), "wav header"
    return np.frombuffer(w.readframes(30 * 2880), dtype=np.int16).astype(float)
def peaks(x):  # the five strongest partials over the first 10 notes
    f = sum(np.abs(np.fft.rfft(x[i * 2880:(i + 1) * 2880] * np.hanning(2880))) for i in range(10)); fr = np.fft.rfftfreq(2880, 1 / 24000)
    return fr[np.argsort(-f)[:5]]
def harmonic_err(p, f0): return max(abs(q / f0 - round(q / f0)) for q in p)
res = {}
for L in (4, 8):
    x = tool(L); res[L] = peaks(x)
f4, f8 = 110 * 2 ** (3 * 4 / 24), 110 * 2 ** (3 * 8 / 24)
print("L4 peaks %s err(f0 L4) %.3f err(f0 L8) %.3f | L8 peaks %s err(f0 L8) %.3f err(f0 L4) %.3f" % (res[4].round(), harmonic_err(res[4], f4), harmonic_err(res[4], f8), res[8].round(), harmonic_err(res[8], f8), harmonic_err(res[8], f4)))
ok = harmonic_err(res[4], f4) < 0.05 and harmonic_err(res[8], f8) < 0.05 and harmonic_err(res[8], f4) > 0.1  # the mutant (L8 voiced as L4) is rejected
sys.exit(0 if ok else print("FAIL sound: pitch does not follow the layer") or 1)
PYEOF
grep -q '"tool":"sound"' "$NX_RLOG"/2*.jsonl || fail "no rlog line"
echo "PASS sound: golden $h1, pitch per layer, data-dependent, rlog"
