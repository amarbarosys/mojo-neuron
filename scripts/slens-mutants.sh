#!/usr/bin/env bash
# Mutants of a good steer.bin that slens-verify MUST reject (run after slens-check): zeroed down records, up/down swapped.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:-q05}; LAYER=${LAYER:-12}; O=$PWD/.work/slens-check-$TAG
S=$PWD/.work/steer/$TAG; PY=${PY:-.work/venv-gne/bin/python}
GG=${GGUF:-$HOME/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf}
[ -f "$O/$LAYER.steer.bin" ] || { echo "FAIL setup: run slens-check.sh first"; exit 1; }
$PY - "$O/$LAYER.steer.bin" "$O" <<'PYEOF'
import sys, struct, numpy as np
src, out = sys.argv[1:3]
b = bytearray(open(src, "rb").read())
magic, ver, layer, ffn, k, dose = struct.unpack("<4sIIIIf", b[:24])
a = np.frombuffer(b, dtype=np.uint8, offset=24).reshape(ffn, 16 * k).copy()
z = a.copy(); z[:, 8 * k:] = 0; z[:, 8 * k:12 * k] = 0xFF   # down ids -1, values 0
sw = a.copy(); sw[:, :8 * k], sw[:, 8 * k:] = a[:, 8 * k:], a[:, :8 * k]   # up and down halves swapped
for name, m in (("zerodown", z), ("swapped", sw)):
    open(f"{out}/{layer}.mut-{name}.steer.bin", "wb").write(bytes(b[:24]) + m.tobytes())
PYEOF
rc=0
for m in zerodown swapped; do
  cp "$O/$LAYER.steer.kl.f32" "$O/$LAYER.mut-$m.steer.kl.f32"
  if $PY tools/slens-verify.py "$O/$LAYER.mut-$m.steer.bin" "$GG" "$S/rho.f32" "$PWD/.work/cvec" "$O/p4.txt" 5 77 >/dev/null 2>&1; then
    echo "FAIL mutant $m: slens-verify accepted it"; rc=1
  else echo "ok   mutant $m rejected"; fi
done
[ $rc = 0 ] && echo "PASS slens-mutants"; exit $rc
