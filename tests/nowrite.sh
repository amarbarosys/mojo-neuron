#!/usr/bin/env bash
# Gate for tools/nowrite.sh: a command writing only into the allowed dir passes, one writing beside it (and a .gguf) fails.
set -euo pipefail
cd "$(dirname "$0")/.."; d=.work/nowrite-test; rm -rf "$d"; mkdir -p "$d/ok" "$d/other"
tools/nowrite.sh "$d/ok" "$d" -- bash -c "echo a > $d/ok/p.md" || { echo "FAIL nowrite test: allowed write rejected"; exit 1; }
tools/nowrite.sh "$d/ok" "$d" -- bash -c "echo a > $d/ok/p.md; echo b > $d/other/x.gguf" >/dev/null && { echo "FAIL nowrite test: outside write accepted"; exit 1; }
tools/nowrite.sh "$d/ok" "$d" -- true || { echo "FAIL nowrite test: no-op rejected"; exit 1; }
echo "PASS nowrite: allowed write ok, outside write caught, no-op ok"
