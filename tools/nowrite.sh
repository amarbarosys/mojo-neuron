#!/usr/bin/env bash
# nowrite.sh ALLOWED_DIR ROOT... -- CMD...: run CMD, then fail if any file under a ROOT is new or modified since the start and
# not under ALLOWED_DIR. Used by the F29 gate (docs/prereg-f29.md bar 4).
set -euo pipefail
allowed=$(realpath -m "$1"); shift; roots=(); while [ "$1" != "--" ]; do roots+=("$(realpath -m "$1")"); shift; done; shift
stamp=$(mktemp); trap 'rm -f "$stamp"' EXIT; sleep 1.1   # mtime granularity
"$@"; rc=$?
bad=$(find "${roots[@]}" -type f -newer "$stamp" 2>/dev/null | grep -v -F -e "$allowed/" -e "$stamp" || true)
[ -z "$bad" ] || { echo "FAIL nowrite: written outside $allowed:"; echo "$bad" | head -5; exit 1; }
exit "$rc"
