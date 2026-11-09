#!/usr/bin/env bash
# G3 chain for T1 (docs/prereg-implants.md): mine, capture, select, eval-real, eval-ctl, verdict; each stage skips when its output exists. Facts chain: scripts/implants-run.sh facts.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python3}; W=${W:-.work/impl-run}; H="$PY -I tools/implants-harness.py $W"
if [ "${1:-t1}" = facts ]; then
  $H facts-base; $H facts-cap; ARM=real $H facts-eval; ARM=ctl $H facts-eval; $H facts-verdict
else
  $H data; $H mine; $H capture; $H select; $H selectk; $H eval-real; $H eval-ctl; $H verdict
fi
