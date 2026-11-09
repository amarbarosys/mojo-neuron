#!/usr/bin/env bash
# G4 chain (docs/prereg-g4-tasks.md): data, gen, cap, dev, test, verdict on the T1c workdir (needs select.json for the layer). The engine must have implants_lo (neuron2b engine-9b-g4 or newer).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python3}; W=${W:-.work/impl-m3}; export ENGINE=${ENGINE:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2b/.work/engine-9b-g4}
for st in data gen cap dev test verdict; do $PY -I tools/g4-harness.py $W $st; done
