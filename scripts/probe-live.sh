#!/usr/bin/env bash
# `nx probe` with a `text` argument: capture one statement through the engine (n=1) into CAP, every captured layer of the lie probe. usage: probe-live.sh TEXT CAP
set -euo pipefail
cd "$(dirname "$0")/.."
TEXT=${1:?usage: probe-live.sh TEXT CAP}; CAP=${2:?}
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2}; ENGINE=${ENGINE:-$BARO/.work/engine-9b}; PACK=${PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
GG=${GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf}; LAYERS=${LAYERS:-3,7,11,15,19,23,27,31}
export LD_LIBRARY_PATH=$HOME/Models/llama.cpp/build/bin
ids=$("$HOME/Models/llama.cpp/build/bin/llama-tokenize" -m "$GG" --ids --log-disable -p "$TEXT" 2>/dev/null | tail -1 | tr -d '[] ')
[ -n "$ids" ] || { echo "FAIL tokenize: no ids for the text"; exit 1; }
rm -rf "$CAP"; mkdir -p "$CAP"; echo "{\"id\":1,\"prompt\":[$ids],\"n\":1}" > "$CAP.req"
cat > "$CAP.run.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACK' BARO_SERVE=1 BARO_SPEC=0 BARO_CAPTURE='$CAP' BARO_CAPTURE_LAYERS='$LAYERS' BARO_CAPTURE_FIELDS=r BARO_CAPTURE_MODEL=qwythos-9b
exec '$ENGINE' < '$CAP.req' > '$CAP.out' 2> '$CAP.err'
EOS
chmod +x "$CAP.run.sh"
gpu-wait run --label probe-live --priority 50 --timeout 300 -- "$CAP.run.sh" || true
[ -s "$CAP/req.i32" ] && echo "ok   $CAP" || echo "FAIL run: no capture: $(grep -iE 'error|abort' "$CAP.err" | tail -1 | cut -c1-200)"
