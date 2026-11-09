#!/usr/bin/env bash
# Gate A5 (docs/prereg-a5.md): RegesCore qwen35moe engine expert trace (ids + routing weights) vs llama.cpp's ffn_moe_topk / ffn_moe_weights on 6 tokens, 40 layers.
set -euo pipefail
cd "$(dirname "$0")/.."
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2b}; ENGINE=${ENGINE:-$BARO/.work/engine-moe}; PACKD=${PACKD:-$HOME/Projects/mojo/mojo-baro/.work/moe-ngram/pack}
GG=${GGUF:-$HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-3c6c25c.gguf}; TOKJ=${TOKJ:-$HOME/Projects/mojo/mojo-baro/.work/latent-spec/pack-tier/tokenizer.json}
PYB=${PYB:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python3}; ORACLE=$PWD/.work/a5/moe-topk-oracle; NT=6
O=$PWD/.work/a5-check; mkdir -p "$O"; mkdir -p "$O/old"; mv -f "$O"/trace*.txt "$O"/engine*.log "$O/old/" 2>/dev/null || true
[ -x "$ENGINE" ] && [ -f "$PACKD/pack.bin" ] && [ -x "$ORACLE" ] || { echo "FAIL setup: engine $ENGINE, pack $PACKD/pack.bin or oracle $ORACLE missing"; exit 1; }
"$PYB" -I -c "
from tokenizers import Tokenizer
ids = Tokenizer.from_file('$TOKJ').encode('The capital of France is a very old city', add_special_tokens=False).ids[:$NT]
assert len(ids) == $NT, ids
open('$O/ids.txt', 'w').write(' '.join(map(str, ids)) + '\n'); print('ids', ids)"
cat > "$O/oracle.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME'
'$ORACLE' '$GG' '$O/ids.txt' > '$O/oracle.txt' 2> '$O/oracle.err'
EOS
cat > "$O/engine.sh" <<EOS
#!/usr/bin/env bash
export PATH='$PATH' HOME='$HOME' BARO_PACK='$PACKD' BARO_PROMPT='$O/ids.txt' BARO_EXPERTS='\$TRACE' BARO_MEGA=0 BARO_SPEC=0 BARO_TMAX=256 MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2
# variants in one job (one queue wait): v1 one prompt row per step, v2 the same without prefix checkpoints, v3 the engine's defaults
for v in 1 2 3; do
  case \$v in 1) EX="BARO_REPLAY_M=1";; 2) EX="BARO_REPLAY_M=1 BARO_CKPT=0";; 3) EX="";; esac
  rm -f '$O/trace-v'\$v'.txt'; env \$EX BARO_EXPERTS='$O/trace-v'\$v'.txt' '$ENGINE' > '$O/engine-v'\$v'.log' 2>&1
done
EOS
chmod +x "$O/oracle.sh" "$O/engine.sh"
[ -s "$O/oracle.txt" ] || gpu-wait run --room cpu --cores 8 --mem 32 --timeout 1500 -- "$O/oracle.sh" || true  # llama.cpp output of the first run is reused
[ -s "$O/oracle.txt" ] || { echo "FAIL oracle: $(tail -n 2 "$O/oracle.err" | cut -c1-200)"; exit 1; }
gpu-wait run --vram 23 --mem 12 --timeout 900 -- "$O/engine.sh" || true
[ -s "$O/trace-v1.txt" ] || { echo "FAIL engine: $(grep -iE 'error|abort|out of memory' "$O/engine-v1.log" | tail -n 1 | cut -c1-200)"; exit 1; }
for v in 1 2 3; do echo "== variant $v"; [ -s "$O/trace-v$v.txt" ] && "$PYB" -I tools/moe-router-compare.py "$O/trace-v$v.txt" "$O/oracle.txt" $NT || true; done
"$PYB" -I tools/moe-router-compare.py "$O/trace-v1.txt" "$O/oracle.txt" $NT
echo "PASS moe-router-check: expert ids equal llama.cpp ffn_moe_topk on $NT tokens x 40 layers, routing weights within 1e-3"
