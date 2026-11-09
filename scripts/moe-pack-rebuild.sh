#!/usr/bin/env bash
# A5-prep: rebuild the RegesCore qwen35moe pack from CURRENT baro sources with the pack tool directly, not through `tools/baro run` (that builds the engine
# from the sources embedded in the GGUF, 3c6c25c, which today's toolchain rejects: 'existing function with conflicting signature' on external_call["read"]).
# The pack needs no engine build: tools/engine-pack.py (gguf-py, numpy) writes pack.bin + index.txt. The old identity.json names another bake (BARO-e340ee1, not on disk), so its pack_sha256 is only
# reported; the pass check is structural: index.txt identical to the old one (same tensors, dtypes, offsets) and pack.bin the same size as the old pack_bytes.
# One CPU-room gpu-wait job (RAM heavy). Writes into the baro main checkout's .work/moe-ngram/pack: run it only when the boss says go.
set -euo pipefail
BARO=${BARO:-$HOME/Projects/mojo/mojo-baro}; SRC=${SRC:-$HOME/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO-3c6c25c.gguf}; OUT=${OUT:-$BARO/.work/moe-ngram/pack}
[ -f "$SRC" ] || { echo "FAIL setup: no GGUF at $SRC"; exit 1; }
want=$(python3 -c "import json;d=json.load(open('$OUT/identity.json'));print(d['pack_sha256'],d['pack_bytes'])" 2>/dev/null || true); [ -n "$want" ] && [ -f "$OUT/index.txt" ] || { echo "FAIL setup: no old identity.json and index.txt in $OUT (the references for the check)"; exit 1; }
J=$PWD/.work/moe-pack-rebuild; mkdir -p "$J"; cp "$OUT/identity.json" "$J/identity.old.json"; cp "$OUT/index.txt" "$J/index.old.txt"
cat > "$J/run.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
export PATH='$PATH' HOME='$HOME'
cd '$BARO'
./.venv/bin/python3 -I tools/engine-pack.py '$SRC' '$OUT' --arch qwen35moe > '$J/pack.log' 2>&1
got=\$(sha256sum '$OUT/pack.bin' | cut -d' ' -f1); size=\$(stat -c %s '$OUT/pack.bin')
echo "pack_sha256 \$got size \$size (old: $want)" >> '$J/pack.log'
if cmp -s '$J/index.old.txt' '$OUT/index.txt' && [ "\$size" = "\$(echo $want | cut -d' ' -f2)" ]; then echo "STRUCTURE MATCH" >> '$J/pack.log'; else echo "STRUCTURE DIFFERENT" >> '$J/pack.log'; fi
EOS
chmod +x "$J/run.sh"; gpu-wait run --room cpu --cores 8 --mem 40 --timeout 5400 -- "$J/run.sh" || true
tail -n 3 "$J/pack.log"; grep -q "^STRUCTURE MATCH" "$J/pack.log" && echo "PASS moe-pack-rebuild: index.txt identical to the old pack and the same size" || { echo "FAIL moe-pack-rebuild: see $J/pack.log"; exit 1; }
