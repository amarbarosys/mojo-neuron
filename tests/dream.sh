#!/usr/bin/env bash
# Gate F26 (CPU, docs/prereg-f26.md bars 1-5 on a stand-in engine): protocol over the FIFOs, 128 tokens per dream, scorer passes a
# planted engine (steered dreams emit the unit's lens tokens), fails a plain engine, and the DREAM_MUT=1 scorer fails the planted one.
set -euo pipefail
cd "$(dirname "$0")/.."; scripts/build-agent.sh
MODEL=${MODEL:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf}
STEM=$(basename "$MODEL" .gguf | tr 'A-Z' 'a-z'); export MNEURON_DIR=${MNEURON_DIR:-$PWD/.work/neurons}
LENS=$MNEURON_DIR/$STEM/16.lens.bin; [ -f "$LENS" ] || { echo "FAIL setup: no $LENS (lens MODEL 16 32 --out)"; exit 1; }
export NX_RLOG=$PWD/.work/dream-rlog-test; rm -rf "$NX_RLOG"
one() { local o=.work/dream-test-$1; rm -rf "$o"; mkdir -p "$o"; mkfifo "$o/req" "$o/resp"; local mk=""; [ "$1" = sweep ] && mk="mults=1,2,3,4,6"
  python3 -I tools/fake-engine.py "$o" "$LENS" "$1" > "$o/resp" < "$o/req" 2> "$o/engine.err" &
  .work/dream run "$MODEL" "$o" rho=100 lens="$STEM" nfeat=4 nctrl=2 $mk < "$o/resp" > "$o/req" || { echo "FAIL dream run ($1): $(tail -1 "$o/engine.err")"; exit 1; }
  wait; echo "$o"; }
d=.work/dream
sc() { "$1" score "$2" lens="$STEM" 2>&1 | tail -1; }
p=$(one planted); n=$(one plain)
[ "$(wc -l < "$p/dream.jsonl")" = 13 ] || { echo "FAIL dream: $(wc -l < "$p/dream.jsonl") dreams, want 1 base + 4 feat + 8 ctrl"; exit 1; }
[ "$(sc .work/dream "$p")" = "PASS dream" ] || { echo "FAIL dream: planted engine not passed"; exit 1; }
[ "$(sc .work/dream "$n")" != "PASS dream" ] || { echo "FAIL dream: plain engine passed (the bars cannot fail)"; exit 1; }
[ "$(sc .work/dream_m "$p")" != "PASS dream" ] || { echo "FAIL dream: shifted-lens mutant passed"; exit 1; }
[ "$("$d" score "$p" lens="$STEM" distinct=40 2>&1 | tail -1)" != "PASS dream" ] || { echo "FAIL dream: collapsed planted dreams passed bar 7"; exit 1; }
pl=$("$d" score "$n" lens="$STEM" distinct=40 2>&1 || true); case "$pl" in *"bar7"*"ok"*) ;; *) echo "FAIL dream: plain dreams missed bar 7"; exit 1;; esac
grep -q '"kind":"anomaly"' "$NX_RLOG/anomalies.jsonl" || { echo "FAIL dream: missed bar wrote no anomaly"; exit 1; }
s=$(one sweep); [ "$(wc -l < "$s/dream.jsonl")" = 65 ] || { echo "FAIL sweep: $(wc -l < "$s/dream.jsonl") dreams, want 1 base + 5x(4 feat + 8 ctrl) + 4 prompt-only = 65"; exit 1; }
[ "$(.work/dream sweepscore "$s" lens="$STEM" 2>&1 | tail -1)" = "PASS sweep" ] || { echo "FAIL sweep: scale-dependent engine not passed"; .work/dream sweepscore "$s" lens="$STEM" | tail -8; exit 1; }
[ "$(.work/dream_m sweepscore "$s" lens="$STEM" 2>&1 | tail -1)" != "PASS sweep" ] || { echo "FAIL sweep: shifted-lens mutant passed"; exit 1; }
[ "$(.work/dream sweepscore "$p" lens="$STEM" 2>&1 | tail -1)" != "PASS sweep" ] || { echo "FAIL sweep: single-run dreams passed the sweep scorer"; exit 1; }
echo "PASS dream: protocol, 128 tokens per dream, planted passes, plain fails, shifted-lens mutant fails, anomaly logged"
