#!/usr/bin/env bash
# Engine runs for docs/prereg-f-groups.md (F2b, F27b, F29c): capture + prerank + group measurements on Qwythos-9B, layer L.
# usage: scripts/fgroups.sh capture FACTS OUTDIR        A1 capture of field x at layer L (n = 1 per context)
#        scripts/fgroups.sh prerank CAPDIR FACTS OUT    direct-effect ranking (OUT.rank.tsv)
#        scripts/fgroups.sh measure FACTS GROUPS OUTDIR group ablation/gain, one resident engine (OUTDIR/L<L>.group.tsv)
#        scripts/fgroups.sh summary OUTDIR              mean dld and mean |dld| per group
# Env: L (default 28), ENGINE_DIR (mojo-baro lane with an engine that prints dld), GROUP_BIN, PRERANK_BIN. GPU steps go through gpu-wait.
set -euo pipefail
cd "$(dirname "$0")/.."
L=${L:-28}
GG=${QWYTHOS_GGUF:-$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-q4_k_m/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf}
ED=${ENGINE_DIR:-$HOME/Projects/mojo/mojo-baro-lanes/neuron2b}
PK=${BARO_PACK:-$HOME/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack}
GB=${GROUP_BIN:-$HOME/Projects/mojo/mojo-neuron-lanes/causal/.work/group-check/group}
PB=${PRERANK_BIN:-$HOME/Projects/mojo/mojo-neuron-lanes/causal/.work/group-check/prerank}
eng="baro=$ED spark=$ED/.work/engine-9b-g pack=$PK runner=none"
gpu() { gpu-wait run --shared --vram 10 --timeout 900 -- env PATH="$PATH" HOME="$HOME" bash -c "cd '$PWD' && $*" || { echo "FAIL gpu: $*"; exit 1; }; }
case "${1:-}" in
  capture) o=$3; rm -rf "$o/cap"; mkdir -p "$o/cap"; gpu "'$GB' '$GG' '$o' layer=$L facts='$2' capture='$o/cap' $eng" ;;
  prerank) timeout 600 "$PB" "$GG" "$2" "$L" "$3" "$4" || { echo "FAIL prerank"; exit 1; } ;;
  measure) mkdir -p "$4"; gpu "'$GB' '$GG' '$4' layer=$L facts='$2' groups='$3' $eng" ;;
  summary) awk -F'\t' 'NR > 1 { if (!($1 in n)) o[++k] = $1; n[$1]++; s[$1] += $5; a[$1] += ($5 < 0 ? -$5 : $5) }
             END { for (i = 1; i <= k; i++) { g = o[i]; printf "%s\tn=%d\tmean_dld=%.4f\tmean_abs=%.4f\n", g, n[g], s[g] / n[g], a[g] / n[g] } }' "$2/L$L.group.tsv" ;;
  *) echo "FAIL usage: scripts/fgroups.sh capture|prerank|measure|summary ..."; exit 1 ;;
esac
