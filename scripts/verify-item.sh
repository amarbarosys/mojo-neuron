#!/usr/bin/env bash
# verify-item ITEM SHA GATE...: run a lane gate on the exact sha in a detached worktree, write the verdict.
# Verdict .work/verify/ITEM.md (PASS/FAIL, sha, diff stat, rules + rlog lines, receipt), log .work/verify/ITEM-SHA.log.
set -euo pipefail
ITEM=$1; SHA=$2; shift 2; GATE="$*"
MAIN=$(cd "$(dirname "$0")/.." && pwd); VW=${VERIFY_WT:-$MAIN/../mojo-neuron-lanes/verify}; OUT=$MAIN/.work/verify; mkdir -p "$OUT"
LOG=$OUT/$ITEM-${SHA:0:7}.log; MD=$OUT/$ITEM.md
git -C "$MAIN" cat-file -e "$SHA^{commit}" || { echo "FAIL sha: $SHA not in repo"; exit 1; }
[ -d "$VW" ] || git -C "$MAIN" worktree add --detach "$VW" "$SHA" >/dev/null
[ -z "$(git -C "$VW" status --porcelain --untracked-files=no)" ] || { echo "FAIL worktree dirty: $VW"; exit 1; }
git -C "$VW" checkout -q --detach "$SHA"
RC=0; (cd "$VW" && bash -c "$GATE") > "$LOG" 2>&1 || RC=$?
V=PASS; [ $RC = 0 ] || V=FAIL
{ echo "# $ITEM: $V"; echo; echo "- sha: $SHA ($(git -C "$MAIN" log -1 --format=%s "$SHA")) tree $(git -C "$MAIN" rev-parse "$SHA^{tree}" | cut -c1-12)"; echo "- gate: \`$GATE\` exit $RC, $(date -u +%FT%TZ)"
  echo "- receipt: $LOG"; echo "- gate tail:"; echo '```'; tail -5 "$LOG"; echo '```'
  echo "- diff stat:"; echo '```'; git -C "$MAIN" show --stat --format= "$SHA" | tail -15; echo '```'
  echo "- rules added:"; git -C "$MAIN" show "$SHA" -- docs/rules.md | grep '^+[^+]' | cut -c1-200 | sed 's/^/  /' || true
  echo "- rlog in diff: $(git -C "$MAIN" show "$SHA" | grep -ci 'rlog' || true) lines"; } > "$MD"
echo "$V $ITEM ${SHA:0:7} -> $MD"; [ $RC = 0 ]
