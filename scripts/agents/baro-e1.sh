#!/usr/bin/env bash
# baro-e1.sh [SHA]: build engine-9b and baro-serve from a detached baro worktree at SHA (default 7c37795, E1 steer + capture passthrough), never from an
# older lane binary. Receipt .work/baro-e1/SHA names the commit the two binaries were built from; baro-serve.sh refuses to start without it.
set -euo pipefail
cd "$(dirname "$0")/../.."; mkdir -p .work/baro-e1
SHA=${1:-${BARO_SHA:-7c37795}}; BARO=${BARO:-$HOME/Projects/mojo/mojo-baro}; WT=${BARO_WT:-$HOME/Projects/mojo/mojo-baro-lanes/app-e1}; O=$PWD/.work/baro-e1
fail() { echo "FAIL baro-e1: $1"; exit 1; }
git -C "$BARO" cat-file -e "$SHA^{commit}" || fail "$SHA is not in $BARO"
[ -d "$WT" ] || git -C "$BARO" worktree add --detach "$WT" "$SHA" >/dev/null
[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || fail "worktree dirty: $WT"
git -C "$WT" checkout -q --detach "$SHA"; HEADSHA=$(git -C "$WT" rev-parse HEAD)
[ -e "$WT/.venv" ] || ln -s "$BARO/.venv" "$WT/.venv"
rm -f "$O/SHA" "$O/engine-9b" "$O/baro-serve"
(cd "$WT" && ./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o "$O/engine-9b") > "$O/build-engine.log" 2>&1 || { grep -m3 error "$O/build-engine.log"; fail "engine build"; }
cargo build --release --manifest-path "$WT/serve/Cargo.toml" > "$O/cargo.log" 2>&1 || fail "cargo build, see $O/cargo.log"
cp "$WT/serve/target/release/baro-serve" "$O/baro-serve"
[ -x "$O/engine-9b" ] && [ -x "$O/baro-serve" ] || fail "binaries missing"
echo "$HEADSHA" > "$O/SHA"; echo "built engine-9b and baro-serve from $HEADSHA -> $O"
