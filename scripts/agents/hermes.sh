#!/usr/bin/env bash
# hermes on baro-serve: custom provider flags, file toolset only (no terminal), no --yolo, 6 turns, in the throwaway smoke dir. Needs baro-serve.sh up.
. "$(dirname "$0")/common.sh"
cd "$SMOKE"
timeout "${AGENT_TIMEOUT:-600}" hermes chat --provider custom --base-url "$AGENT_URL/v1" --api-key baro -m "$MODEL" --in "$SMOKE" -t file --max-turns 6 --ignore-user-config -Q -q "$TASK" > "$OUT" 2>&1 || true
verdict
