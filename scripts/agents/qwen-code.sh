#!/usr/bin/env bash
# qwen-code on baro-serve: a project settings file in the throwaway dir names the model, endpoint and explicit limits (its default max_tokens plus the
# prompt overran the server's 32k window: HTTP 400 on the first request), one headless task in plan mode (analysis only: it can read, not edit or run commands).
. "$(dirname "$0")/common.sh"
mkdir -p "$SMOKE/.qwen"
cat > "$SMOKE/.qwen/settings.json" <<JSON
{"security": {"auth": {"selectedType": "openai"}},
 "model": {"name": "$MODEL"},
 "modelProviders": {"openai": [{"id": "$MODEL", "name": "baro $MODEL", "envKey": "OPENAI_API_KEY", "baseUrl": "$AGENT_URL/v1",
   "generationConfig": {"contextWindowSize": 30000, "timeout": 600000, "maxRetries": 0, "samplingParams": {"temperature": 0, "max_tokens": 2048}}}]}}
JSON
cd "$SMOKE"
OPENAI_API_KEY=baro timeout "${AGENT_TIMEOUT:-600}" qwen --auth-type openai -m "$MODEL" --approval-mode plan --chat-recording=false -p "$TASK" > "$OUT" 2>&1 || true
verdict
