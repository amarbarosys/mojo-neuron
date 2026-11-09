#!/usr/bin/env bash
# omp on baro-serve (only the read tool enabled) through its own isolated profile `baro-nx` (provider baro, models list from the server). Needs baro-serve.sh up.
. "$(dirname "$0")/common.sh"
P=$HOME/.omp/profiles/baro-nx/agent; mkdir -p "$P"
cat > "$P/models.yml" <<YML
providers:
  baro:
    baseUrl: $AGENT_URL/v1
    api: openai-completions
    auth: none
    models:
      - id: $MODEL
        name: $MODEL
        contextWindow: 32768
        maxTokens: 4096
YML
cd "$SMOKE"
timeout "${AGENT_TIMEOUT:-600}" omp --profile baro-nx --model "baro/$MODEL" --tools read --no-session -p "$TASK" > "$OUT" 2>&1 || true
verdict
