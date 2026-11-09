#!/usr/bin/env bash
# Gate (spends a few haiku tokens, run once per MCP change): a real Claude Code client connects to .work/mcp over its own stdio (a socket pair),
# calls models and lens, and must report the token the nx CLI returns. This is the way a user meets the server.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work
MOD=qwen2.5-0.5b-instruct-q4_k_m; D=$PWD/.work/neurons-mcp
[ -f "$D/$MOD/12.lens.bin" ] || tests/mcp.sh >/dev/null
printf '{"mcpServers":{"mojo-neuron":{"command":"%s/.work/mcp","args":[],"env":{"MNEURON_DIR":"%s"}}}}' "$PWD" "$D" > .work/mcp.json
want=$(MNEURON_DIR=$D .work/nx lens "{\"model\":\"$MOD\",\"layer\":12,\"neuron\":5}" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["top"][0][0])')
got=$(echo "Call the mojo-neuron models tool, then the lens tool for the first model, layer 12, neuron 5. Reply with only the first promoted token, nothing else." \
  | timeout 150 claude -p --model haiku --effort low --strict-mcp-config --mcp-config .work/mcp.json --allowedTools "mcp__mojo-neuron__models" "mcp__mojo-neuron__lens" 2>&1 | tail -1)
echo "$got" | grep -qF "$want" || { echo "FAIL mcp client: reply '$got' does not contain the CLI's first token '$want'"; exit 1; }
echo "PASS mcp client: real Claude client got '$want' through the server"
