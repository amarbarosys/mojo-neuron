# Parity checklist

Each row: feature of gne (G) or neuron-explorer (N), owner lane, gate that ticks it. `[ ]` open, `[x]` ticked with gate name.
Matrix of model x feature is added at the end of the lane (PASS / UNSUPPORTED(reason)).

| # | feature | src | lane | gate |
|---|---|---|---|---|
| 1 | logit lens, every neuron of a layer, CPU | G N | app | [x] `scripts/check.sh` lens (step 1) |
| 2 | lens for every tier A+B arch | G | app | [x] `scripts/lens-matrix.sh`, docs/matrix.md |
| 3 | R lens / J++ lens (own builder) | N | app | [~] R-style ridge lens PASS (`tests/rlens.sh`: numpy oracle, held-out beats identity, golden); LRP-Jacobian R and J++ UNSUPPORTED(needs backward pass, calibration tasks); parity vs downloaded Qwen3.5-4B maps UNSUPPORTED(no 4B weights, no qwen35 CPU forward) |
| 4 | rank: word-cluster mass vs random controls | G N | app | [x] `tests/rank.sh` golden + 2 mutants |
| 5 | steer lens (next-token dlogprob at +-dose) | G | engine | [ ] |
| 6 | reverse lookup: persona contrast -> cosine rank all layers | G | engine | [ ] |
| 7 | sweep neurons x doses x prompts, resident | G | engine | [ ] |
| 8 | BARO_STEER add vector per layer on GPU | G | engine | [ ] |
| 9 | cvec export `.cvec.gguf`, llama.cpp loads it | G | engine | [ ] |
| 10 | ablation (neuron zero, directional) | G | engine | [ ] |
| 11 | activations: `fires` over user text, filter neurons | N G | app | [x] `tests/calls.sh` fires + `fires_check` vs llama.cpp |
| 12 | connections up/downstream (weight cosine) | N G | app | [x] `conn_check` + `tests/calls.sh` |
| 13 | dossier per neuron | G | app | [x] `tests/calls.sh` dossier |
| 14 | recipes `L:N:dose,...` parse, compose, export | G | app | [x] `tests/calls.sh` recipe |
| 15 | 3-column steered chat (-, 0, +) | G N | app (engine sweep) | [x] `tests/live.sh` (call = engine sweep run directly), `tests/app.sh` (base column from the file rows, generate button runs the sweep); chat tokens are not streamed: one batch per click |
| 16 | map view (layers x neurons) | G N | app | [x] `tests/app.sh` map view |
| 17 | list view (rank, lens search by token) | N | app | [x] `tests/app.sh` list view, token search |
| 18 | neuron view | N | app | [x] `tests/app.sh` neuron view |
| 19 | MCP server over the same calls | G N | app | [x] `tests/mcp.sh` pipe, `tests/mcp_client.sh` real Claude client |
| 20 | brain-net export: explored neuron -> note; `brain-ask` finds it | new | app | [x] `tests/calls.sh` brain-ask |
| 21 | gain steering `xN`, LoRA export | G | out of scope v1 | UNSUPPORTED(not planned) |
| 22 | heads, MoE experts, audio, FineFineWeb contexts, UMAP | G N | out of scope v1 | UNSUPPORTED(not planned) |
| 23 | OpenAI-compatible endpoint with `when` | G | out of scope v1 | UNSUPPORTED(not planned) |
