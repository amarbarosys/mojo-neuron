# Provenance

Ideas only, no code copied. Both projects are MIT; their licence texts sit in their repos.

| source | what we took (idea) | where we differ |
|---|---|---|
| gguf-neuron-explorer (`gne`), `~/Projects/imports/gguf-neuron-explorer-main` | sweep+rank against random controls, logit/steer lens, reverse lookup, dossier, recipes `L:N:dose`, 3-column chat, cvec export, MCP | Mojo, resident process, CPU lens in seconds |
| neuron-explorer, `~/Projects/imports/neuron-explorer-main` | map/list/neuron views, `fires` over user text, connections up/down, R and J++ lens, name per neuron | mojo-imkit native app, any GGUF arch |
| bubskee/qwen3.5-4b-r-jpp-lenses (Apache-2.0) | downloaded maps used only as oracle for our own R/J++ builder on Qwen3.5-4B | our builder runs for every model |

Own code reused: `src/mneuron/pngw.mojo` is `stock-review/src/stock_review/pngw.mojo` (same owner). App pattern (script events, shots) follows stock-review.
