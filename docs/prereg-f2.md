# Prereg F2: circuit unit tests (written 2026-11-09 before any code; needs A4, lane probe)

Pin: `tests/circuits/<name>.json` = `{"name", "model": GGUF path, "prompts": [{"text", "expect": token id}], "units": [{"layer", "neuron"}...] (the top
edges' source units of the A4 graph, at most 8), "effect": E0, "fraction": 0.5, "engine": {...}}`. E0 = mean over prompts of
(logprob of `expect` unsteered) - (logprob of `expect` with every unit ablated by engine `gain L:N:0`), measured once when pinned.
`nx circuit-test NAME [--gguf FILE]` re-measures E on the given file (default the pinned model) and fails (exit 1) when E < fraction * E0.

## Bars
1. Unchanged model: E within 5% of E0 (the engine is deterministic; a larger gap is an engine fault) and the test passes.
2. Broken model: surgery (A3) that zeroes the down-projection column of the pinned top unit (one edge) gives E < fraction * E0 and the test FAILS.
3. A pin with fewer than 1 unit, an `expect` outside the vocabulary, or E0 <= 0 is refused at pin time with a message.
Order of work: the pin format and the verdict arithmetic (pure code, CPU test with a fake measure) now; the engine path when A4 and A3 land.
