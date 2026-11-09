# Prereg A5: MoE router capture (written 2026-11-09 before any run; the engine code exists, the gate has not run)

Engine: baro neuron2b commit "wip(engine): A5 routing weights ..." on top of the existing `BARO_EXPERTS` trace (ids per layer per position) adds the top-k routing weights.
Gate (not yet run, blocked): RegesCore-1.0-35B (qwen35moe) engine with `BARO_EXPERTS=FILE BARO_PREFILL=0 BARO_SPEC=0`, a 6-token prompt, every one of the 40 layers, positions 0 to 5:
1. the set of top-8 expert ids per (position, layer) equals llama.cpp's `ffn_moe_topk` for the same tokens on CPU (oracle: a small C++ callback over libllama that prints the i32 tensor, `tools/moe-topk-oracle.cpp`, to be written; `llama-eval-callback` elides the middle of an 8-wide row, so it cannot be the oracle);
2. the routing weights match llama.cpp's `ffn_moe_weights` within 1e-3 relative (both normalised the same way, checked first on layer 0);
3. mutant: comparing layer l's ids against llama's layer l+1 must differ in at least one position.
Allowed misses: none on ids. A tie between experts of equal logit may order differently; the comparison is on the set, and a tie is reported, not hidden.
Blocker found: no `pack.bin` for RegesCore exists on this machine (`.work/latent-spec/pack-tier` holds index and tokenizer files only; the GGUF is at `~/Models/RegesCore-1.0-35/`). Baking one is `tools/engine-pack.mojo --arch qwen35moe` on a 20 GB file, then a tier run with experts in pinned host RAM.

## Result (2026-11-09, engine neuron2b 7cfe9b5 build, pack rebuilt byte-identical, receipts `.work/a5-check/`, `.work/a5-first-trace.txt`): MISSED

6 tokens ("The capital of France is a"), 40 layers, 240 (position, layer) cells. After fixing my oracle (below), the engine's top-8 expert id set equals llama.cpp's `ffn_moe_topk` in **198 of 240 cells**; 41 cells differ by one expert and 1 cell by two (overlap histogram 8:198, 7:41, 6:1).
The swapped experts are not far-tail ties: their routing weights sit around 0.09 to 0.11 on both sides (largest 0.113), so the 8th and 9th choice are close but not equal. Routing weights of the 1877 shared experts against `ffn_moe_weights_norm`: median relative difference 1.2e-2, 90th percentile 4.1e-2, maximum 1.5e-1 (maximum absolute 0.031); bar 1e-3.
Layer-shift mutant: differs (the check can fail). Bars: ids equal everywhere MISSED, weights within 1e-3 MISSED. Reading: the engine's router is faithful to llama.cpp at the percent level, not bit-near; the Q4 pack and the engine's activation precision are the likely cause, not tested here.
Three engine variants (one prompt row per step, without checkpoints, defaults) were run in one job; their traces were deleted by a script edit before they could be re-compared with the fixed oracle (the first run's trace, default settings, is the one kept above); before the oracle fix all three gave the same wrong 203 of 240.
Oracle bug found on the way and fixed: `ggml_argsort_top_k` returns a view whose row stride is the number of experts (256), not 8, so reading `token * 8 + e` is right for token 0 only (position 0 matched, positions 1 to 5 did not). `tools/moe-topk-oracle.cpp` now reads through the tensor strides.
