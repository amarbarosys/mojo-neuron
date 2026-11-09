# Prereg A3: neuron surgery (written before the result run, 2026-11-09)

Source GGUF = the one the engine pack was baked from (`identity.json` of e11b-qwythos-pack): Qwythos-9B Q4_0-pure. `surgery ... op=zero` (default via=up)
zeroes the ffn_up row of neuron N of layer 16 (exact: all-zero blocks dequantize to 0, nothing else changes). Pack baked from the surgery GGUF
(`tools/engine-pack.py --q4`), then the same engine, same 13-token prompt, raw next-token logits (BARO_DUMP_LOGITS_DIR):
Bars (any miss = FAIL):
1. engine `gain 16:N:0` on the original pack == identity-gain run on the surgery pack: max |logit diff| <= 1e-3 (expected 0).
2. Mutant: the same comparison against the unablated run (identity gain on the original pack) must show max |logit diff| > 10x the bar-1 value
   and > 1e-3 (the neuron has an effect, the check can fail).
3. llama.cpp (CPU) loads the surgery GGUF and generates 4 tokens, exit 0.
4. Only bytes of the one ffn_up row differ between the two GGUFs, all written as 0 (tests/causal.sh checks the same on 0.5B).
Not claimed: down-column surgery on K quants is exact (it is not; `new_max` in the tool output shows the residual), or that surgery GGUFs run
on the engine without a re-bake (the engine reads a pack, so a surgery file needs `engine-pack.py` first).
