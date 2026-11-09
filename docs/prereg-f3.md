# Prereg F3: concept arithmetic (written before the result runs, 2026-11-09, lane probe)

`nx concept "A - B + C"`: each term is a unit direction (decoder column `L:N`, unembedding row `tok:TEXT@L`, raw vector `vec:FILE@L`, or
captured mean `mean(CAP,L,a-b)`), weighted, summed per layer, times `scale`. Lens of the result = W_U (v * output_norm), top and bottom k.

## CPU gate (tests/concept.sh, qwen2.5-0.5b). Written and green BEFORE this document: a record of the bars, not a prereg
1. `tok:_Paris@12 - tok:_Berlin@12`: " Paris" in the top 3 of the lens and " Berlin" in the bottom 3.
2. The swapped expression puts " Berlin" in its top 3 (the check can fail).
3. `A - A` is the exact zero vector (byte equal to zeros), reports no lens layer.
4. `A - B + B` has the lens of `A` (same top 3).
5. Bad expressions (dangling operator, missing operator, unknown token, layer out of range) fail with a `concept` message.

## Engine gate (Qwythos-9B, scripts/concept-engine-check.sh)
6. Zero concept (`A - A`) steers like no steer: the top-20 logits of a greedy first step, same process (`BARO_STEER_SERVE=1`), equal to the
   unsteered request within one bf16 step (relative 2^-8 of the larger value) and the same token order. 3 prompts.
7. A real concept (`mean` FR - EN at a captured layer, scale 8) changes the top-20 of the same prompts (at least one token id or order differs
   on each prompt), so check 6 can fail.

## Known pair: language swap (scripts/concept-lang-check.sh)
20 French sentences vs their English translations (`tests/data/en-fr.tsv`), captured at layers 15, 23, 27 (last-token residual).
`mean(FR) - mean(EN)` lens, k = 50, at layer 23 or 27 (either passes): top 50 holds >= 4 French function-word tokens and <= 1 English;
bottom 50 holds >= 4 English function-word tokens and <= 1 French. The swapped expression (`EN - FR`) must flip both counts.
French markers: ` le la les de des du et est un une que pour dans qui pas sur avec au aux je il nous vous`.
English markers: ` the of and is to that with for in are was on at by`.
