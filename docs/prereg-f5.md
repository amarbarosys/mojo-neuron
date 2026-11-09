# Prereg F5: neuron telepathy (written before any code or result, 2026-11-09)

Needs B3 (cross-model ridge maps). Model A = Qwythos-9B or qwen2.5-0.5B (whichever B3 maps to the other), model B the other.
Per prompt P (10 fixed prompts): A reads P, its residual at layer La (mean over the last 4 prompt tokens) goes through the B3
map to B's layer Lb and steers B on a neutral prompt `The` (scale 0.8 rho_B, all 24 generated tokens), B continues; A's own
unsteered continuation of P (24 tokens) is the target.
Score = Jaccard of the lowercase alnum word sets of B's continuation and A's continuation, mean over the 10 prompts.

Conditions (same B prompt, same scale): `ridge` (B3 map), `shuffled` (the map's rows permuted, fixed seed), `random` (random
direction, same norm), `zero` (no steer), `identity` (A = B: A's own residual, no map).

Bars (all must hold, any miss = FAIL):
1. `identity` has the highest mean score of the five conditions.
2. `shuffled` is near chance: |shuffled - random| <= 0.02 and shuffled <= zero + 0.02.
3. `ridge` beats `shuffled` on at least 7 of 10 prompts and in the mean (reported, not a claim of telepathy if it only ties).
4. Mutant: scoring B's continuation against another prompt's A continuation drops `identity` below `ridge`'s ordering rank 1 (the check can fail).
5. Rlog: one normal line per condition, anomaly on a missed bar.
