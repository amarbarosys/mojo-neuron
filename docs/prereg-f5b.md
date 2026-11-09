# Prereg F5b: neuron telepathy scored in effect space (written before any code or result, 2026-11-09)

Source: papers synthesis row "cross-model maps: compare effects, not token overlap" (steer-effect correlation, stitching NLL; neuron-level maps
under-report). `docs/prereg-f5.md` (word-overlap score, committed, not re-cut) stays; F5b is the effect-space companion and the one to quote when they disagree.

Pair = B3's bar-1 pair (Qwen2.5-0.5B layer 12 -> VibeThinker-1.5B layer 14, one tokenizer), so effects are comparable token by token. Per prompt P
(the same 10): `e_A` = change in A's next-token log-probs (full vocabulary, last prompt position) when A's residual at La is shifted by the concept vector
v (v = the mean residual difference of P-contexts vs the neutral `The`, scale 0.8 rho_A); `e_B` = the same measure on B with the B3-mapped vector
`M v` at Lb (scale 0.8 rho_B), neutral prompt. Agreement = Pearson correlation of `e_A` and `e_B` over the shared vocabulary ids, mean over the 10 prompts.
Conditions: `ridge` (B3 map), `shuffled` (rows permuted, seed), `random` (random direction, same norm), `identity` (A = B, no map, ceiling).
Also reported: B's NLL change on the 10 prompts' own continuations under the mapped steer (stitching-style side effect).

Bars (all must hold; a miss = FAIL):
1. `identity` agreement >= 0.95 (the same model through no map must reproduce its own effect).
2. `shuffled` and `random` agreement: |mean| <= 0.05.
3. `ridge` agreement >= 0.20 and above `shuffled` by >= 0.15 on at least 7 of 10 prompts.
4. Mutant: correlating `e_B` with another prompt's `e_A` gives |mean| <= 0.05 (the check can fail).
5. Rlog: one normal line per condition, anomaly on a missed bar.
A pass on F5 (word overlap) with a miss here is reported as "overlap without effect agreement".
