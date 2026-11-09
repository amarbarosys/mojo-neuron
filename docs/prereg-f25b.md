# Prereg F25b: thought diff in effect space (written 2026-11-09 before any code; addendum to prereg-f25, which stays as committed)

Folds in the synthesis row "cross-model maps: compare effects, not token overlap" (B3, F5, F25). prereg-f25 flags features by activation KS distance. F25b asks whether a
flagged feature DOES something different: for each of the top 20 flagged features f (and 20 unflagged controls matched by activation scale), steer model A and model B
with the same B3-mapped direction on a fixed 32-prompt set; the effect of f in a model = the vector of mean logit-difference changes over 16 fixed probe tokens.

## Bars
1. Self gate: a model against itself has effect-vector correlation >= 0.99 for every f (the engine is deterministic; lower is a mapping bug).
2. Flagged features have lower A-vs-B effect correlation than the controls: median over flagged < median over controls, and the gap is larger than the gap
   between two bootstrap halves of the prompt set (seeded, reported).
3. Shuffled map (B3 map rows permuted) gives effect correlation near 0 (|median| <= 0.15).
Reported, not barred: per-feature effect correlation next to its KS distance (rank correlation of the two lists), contrastive lens (`lens-diff A B`) tokens of the top 5.
