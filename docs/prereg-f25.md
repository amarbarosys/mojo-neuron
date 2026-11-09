# Prereg F25: thought diff (written 2026-11-09 before any code; needs B3, lane probe)

Compare two models (Qwythos vs its base, or two quants of one model) on one token set: per feature of the shared basis (B3 map output) the
two activation distributions over the same tokens; statistic = Kolmogorov-Smirnov distance D_f. Listed: the top 20 features by D_f with their lens tokens.
Noise floor: the token set is split in three random thirds (seeded); floor = max_f KS(model A on third 1, model A on third 2).

## Bars
1. Self gate: a model against itself: KS(A third 3, A third 1) <= 1.5 * floor for every feature, i.e. zero features flagged (flag = D_f > 1.5 * floor).
2. Real gate: between two different quants (Q4_0 pure vs Q4_K_M of Qwythos-9B) >= 1 feature is flagged; between Qwythos and a different model of the
   same size the flagged count is larger than between the quants.
3. Mutant: shifting one feature's activations in model B by 0.5 of that feature's standard deviation is flagged and ranks in the top 20.
Reported, not barred: the flagged features' lens tokens, the fraction of features flagged, the token count.
