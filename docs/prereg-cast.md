# Prereg CAST: persona applied only when a condition direction fires (written before the result runs, 2026-11-09, lane agent)

Source: papers synthesis (CAST condition gate). Scope: the REQUEST-level gate. One decision per request from the last prompt token; per-token gating needs an
engine change (not done, said in the report). `nx cast fit` builds the condition direction and picks layer and threshold; `nx cast gate` returns the steer
vector for one request: the persona vector when the gate fires, exactly zeros when not (so a non-firing request is the unsteered request).
Condition = "the prompt is French". Labelled set `tests/data/en-fr-60.tsv` (61 pairs, one French and one English sentence each), captured with
`scripts/cap-texts.sh` (French lines 1..61, English lines 62..122, layers 3,7,...,31, field r, last-token residual). Split by pair: pairs 1..41 train,
pairs 42..61 test (a pair never straddles).
Score = cosine(r - mu, d) where mu = mean residual of all train prompts at that layer, d = unit(mean(French train) - mean(English train)) at that layer.
Selection on train only: layer and threshold tau maximise 5-fold cross-validated balanced accuracy over train pairs (folds by pair), tau on a grid of 41 values
between the lowest and highest train score of that layer, ties -> lower layer, then lower tau. The test pairs are scored once.
Persona vector for the gate test: the F3 vector `mean(French) - mean(English)` direction at the chosen layer, scale 1 (BARO_STEER layout from `nx concept`).

Bars (all must hold, a miss = FAIL, no post hoc change):
1. Held-out accuracy (40 test prompts) >= 0.95 at the train-chosen layer and tau.
2. Shuffled-label control: labels permuted within train (20 seeds, whole pipeline including selection): mean held-out accuracy <= 0.60.
3. Gate behaviour: for every test prompt the gate output is byte-equal to zeros when the score < tau and byte-equal to the persona vector when score >= tau; the
   persona vector is not all zeros (so the check can fail).
4. Threshold width: held-out accuracy >= 0.90 for every tau in +-25% of the chosen tau (relative to the score range of the chosen layer), and held-out AUC >= 0.98.
5. Out of domain: 300 English statements of `.work/cap/qwythos-lie` (requests 1..300, English by construction): false-fire rate <= 0.05 at the chosen layer and tau
   (the lie capture holds the same layer grid 3,7,...,31, so the chosen layer is always available there).
6. Mutant `-D CAST_MUT=1` (comparison inverted) must FAIL bars 1 and 3 in the gate.
7. One normal rlog line per fit and per gate decision batch with accuracy, tau, layer; a non-finite score gives an anomaly line.
Report only: accuracy per layer (train CV and test), score histogram per class at the chosen layer.
Tier 2 (not preregistered here, UNRUN): effect of the gated persona through the engine on generations; a separate prereg when the GPU slot is used.

Receipts: `.work/cast/<run>/{fit.json, gate/*.f32}`, `tests/cast.sh`.
