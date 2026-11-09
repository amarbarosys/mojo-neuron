# Prereg F29b: self-surgery agent on groups, propose only (written before any run, 2026-11-09)

Supersedes `docs/prereg-f29.md` for the run. That document is committed and unchanged, F29 never ran; the boss pointed out (A2c) that single-neuron ablation on Qwythos-9B
layer 16 sits at the engine's bf16 noise floor (`causal` report: every sampled neuron scores KL 7.7e-6 to 1.2e-5, gain 10 on one neuron 1.9e-5, gain 100 1.4e-3), so the F29 plant (one neuron
x8) and its single-unit search could not be seen. F29b searches and plants groups, the unit A2c-group measures (`causal/docs/prereg-a2c-group.md`). Needs: A2c group tooling
(`prerank`, group `gain` list in one request, `slens_foil` logit-difference metric) and A3 for the printed surgery command.

## What the agent does (scripted search, no LLM, no GGUF written)
Input `spec.json`: bad token T, foil F, eval prompts (the 50 capital prompts of A2c, `tests/data/capitals.tsv`), layer 16.
1. Floor first: the neutral-perturbation floor of A2c (5 random 128-groups at gain 0.99), printed before anything is scored; every effect below is quoted as a multiple of it.
2. Pre-rank all neurons by direct effect on logit(T) - logit(F) (A2c `prerank`), take the top-k groups for k in {8, 32, 128}.
3. Simulate each group through the engine `gain` list at gain 0 and 0.5 on the eval and on 16 neutral prompts; score = share of eval generations containing T, plus NLL change on the neutral prompts.
4. Write the best proposal to `.work/surgery/proposals/<id>.md`: the group (layer:neuron list), gain, floor multiple, both numbers before and after, the exact `nx surgery` command. If no group beats 3 x floor, the proposal says `no action`.

Plant: the engine runs with the pre-rank top-128 supporters of T (computed independently of the agent run, saved as `.work/surgery/plant.json`) at gain 8: a bad behaviour that is a
property of this run's gains, not of the weights.

## Bars (all must hold; a miss = FAIL, no post hoc change)
0. Plant effective (checked before the agent runs): share of eval generations containing T under the plant >= 0.30 and <= 0.05 unplanted, and the plant's logit-difference change >= 3 x floor.
   If this fails the run stops there and F29b is FAIL "plant not visible", reported with the numbers (the next prereg would raise the gain).
1. Found: the proposed group contains >= 50% of the planted 128 neurons.
2. Fix in simulation: share with the proposal's gain applied on top of the plant <= 0.25 x the planted share, and neutral-prompt NLL change <= 0.1 nats.
3. Control: with no plant the agent proposes `no action` (unplanted share < 0.05, no group above 3 x floor).
4. No write outside proposals/: `tools/nowrite.sh .work/surgery/proposals <repo> <model dir> <scan dir> -- <agent>` passes; `tests/nowrite.sh` (own gate, mutant that writes beside) passes.
5. Rlog: one normal line per run; anomaly line on a missed bar 0 to 2.
Reported without a bar: group size k and gain chosen, multiple of the floor, per-prompt share, wall time.
Not claimed: single-neuron surgery effect; that the found group is the mechanism of any real behaviour; anything beyond the planted artefact.
