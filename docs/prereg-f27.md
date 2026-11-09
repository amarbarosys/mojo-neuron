# Prereg F27: memory locator (written 2026-11-09 before any code; needs A2, A3, lane probe)

Input: a fact prompt ("The capital of France is") and its answer token. Steps: (1) rank layers by ablation damage on the answer token
(A2 scan restricted to the answer position, damage = drop of the answer logprob), (2) inside the top-3 layers rank neurons by the same damage,
(3) the edit: A3 surgery that scales the top-k neurons' down columns (k = 1, 2, 4, 8, first k that flips the answer) written as a NEW GGUF in `.work/surgery/`
(never over the source file), (4) check on 5 paraphrases of the fact and 5 unrelated facts (fixed list in `tests/data/facts.tsv`, written before the run).

## Bars (reported against these, not re-cut)
1. The edited model answers a different top-1 token than the original on >= 4 of the 5 paraphrases.
2. >= 5 of 5 unrelated facts keep their top-1 token and the mean logprob drop of their answer is <= 0.1 nats.
3. The locator finds the unit set with a single resident engine process per scan (no per-neuron relaunch); wall time and the number of scored units are reported.
4. Control: the same procedure on a random neuron set of equal size changes <= 1 of the 5 paraphrases.
A missed bar is a result: the report names which fact types fail (a single-neuron edit is not expected to work on every fact).
