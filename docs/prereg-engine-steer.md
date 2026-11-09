# Pre-registered tolerances: steering gate for engine.mojo (review row E1)

Written and committed before the 27B gate is run again. Nothing below may be changed after a reading; a change needs a new
file with its own date and the old verdict stays on record.

## Why the old rule is withdrawn

`scripts/engine-steer-check.sh` accepted a noisy pack on delta cosine >= 0.95 and error <= 1.5 x the unsteered discrepancy,
dropped top-1 there, and the 1.5 was chosen after the first 27B reading. A delta of the right direction and the wrong size
passes that rule. Demonstration (run before any fix, see report): the engine fed a vector scaled by 0.6 while the oracle
used the full vector.

## Gate A: residual tap, same numerics, no oracle noise (decides)

The engine's own `BARO_DUMP` holds the residual after every layer at the first generated step; the steer add happens before
that copy, so for a vector v added after layer L (one or several layers, first steered layer Lf):

1. For every layer l < Lf the steered dump equals the unsteered dump byte for byte.
2. At layer Lf: ||(x_steer - x_base) - v|| <= 1e-6 * ||x_base|| + 1e-4 * ||v||  (f32 rounding of x + v, nothing else).
3. At every layer above Lf the difference is finite and non-zero.
4. Mutants that must fail Gate A: the engine fed 0.6 x v, fed v shifted one layer up, fed an all-zero file (rule 3 and 2).

Gate A runs on every qwen35 pack (9B, 27B).

## Gate B: agreement with llama.cpp (reported, decides only together with A)

Calibration set, independent of the steering prompt: four prompts ("The weather today is", "Write a poem about the sea.",
"def fib(n):", "Bonjour, comment"). For each, unsteered next-token logits from the engine (e) and llama.cpp (l):
N = median over the four of ||e - l||_2, n99 = median over the four of the 99th percentile of |e - l|.

For a recipe (steered logits e_s, l_s; delta dl = l_s - l_b, de = e_s - e_b), Gate B passes when all hold:

1. error ||de - dl|| <= 1.6 N. Derivation, not fit: if the engine and llama.cpp each carry noise of size N, the difference of two
   noisy deltas has noise sqrt(2) N = 1.41 N; 1.6 N leaves 13 % margin. (The two 27B readings that came before this
   file, 1.46 N and 1.52 N, fall under it; that is a coincidence of the derivation, not its source, and a reader should weigh it so.)
2. cos(de, dl) >= 0.95.
3. top-1: the engine's steered top-1 token equals llama.cpp's, or llama.cpp's own logit gap between its top-1 and the
   engine's top-1 is <= n99 (the engine's choice is inside the per-logit noise).
4. The layer-shifted mutant has cos at least 0.05 lower, or error above 1.6 N.

Packs whose unsteered discrepancy N / ||l|| is <= 0.05 (9B reading: 0.021) are also held to the old exact rule
(relative error <= 0.10 and the same top-1).

## Verdict wording

A pack is PASS when Gate A passes and Gate B passes; Gate A alone passing is reported as "A pass, B fail (noise)", never as PASS.
