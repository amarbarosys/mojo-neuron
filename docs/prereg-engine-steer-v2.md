# Pre-registered tolerances v2: engine steering gate, Gate B (review row E1)

Written and committed 2026-11-09 after the v1 readings and before any v2 reading. v1 (docs/prereg-engine-steer.md) stays on
record with its verdict.

## v1 readings (the verdicts v1 produced; unchanged)

| pack | recipe | Gate A | Gate B error / N | cos | top-1 |
|---|---|---|---|---|---|
| Qwythos-9B Q4_0 | 8:100:30,20:50:120 | pass | 2.27 N (bound 1.6 N): FAIL | 0.9984 | 271/271 |
| OrcaSAQ-27B q4 | 16:100:60,40:50:200 | pass | 2.10 N: FAIL | 0.9526 | 198/198 |
| OrcaSAQ-27B q4 | 8:100:100,56:50:300 | pass | 2.18 N: FAIL | 0.9650 | 74/13, llama gap 0.14 <= n99 0.89: explained |

Under v1 each is "A pass, B fail". The v1 bound assumed the engine-vs-llama.cpp noise is the same with and without steering;
all three readings say the noise of the steered difference is larger (steering enlarges logit magnitudes).

## v2 Gate B

Same four calibration prompts, now also run under the steering vector being tested (engine and llama.cpp both steered).
Per prompt: base noise nb_i = ||e_b,i - l_b,i||, steered noise ns_i = ||e_s,i - l_s,i||; NB = median nb_i, NS = median ns_i.
The error of the test delta is a difference of those two discrepancies, so by the triangle inequality
||de - dl|| = ||(e_s - l_s) - (e_b - l_b)|| <= ||e_s - l_s|| + ||e_b - l_b||. Bound: error <= NS + NB, with NS and NB from the
calibration prompts only (not the test prompt).

Gate B passes when: error <= NS + NB; cos(de, dl) >= 0.95; top-1 equal or llama's gap <= n99 (v1 definition); shifted-layer
mutant has cos at least 0.05 lower or error above NS + NB. The exact rule for packs with unsteered N/||l|| <= 0.05 stays.

Caveat written down in advance: the triangle bound is loose, so Gate B cannot catch small magnitude errors (a 0.95 x vector
passes it). That is why Gate A decides and Gate B is the cross-check against llama.cpp's control-vector semantics (position,
layer index, sign), which Gate A cannot see.

## Verdict wording

PASS = Gate A and v2 Gate B. v1 Gate B is printed beside it and not used to decide.

## Status (added 2026-11-09 after the boss's review)

This file is NOT a pre-registration. It was written after the v1 readings, so its bound was chosen with their outcome known; the v1
verdicts (Gate A pass, Gate B fail on all three readings) are the pre-registered result. Gate A (residual tap, same numerics, from
docs/prereg-engine-steer.md, written before any reading) alone decides whether a steering hook is correct. Gate B v2 is printed as a
diagnostic of agreement with llama.cpp's control-vector semantics and cannot turn a failing hook into a pass or a passing one into a fail.
