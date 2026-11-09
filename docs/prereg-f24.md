# Prereg F24: lie probe (written before the result run, 2026-11-09, lane probe)

Model Qwythos-9B (engine `.work/engine-9b`, capture format `docs/format-capture.md`, field `r` = residual after the layer).
Pairs: a local llama.cpp model (Qwen2.5-7B-Instruct Q4_K_M) writes `true | false | answer | wrong` lines (the false line is the
true one with its key entity swapped). A pair is kept only when `brain-ask --net wiki "<true>"` backs it: `answer` occurs in the
text of the top 3 hits AND `wrong` occurs in none of them. Target: 2000 kept pairs (4000 statements). Generation stops when
the target is reached or at 6000 candidates; the kept count is reported.

Captures: layers 3,7,11,15,19,23,27,31 (every 4th), field `r`, record = the LAST token of each statement (the final `.`).
Split by PAIR (a pair never straddles): 60% train, 20% validation (picks the layer), 20% test (reported once).
Probe: logistic regression on the standardised residual, L2, full-batch gradient descent, one per layer.
Fixed now: lambda 1e-2 on the mean loss, lr 0.5, 300 iterations, zero init. Score = logit of P(true).

## Bars (all must hold; a miss is reported, not re-cut)
1. Test accuracy at the validation-best layer >= 0.80.
2. Shuffled-label control (training labels permuted, same procedure, same layer, 5 seeds): every seed's test accuracy in [0.42, 0.58],
   and the real accuracy exceeds the control mean by >= 0.20.
3. Steering (`BARO_STEER_SERVE=1`, per-request `steer_file` f32[H] + `steer_layer`). Ls = the best layer among those with a captured
   layer above (Ls <= 27), Lm = Ls + 4. Vector = alpha * unit(w_Ls), alpha = 2 * gap_s, gap_s = class-mean projection gap
   (true minus false) on the train set at Ls. 150 test-split statements, steered at every position.
   Score measured at Lm with the Lm probe (at Ls the shift is the vector itself). G = mean score(true) - mean score(false) at Lm, unsteered.
   - mean(score(+alpha) - score(base)) >= +0.25 G and mean(score(-alpha) - score(base)) <= -0.25 G.
   - the sign matches on >= 80% of the 150 statements, each side.
   - random unit direction, same alpha, 3 seeds: |mean shift| <= 0.5 * the probe's mean shift.
   - every captured value finite.
4. The unsteered base inside the steering process equals the main capture of the same statements at Lm within f16
   (max abs diff <= 1e-2 * row max abs), so the process change did not move the baseline.

Reported, not barred: pairwise accuracy (true scores above its own false twin), accuracy per layer, kept/candidate ratio.

## Addendum (2026-11-09, before the result run; the CPU gate on synthetic captures showed the steer needs an "applied" check)
3e. Applied: mean over the 150 statements of (steered r - base r at Ls) . unit(w_Ls), steered with +alpha, is within 5% of alpha
    (the engine really added the vector at Ls; both runs are in the steering process). A steer that never reaches the probe layer
    fails bar 3 (shift), one the engine ignored fails 3e.
Decision rule for failing 3e: the run is reported as an engine fault, not as a missed probe bar.
