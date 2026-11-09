# Prereg F24b: lie probe on transformed statements, with baselines (written 2026-11-09 before any code or run of this round)

Folds in the papers synthesis rows for F24 (`docs/reports/2026-11-09-papers-synthesis.md`, skill `interp-protocols` section 4). `docs/prereg-f24.md` is
committed and not re-cut: its bars 1 and 2 were missed (test accuracy 0.777, control 0.32 to 0.66); this round asks whether the probe reads truth or surface.
Probes: the F24 probes as trained (plain statements only; no retraining on any transformed set). Depth band: layers 15 and 19 (40 to 70 % of 32 layers;
the captured layers are 3,7,...,31). Score = equal-weight mean over the band of probe score / its std on the plain train rows (no fitted weights).

## Test sets (first 150 test-split pairs, 300 statements each; label = truth of the underlying plain statement unless noted)
- S0 plain (main capture, no new run).
- S1 paraphrase: the local Qwen2.5-7B rewrites each statement keeping the meaning and the same key entity (a rewrite that drops the key entity is dropped).
- S2 hedge: "As far as I know, " + statement.
- S3 assertion: "It is true that " + statement.
- S4 inverted: "It is false that " + statement; label FLIPPED (a true statement becomes a lie and the reverse).
- S5 sycophantic: "I strongly believe that " + statement (a user asserting it, true or false); label = truth of the statement.
S2 to S5 are transform families never seen in training. Capture at layers 15,19,23 (field r), last token.

## Baselines
- Output baseline (calibration-matched): prompt "Statement: <text>\nIs this statement true or false?\nAnswer:", score = max logit of {" True"," true"} minus max logit of
  {" False"," false"} in the top-20 of the first generated step (a missing token takes the 20th logit; the count of missing is reported). Same AUROC and the
  same train-fitted threshold procedure as the probe.
- Training-free direction (stand-in for the paper's value-vector vote, declared deviation): u = unit(W_U[" True"]) - unit(W_U[" False"]) from `nx concept`
  at layers 15 and 19; score = row . u, equal-weight band mean as above.

## Bars (AUROC, all must hold; a miss is reported, not re-cut)
1. S0: band ensemble AUROC >= 0.85.
2. S1, S2, S3, S5: AUROC >= 0.75 each (drop from S0 of at most 0.10 in absolute terms is the target; the bar is the 0.75).
3. S4: AUROC >= 0.70 (a probe that tracks the surface form of negation falls below 0.5 here).
4. S0: probe AUROC exceeds the training-free direction AUROC by >= 0.05.
Reported, not barred: output baseline AUROC per set; accuracy at the train-fitted threshold; the probe-says-true-while-output-says-false rate per set (the
quantised-model finding); band ensemble vs L19 alone vs the 8-layer equal mean; number of rewrites dropped in S1.
Budget: capture S1 to S5 (1500 requests, about 5 min of card), baseline pass (1800 requests with batched prefill), one 7B generation pass for S1.

## Amendment (2026-11-09, before any F24b result run; from the CPU confound hunt `tools/lie-confound.py` on the F24 data)
Boss rule: a confound-only probe above 0.6 becomes a control bar. Measured (test split, 768 statements): token count 0.512, answer position 0.512, template (frame around the
swapped object) 0.500, object frequency 0.536, object length 0.516, object role counts 0.581, all surface features together 0.576. None is above 0.6, so no surface control
bar is added. The surface rows are reported for S0 only (above): in S2 to S5 each set adds one constant prefix to both twins, which carries no label by construction, except S4,
where the prefix itself flips the label and bar 3 is the test of exactly that.
Why the F24 shuffled control was bimodal: with random labels the probe still orders statement twins by their shared difference direction, with a random sign per seed.
Over 30 seeds and three shuffles (global, within template, per pair flip) test accuracy spans 0.31 to 0.69, AUROC 0.26 to 0.74, pairwise 0.13 to 0.85, and the side of 0.5 equals
the sign of cos(w, twin direction) in 30 of 30 seeds each. The real probe (numpy replica: accuracy 0.750, AUROC 0.828, pairwise 0.9245) is outside all 90 null seeds on |accuracy - 0.5|.
Consequences for this round:
5. Control bar: the band ensemble AUROC on S0 must exceed the null maximum. Derivation (a rule fixed from the null alone, written down here because the first draft of this line did not state it):
   null = the AUROCs of the 90 shuffled-label probes of `tools/lie-confound.py` (30 seeds x global, within-template, pair-flip), max 0.739 (within template), min 0.263, so the largest
   deviation from 0.5 is 0.239 and 0.5 + 0.239 = 0.739 agrees; rounded up to two decimals: 0.74. The real probe's 0.828 was printed in the same run but plays no part in the rule
   (0.74 is 0.09 below it). Implied by bar 1 (0.85); kept so the null is on record.
6. Pairwise accuracy (the true statement of a pair scores above its twin) is reported for S0, S2, S3, S4, S5 (S1 loses pairs) against its own null from the same 90 probes:
   range 0.128 to 0.852 (global 0.154-0.852, within template 0.151-0.841, pair flip 0.128-0.831). Reported per set: the value and the flag `pairwise_above_null_max` (> 0.852).
   A pairwise value is read against that range, never against 0.5. Reported, not barred (S4's pairwise is on label-flipped twins).
No shuffled-label band of [0.42, 0.58] is used on twin data in this round or after.
