# prereg B2: feature steer moves its lens tokens (written before the result run, 2026-11-09)

Features: 8 transcoder features of Qwythos-9B layer 16 (`tc_train` run `tc-qw`, F=16384), chosen by the largest held-out activation sum among
features that fire on at most 25x the mean firing count (no always-on units). Lens of a feature = top 8 tokens of `lens --tc` (decoder row, unit,
times output norm, through W_U). Steer = unit decoder row x dose x mean |y| of the capture (the feature's own write scale), added after layer 16
(engine `steer_file`, `steer_layer`), doses 2, 8, 32 x mean |y|; 6 prompts of 32 corpus tokens, one next token each, `logits_topk` 20.

Score of (lens i, steer j) = |lens_top8(i) intersect steered_top20(j)|, mean over prompts. diag = mean over i = j, off = mean over i != j, base = lens
overlap with the unsteered top 20.

## Bar
At the best dose: diag >= 2 x off AND diag - base >= 0.5 tokens. Missed -> reported as missed. The transcoder behind the features missed its own
FVU bar (B1), so a miss here is a statement about these features, not about the method.
