# Prereg F1b: persona strength and baselines (with the E1 serve hook)

Folds in papers-synthesis rows "Steering strength is an inverted U" and the single-reader CAST idea (`interp-protocols` section 2). Adds to
`prereg-f1.md` (committed, not changed). Starts when E1 lands (the boss forwards the sha).

- `nx persona tune {name, prompts_train, prompts_held}`: sweep layer x scale (3 layers x 6 scales incl. 0) of the persona's vector through the
  engine `steer_file`; metric = mean logprob rise of the persona's target tokens (its lens `up` tokens) on the held-out prompts; the stored
  recommended scale is the argmax on HELD-OUT, chosen with the train prompts only.
- Side effect: mean NLL on 16 neutral held-out texts at every cell; the recommended cell has NLL <= baseline NLL + 0.5 nat, else the next lower scale.
- Per-input sign: share of held-out prompts whose target rises at the recommended scale, printed (can be near 50 %).
- Baseline: the same target measured with a one-line system prompt naming the persona; the report states which wins; no bar on winning.
- Bars: scale 0 reproduces the unsteered logits (max abs diff 0, the F1 engine gate); the curve is not monotone-up (a point past the peak is lower), or
  the report says it never peaked in the swept range; every cell is in the Rlog.
- CAST (stretch, not barred): apply the persona only when a condition direction (mean of the prompt-side residual rows) fires; the stretch reports
  the share of off-topic prompts left unsteered.
