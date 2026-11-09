# Prereg F26: model dreams (written before the result run, 2026-11-09)

Setup: Qwythos-9B, engine `engine-9b` (serve, `BARO_STEER_SERVE=1`, greedy, spec off), layer 16 FFN neurons.
A dream of neuron f = start from the neutral prompt `The`, repeat 8 steps: the request carries the prompt plus every token
generated so far, steers the residual after layer 16 along the unit decoder column of f (`ffn_down` column f) with scale
`rho16 * 0.2 * (step + 1)` (rising 0.2 .. 1.6 rho), generates 16 tokens, feeds them back. 128 tokens per dream.
`rho16` = median residual L2 norm at layer 16 from an unsteered capture (`tools/rho-cap.py`).

Units: 8 features = the 8 layer-16 neurons with the highest top-1 lens z (store `16.lens.bin`, k=32); 4 controls per feature =
other neurons of layer 16, drawn from a fixed seed (xorshift 0x9E3779B97F4A7C15), never the feature. A control is dreamed
exactly like a feature.

Metric: `overlap(dream, f)` = share of the 128 generated token ids that are in the lens top-32 ids of f.
`feature rate` = overlap(dream_f, f); `control rate` = overlap(dream_g, f) for each control g of f.

Bars (all must hold, any miss = FAIL, no post hoc change):
1. Mean feature rate >= max(3 x mean control rate, 0.02).
2. At least 6 of 8 features have a feature rate strictly above the mean of their own 4 control rates.
3. Every dream has exactly 128 tokens and finite scale values; every dream file line parses.
4. Unsteered base dream (scale 0, same loop) rate vs the same 8 lens sets <= mean feature rate (sanity: steering, not the prompt, moves it).
5. Mutant: scoring dream_f against the lens of the NEXT picked feature (shifted by one) must give a mean ratio to its controls
   below 3 (the check can fail); the scorer builds with `-D DREAM_MUT=1` for this and the gate demands FAIL there.
6. Image: one ComfyUI image of the last dream text exists (png/jpg, > 10 KB), opened detached for the maintainer. Not a bar on quality.

Receipts: `.work/dream/<run>/dream.jsonl` (one line per dream step), `score.txt`, rlog lines (normal; anomaly when a bar is missed).
