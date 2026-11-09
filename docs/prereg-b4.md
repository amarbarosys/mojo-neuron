# prereg B4 (+ B1 seed stability, B3 stitching): written before any of these results, 2026-11-09

Folds in the rows of `docs/reports/2026-11-09-papers-synthesis.md` and the `interp-protocols` skill (sections 3, 2) for lane features.
Earlier preregs (prereg-b1, prereg-b2, prereg-b3) are not changed.

## B1 add-on: seed stability and neuron-feature share (0.5B rehearsal transcoder, layer 12, F=4096, K=32)
Second training run, seed 2, same data and steps as the seed-1 run `tc-q05`. `tcstab`: per feature of run 1 the best cosine of its decoder row against
all decoder rows of run 2.
- stability: share of features with best cosine >= 0.7 is reported; bar >= 0.30. (Random 896-d unit vectors almost never exceed 0.2.)
- neuron-feature share: share of run-1 features whose decoder row has cosine >= 0.5 with some unit W_down column of layer 12 (the model's own neurons).
  Reported, no bar: it says how much of the dictionary is single neurons.

## B3 add-on: stitching NLL (Qwen2.5-0.5B layer 12 -> vibethinker-1.5B layer 14, the xmap pair)
Stitch = A's layer-12 residual through the ridge map M, then B's blocks 15..28, final norm, head. NLL on the held-out windows.
Stitch ratio = (NLL_stitch - NLL_B) / (NLL_control_stitch - NLL_B), control = the shuffled-fit map. Bar: ratio <= 0.5 (the real map recovers at least half
of what the shuffled map destroys). Effect-space agreement (steer-effect correlation) is deferred and is stated as such in the report.

## B4: labels, tested the way the label will be used (0.5B rehearsal features, CPU pipeline, labeller = local qwen2.5-7b-instruct via llama.cpp)
12 features, chosen like B2 (largest held-out activation sum, not always-on). Dossier = 10 top contexts (8 tokens either side, firing token marked).
Label = one short phrase. A label is KEPT only if every check passes:
1. **steer-first**: steering with the feature's decoder row (after block 12, dose 8 x mean |y|) raises the log-probability of the label's first word
   on 6 neutral prompts by more than it does for the median control feature (other features' labels' words). Lens agreement is reported, not required.
2. **fire check**: 10 sentences the labeller writes for the label vs 10 written for other features' labels; the feature's max activation, AUROC >= 0.80.
3. **non-top precision / recall**: precision = share of contexts at activation ranks 50-500 that the labeller judges (yes/no) as matching the label, >= 0.50;
   recall = share of the 10 label sentences on which the feature fires above its held-out 99th-percentile max-activation, >= 0.50. Top-context precision is printed beside.
Controls and global bars:
- **blind/shuffle**: labels shuffled across features: mean AUROC of check 2 <= 0.60, and the real mean AUROC exceeds the shuffled mean by >= 0.20.
- **synthetic units**: 8 synthetic features with known labels (direction = mean x of 40 sentences on a topic minus the background mean, topics: animals, cooking,
  music, sports, law, medicine, space, finance), dossiers built identically; the labeller must recover the topic (labeller judge yes) for >= 6 of 8; a
  shuffled-dossier control <= 2 of 8.
- **flip rate**: relabel every feature once at temperature 0.7; reported (share where the judge says the two labels differ).
- Report: kept share. A missed bar is reported as missed; the bars do not move.
