# Prereg A2c and G addendum: checks folded in from the papers synthesis (written 2026-11-09 before any run of them)

Source: `docs/reports/2026-11-09-papers-synthesis.md` (master 898d046) and the skill `interp-protocols` sections 1 and 5. This adds checks. `docs/prereg-implants.md` (G0, 0604f51 and its
amendments), `docs/prereg-a2.md` and `docs/prereg-a2b.md` stay exactly as committed; nothing here re-cuts a bar in them. G2 and G3 result runs wait for the maintainer's read of G0; this file is read with it.

## A2c: direct-effect pre-rank, mode and metric printed

1. Every scan prints and logs its ablation mode (zero, or gain k), the corrupt setting (clean prompts only, no corrupted run, no mean or resample ablation) and its metric (KL of the next-token
   distribution plus NLL change of the target token; not a logit difference). Done in `scan` (stdout, rlog args, Brain note). Scores are not comparable across modes.
2. `prerank` (to build): for a layer, neuron n and prompt contexts, a cheap linear direct effect `de_n = a_n * (gamma * U[target]) . down_n`, with `a_n = silu(gate_n . x) * (up_n . x)` computed on the CPU
   from the A1 capture of `x` and the layer's weights, `gamma` the output-norm weight, `U[target]` the unembedding row of the prompt's target token. Averaged |de| over the prompts ranks all 12288 neurons.
3. Bars, on layer 16 with the 16-prompt dev slice of `tests/data/eval64.txt`, exact `scan ctl=0.99` over the union of the pre-rank top 64 and 256 random neurons (seeded):
   - P-A: Spearman rho between `de` and the exact excess over those 320 neurons >= 0.4, p < 0.01.
   - P-B: the pre-rank top 64 contain at least 5x as many detectable neurons (excess > 2x the median kl_ctl) as 64 random neurons do, and at least 3 in absolute terms; otherwise the pre-rank is reported as
     not predictive at this layer and the scan stays a list scan.
   - P-C: the pre-rank step costs under 60 s CPU for a full layer.
   Not claimed: that a pre-rank can replace the exact ablation, only that it picks where to spend it.

## G addendum (applies to every G1, G2, G3 and G4 run; checks added to G0, none removed)

- R: regression set after every implant batch. After building the prefix sets k = 1, 10, 100, 1000 (each a superset of the one before): (R-1) locality: the 100 unrelated items keep their accuracy (drop <= 1 pp, as
  G-C) AND their next-token distribution (mean KL against implants off <= 1e-3 outside the keyed prompts); (R-2) earlier implants: the mistakes fixed by set k_i are still fixed by every larger set (retention >=
  95 %, reported per step; a larger set that breaks earlier ones is reported as a chain collapse); (R-3) general ability: the windowed perplexity bar of G-C at every k, not only the last.
- S: symmetry and synonym probes for the memorised facts (verdict 2). For each fact held at k = 100: (S-1) 3 synonym paraphrases are already in M-A; (S-2) the reversed question ("Which country has the capital
  Plinth?") is scored and reported, no bar (an implant keyed on the forward question is not expected to answer it); (S-3) syntax check: mean NLL change of 50 neutral sentences that contain the implanted names,
  implants on against off, <= 0.1 nats; a fact edit that damages ordinary sentences about the name is not knowledge. S-3 is a bar, S-1 and S-2 are reported.
- B (G4 baselines). Besides `s_with`, `s_0`, `s_impl`, `s_ctrl`, `s_vec` of G0: `s_short` = the skill prompt cut to its first sentence (at most 20 % of the tokens), prompt only. Implants count as useful for a skill only if
  `s_impl - s_short >= 0.10` on test in addition to D-B; otherwise the verdict says "no better than a shorter prompt". Prompt-distilled weights (a LoRA trained on prompt-given outputs) are not run: the
  Qwythos trainer has no ready recipe here, so the baseline is reported NOT RUN, not approximated.
- Reporting: the control (random implants of equal size) and every baseline appear in the same table as the implants; a bar that only holds against the unmodified model and not against `s_short` or the control is not a pass.
