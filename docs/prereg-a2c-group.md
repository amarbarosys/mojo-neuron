# Prereg A2c-group: group ablation by direct effect, scored by target logit difference (written 2026-11-09 before any run of it)

Follows the boss's reading of A2/A2b: single-neuron ablation on Qwythos-9B layer 16 is below the engine's bf16 noise floor, so this tests groups and a sharper metric. `docs/prereg-a2.md` (run 1, FAIL), `docs/prereg-a2b.md` (PASS) and
`docs/prereg-papers-fold.md` (its A2c pre-rank section; this document uses that pre-rank as the way to choose the group) are not changed.

## Setup (fixed)

- Model: Qwythos-9B Q4_0 pack on engine.mojo (baro neuron2 with `slens_foil`, built for this run), layer 16, ablation mode = zero (gain 0) of the whole group at once ("gain" list in one request), clean prompts only.
- Prompts: 50 known-fact prompts "Q: What is the capital of <country>? A:" (`tests/data/capitals.tsv`, 50 countries the unmodified model answers right, checked in the run). Target = first token of the right capital, foil = first token of the capital of the
  next country in the file (a different, plausible answer). Metric = mean over prompts of the change in logit(target) - logit(foil) at the answer position, ablated minus unablated (the log-softmax normaliser cancels). Negative = the group supported the target.
- Pre-rank (CPU, `prerank`): per prompt `de_n = a_n * (gamma * (U[target] - U[foil])) . down_n`, `a_n = silu(gate_n . x) * (up_n . x)` from the A1 capture of `x`, averaged over the 50 prompts; weights from the Q4_K_M GGUF (the engine pack is Q4_0 of the same model: a
  proxy, stated). Ranks all 12288 neurons by mean `de` (descending = supports the target; ascending = opposes it).
- Groups: top-k by `de` (supporters), bottom-k by `de` (opposers), and 5 random groups of equal size (seeded), for k = 8, 32, 128. Each group scored in one engine process; the unablated reference is the same layer-loop path (gain 16:0:1 on every prompt).
- Floor first (its own stage, before any group is scored): the metric under a neutral perturbation: 5 random groups of 128 neurons at gain 0.99. floor = the largest |mean change| of those five. The floor is printed before the group stage starts and the bars use it as measured.

## Bars (any miss is reported as missed; no bar is re-cut)

| bar | statement | pass |
|---|---|---|
| F | floor measured and printed before the group stage; the same 5 draws repeated in a second engine process give the same numbers to 1e-6 relative | holds |
| S | supporters: mean change of the top-k group | <= -3x floor at some k in {8, 32, 128} |
| R | random control: all 5 random groups at that k | each |change| <= 2x floor, and the supporters' change is at least 3x the largest random one |
| O | opposers: mean change of the bottom-k group at the same k | >= +3x floor (the opposite sign: the pre-rank knows direction) |
| D | dose: |change| of the supporters at k = 8, 32, 128 | non-decreasing |
| P | per-prompt: Spearman between the supporters' first-order prediction `-sum de` over the group and the measured per-prompt change, at the k of bar S | rho >= 0.3 |

Reported without a bar: measured over predicted ratio per k, the same table for KL (the A2 metric) so the two metrics sit side by side, cost of `prerank` (seconds, CPU).
Verdict: PASS only if S, R, O, D hold at one common k and F holds; P is reported either way. A result where S fails while R holds says "no group effect detectable at k <= 128 on this metric", never "the pre-rank is wrong".
Not claimed: anything about other layers, other prompt families, or that ablated groups are the "mechanism" of the answer (zero ablation of a group can also break downstream computation).

## Amendment written before the run (2026-11-09, GPU batching asked for by the boss)

Floor groups (each of the five draws twice, labels fA*, fB*) and all scored groups run in ONE engine process, in request order floor first; the verifier prints the floor before it reads any group row. Bar F's repeat is therefore a repeat
inside one process, not across two processes. The engine for this run is the neuron2 build that adds `slens_foil` (the logit-difference field `dld`).

## Result (2026-11-09, one run, engine neuron2b 2caa2d0 binary, receipt `.work/group-check/`): MISSED

Floor (5 random groups of 128 at gain 0.99, logit-difference change, mean over 50 prompts): 3.03e-3 (values -2.0e-3, +1.3e-4, -1.9e-3, -3.0e-3, -1.6e-3); the repeat of the same five groups in the same process is identical (bar F holds).
Pre-rank cost 13.6 s CPU for 50 facts and all 12288 neurons (P-C holds). Change of logit(target) - logit(foil), mean over prompts (supporters = top-k by predicted direct effect):

| k | supporters | opposers | random x5 | KL top / bottom / random |
|---|---|---|---|---|
| 8 | +5.5e-2 | -5.3e-3 | +1.0e-2, +1.8e-2, -5.2e-3, +2.1e-4, -5.1e-3 | 2.2e-3 / 4.5e-4 / 4.6e-5 |
| 32 | -4.7e-2 | -4.4e-2 | +1.8e-2, +1.7e-2, -8.1e-3, +1.7e-2, -3.9e-2 | 6.7e-3 / 1.3e-3 / 2.1e-4 |
| 128 | +1.1e-1 | -2.0e-1 | -6.9e-2, -6.5e-3, -8.0e-2, +4.6e-2, +1.9e-2 | 2.0e-2 / 6.8e-3 / 1.2e-3 |

Bars: S holds only at k = 32 (-4.7e-2 against -3x floor = -9.1e-3), but R fails there (random groups reach |3.9e-2|, over 2x floor, and the supporters are not 3x the largest random one), O fails (opposers at -4.4e-2, wrong sign), D fails (|change| 5.5e-2, 4.7e-2, 1.1e-1 is not non-decreasing), P not evaluated (no common k). Verdict: MISSED, the pre-registered words apply:
no group effect on this metric that beats random groups at k <= 128.
Annotation, written after seeing the table, not a bar: the groups chosen by predicted direct effect, in either direction, move the whole next-token distribution 10 to 50 times more than random groups of the same size (KL 2.2e-3 against 4.6e-5 at k = 8), so the pre-rank finds neurons that matter by magnitude;
the sign of the predicted effect on this logit difference did not hold (supporters raised the margin at k = 8 and 128). A check of the pre-rank's sign convention against the engine (one neuron, scaled by hand) was not done and is the cheapest next test before blaming the idea.
