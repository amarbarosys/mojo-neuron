# Facts late layer: the same builder at the last layers (written 2026-11-13 before the run)

Claim under test: implants missed top-1 because they sat at layer 16. The value is an unembedding direction (right minus wrong). At layer 16 it passes
16 more layers that rotate it; facts-rank2 (`docs/receipts/impl-m3-facts-rank2.md`) got rank 1 for at most 4 of 20 facts but top 20 for 14 of 20. At the
last layers the same direction lands almost directly on the logits. The G0 grid (12, 16, 20, 24) never tested this.

Setup: the same 20 unknown facts as facts-rank2 (`facts-known.json`, first 20), exact prompt, n = 1, `logits_topk` 20. One implant per fact keyed at the last
prompt position (A1 capture of field x); right = gold first token, wrong = the model's own top-1; builder `src/causal/implants` unchanged; k = 20.
Layers 28, 30, 31. Scales 0 (implants off), 0.5, 1, 2, 4, 8. Control arm: `control_sets` (random keys and values, same norms, gates, tau), same grid.

Bars, fixed now:
- LL-A: at some layer and scale in 0.5..8, the gold first token is rank 1 for 16 or more of 20. PASS, else MISSED.
- LL-B: at that layer and scale, the control arm has rank 1 for 2 or fewer of 20.
- LL-C (reported, no bar): at that layer and scale, how many of 10 unrelated TRIVIA prompts keep their implants-off top-1.
- Void if no top-1 changes anywhere in the real arm.

A PASS means first-token top-1 on 20 keyed facts with the exact prompt. It says nothing about paraphrase, multi-token recall, k = 1000 or skills; each needs
its own prereg. A MISSED closes the late-layer variant of this builder.
