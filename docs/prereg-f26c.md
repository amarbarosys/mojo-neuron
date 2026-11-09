# Prereg F26c: dream coefficient sweep, peak chosen on held-out features (written before any run, 2026-11-09)

Source: papers synthesis row "steering strength is an inverted U" (`docs/reports/2026-11-09-papers-synthesis.md`) and
`(private notes)` section 2. Follows F26 (FAIL, bar 2) and F26b (one run at 2.5x, result in the report); neither is re-cut.
Written before F26b's result is read into any choice here; F26c runs after F26b whatever F26b says.

Setup as F26 (Qwythos-9B, layer 16, rho16 = 37.61, 8 features = top lens z, seeded controls, prompt `The`, 8 steps x 16 tokens) except:
- coefficient sweep: multiplier m in {1, 2, 3, 4, 6} on the F26 schedule (scale = rho16 * 0.2 * m * (step + 1)); 2 controls per feature per m.
- split: features ranked 1, 3, 5, 7 by lens z are the **selection set A**, ranked 2, 4, 6, 8 the **held-out set B**.
- peak: m* = the m maximising mean(feature rate - own control mean) over A among the m with mean distinct token ids >= 40 of 128 over A's feature dreams
  (collapse excluded). If no m qualifies, F26c FAILs (bar 1).
- prompt-only baseline (interp-protocols: "prompting often wins"): unsteered dream of the prompt `The` + the lens top-8 token texts of the feature
  (`The <t1> <t2> ... <t8>`), same 8 x 16 loop, rate against the same lens set.
- side-effect: the engine driver reads no log-probs, so the side-effect measure is the distinct-token count (stated; not NLL).

Bars (all must hold; a miss = FAIL, no post hoc change):
1. An m* exists (some m has mean distinct >= 40 on A).
2. On B at m*: mean feature rate >= max(3 x mean control rate, 0.02) and >= 3 of 4 held-out features above their own controls.
3. On B at m*: mean distinct >= 40.
4. Dreams: 128 tokens each, finite scales.
5. Mutant: the scorer with the B lens sets shifted by one feature must fail bar 2 (reuses `-D DREAM_MUT=1`).
Reported, not bars: the whole curve (rate, controls, distinct per m for A and B) with the shape named (interior peak = inverted U; monotone = none found);
per-feature sign share (features whose rate beats controls, per m); prompt-only rate vs steered rate at m* (a steered rate below the prompt-only one is
reported as "prompting wins").
Receipt: `.work/dreams/r4-sweep/` (`dream.jsonl` with an `m` field, `score.txt`), rlog.
