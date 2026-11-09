# Prereg F26b: model dreams at mid scale (written before the run, 2026-11-09, boss go on a pane)

Follows F26 (`docs/prereg-f26.md`, verdict FAIL, bar 2 at 3/8, unchanged). F26's post hoc 5x run scored 8/8 but collapsed into token loops (4 to 30
distinct tokens of 128), so it is not a pass. F26b tests one scale between them, chosen before the run: **mult 2.5** (steer scale rho16 * 0.5 * (step + 1),
i.e. 0.5 .. 4.0 rho), a single run, no second multiplier (no forking path). Everything else as F26: same engine, layer 16, rho16 = 37.61, same 8 features,
same seeded controls, base dream, metric and bars 1 to 4 and 6 of F26, same scorer plus bar 7.

Bars (all must hold; a miss = FAIL, no post hoc change):
1 to 4 as in F26 (mean feature rate >= max(3 x control, 0.02); >= 6/8 features above their own controls; 128 tokens per dream; base <= features).
7. No collapse: mean over the 8 feature dreams of distinct token ids >= 40 of 128 (the prereg-scale r1 run had 33 to 68, mean 52; r2 at 5x had mean 21).
   Checked by `score ... distinct=40`; CPU test: the planted stand-in engine (at most 32 distinct) must fail it, the plain one must pass it.
Receipt: `.work/dreams/r3-mult2.5/{dream.jsonl,score.txt}`, rlog line.
