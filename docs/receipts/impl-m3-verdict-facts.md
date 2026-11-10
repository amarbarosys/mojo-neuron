## Memorisation as cheap knowledge injection (prereg verdict 2)

| k | scale | exact recall |
|---|---|---|
| 1 | 0.5 | 0.000 |
| 10 | 0.5 | 0.000 |
| 100 | 0.5 | 0.000 |
| 1000 | 0.5 | 0.000 |

M-A capacity at k=100 (scale 0.5): exact 0.000 (bar >= 0.90 -> MISSED), paraphrase mean 0.000 (bar >= 0.50 -> MISSED); k=1000 paraphrase NOT RUN (reported without a bar in the prereg, skipped for GPU time)
M-B interference at k=100: unrelated accuracy drop +0.000 (bar <= 0.01), ppl change +0.0000 (bar <= 0.01), control drop +0.000 / ppl +0.0000 -> PASS
M-C cost: 0.578 s per fact (engine capture 577.9 s + builder 0.6 s over 1000 facts) -> PASS against 5 s; LoRA step time: UNMEASURED unless the report cites a train log
control exact recall at the same k and scale: 0.000

## Facts rank falsifier (prereg docs/prereg-facts-rank.md, 20 facts, layer 16, k = 100)

| scale | gold in top 20 (of 20) | median gold NLL | median gap (lower bound when absent) |
|---|---|---|---|
| 0 | 1 | 6.97 | 3.45 |
| 8 | 14 | 3.86 | 2.69 |
| 32 | 2 | 34.62 | 7.51 |

sanity: max |dnll| at scale 0 = 0.00e+00 (must be 0)
R1 (top-20 count up by 3 or more): scale 8 True, scale 32 False
R2 (median NLL down by 2.0 nats or more): scale 8 True, scale 32 False
RESULT: MOVES (grid too small), next step to the boss

## Facts rank 2 (prereg docs/prereg-facts-rank2.md, 20 facts, layer 16, k = 100)

| scale | gold rank 1 (of 20) | in top 5 | in top 20 |
|---|---|---|---|
| 0 | 1 | 1 | 1 |
| 8 | 3 | 7 | 14 |
| 12 | 4 | 7 | 11 |
| 16 | 3 | 6 | 7 |
| 20 | 2 | 6 | 7 |
| 24 | 2 | 3 | 4 |

RK-A (rank 1 for 10 or more of 20 at some scale in 8..24): MISSED (best 4/20)
RK-B top-5 counts are in the table, no bar.

## Notes written later (2026-11-10, after the results above; annotations, not re-cuts)

- Verdict 2 as pre-registered: M-A MISSED (exact 0.000, paraphrase 0.000 at k = 100; also 0.000 at k = 1, 10, 1000 and at every scale 0.5 to 8 for exact recall), M-C PASS (0.578 s per fact, capture 577.9 s over 1000 facts; LoRA step time UNMEASURED), M-B reads PASS but is weak: the scale was picked by `pick_scale` as 0.5 because every scale tied at 0.0 recall, and at all five scales unrelated accuracy stays 1.00 on and off and ppl change is 0.0000. Implants keyed to invented-name prompts do not fire on unrelated text; that is not evidence that a working implant would be harmless.
- Control arm (random implants of equal size, `facts-results-ctl.json`): exact recall 0.000 at every k, paraphrase 0.000 at every scale, unrelated 1.00 on and off, ppl change 0.0000. The control equals the real arm on the pre-registered metrics, so there is no implant-specific gain to separate from it.
- The facts-rank table above carries the label "MOVES (grid too small)". The label is wrong; it was removed from the harness in d6bb732. The data say: scale 8 already puts the gold first token in the top 20 for 14 of 20 facts (NLL 6.97 to 3.86), scale 32 overshoots (2 of 20, NLL 34.6).
- The first facts-rank run (receipt `.work/impl-m3/old/facts-rank-void-noimplants.md`) printed NULL with BARO_IMPLANTS unset; it is VOID and was retracted.
- Facts rank 2 (scales 8 to 24, 20 facts): RK-A MISSED, best 4 of 20 with gold rank 1 (scale 12), top-5 7, 7, 6, 6, 3 for scales 8 to 24. Together: the implants move the gold first token to the top 20 and top 5 at scales 8 to 12 but not to top-1, and recall stays 0. By the prereg of rank 2 the facts line is closed as null for this builder at layer 16. Scope: 20 facts, one prompt template, one layer; k = 1000 paraphrase was NOT RUN.
- Harness: facts chain `f5782f5` (prereg and select.json), rank `7b382d5` + `c40dfd2`, rank 2 `c000fe3` + `d6bb732`; engine jobs submit `gpu-wait --shared --vram 10` (`ecfa77c`, `b09ace9`); the real arm ran before that rule (exclusive).
