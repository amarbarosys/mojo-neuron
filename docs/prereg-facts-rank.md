# Facts falsifier: does an implant move the gold first token toward the top 20? (written 2026-11-10 before the run, boss instruction)

Context: the facts real arm (`.work/impl-m3/facts-results-real.json`, committed prereg `docs/prereg-facts-run.md`) recalled 0.000 at every k and scale; M-A is MISSED and stays so. This note is the cheapest check of one hypothesis (from a reading of fact 0, not yet tested): the gold first token lies far outside the top 20 and the scale grid (max 8) cannot lift it. No bar of `prereg-implants.md` changes; no retry of the facts verdict.

Setup. First 20 facts of the unknown list (the same order as `facts-train.json`), implants file of the real arm, layer 16, k = 100 (`rows_cum[100]`, all 20 facts are keyed). Prompt = the exact question, n = 1. Scales 0 (`implants: 0`), 8, 32 (32 is above the grid). Per fact and scale:
- `nll`: absolute NLL of the gold first token, from the steer lens (`slens_tgt`, one-context series). Sanity: at scale 0 `dnll` = 0.
- `top20`: whether the gold first token is in the engine's top-20 list (`logits_topk`), its rank there, and the gap logit(top1) minus logit(gold) when present (else minus the 20th logit, a lower bound).

Added after a void first run (BARO_IMPLANTS was not set, implants inert, every scale identical; receipt moved to .work/impl-m3/old/): the run is VOID unless at least one fact changes its gold NLL between scale 0 and 32. Bars unchanged.

Close as NULL (facts line ends, no further implants-for-facts work) unless at least one of these holds at scale 8 or 32:
- R1: the number of the 20 facts with the gold first token in the top 20 rises by 3 or more over scale 0;
- R2: the median gold NLL falls by 2.0 nats or more against scale 0.
If R1 or R2 holds, the line is reported as "moves, grid too small" and the next step goes to the boss; nothing is claimed about recall. Sample is 20 facts, one prompt template, one layer: a null here says nothing about other layers or keys.
