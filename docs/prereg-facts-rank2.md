# Facts rank 2: where between scale 8 and 32 is the gold first token top-1? (written 2026-11-10 before the run, boss instruction)

Basis: the corrected facts-rank run (`.work/impl-m3/facts-rank.md`, prereg `prereg-facts-rank.md`, harness `c40dfd2`): on 20 facts at layer 16, k = 100, the gold first token is in the top 20 for 1 of 20 facts at scale 0, 14 of 20 at scale 8 (median NLL 6.97 to 3.86) and 2 of 20 at scale 32 (NLL 34.6). Its own label "grid too small" was wrong: scale 8 is already inside the useful range and 32 overshoots. Greedy recall stayed 0.000 in the real arm, so the miss is at top-1, with the peak somewhere between 8 and 32.

Setup (same as the rank run): the same 20 facts, implants file of the real arm, layer 16, k = 100 (`rows_cum[100]`), prompt = exact question, n = 1, `logits_topk` 20. Scales 0 (reference, `implants: 0`), 8, 12, 16, 20, 24. One shared gpu job.

Metric per scale: gold first token at rank 1 (count of 20) and in the top 5 (count of 20), from the top-20 list.

Bars, fixed now:
- RK-A (top-1 exists): at some scale in 8..24 the gold first token is rank 1 for 10 or more of the 20 facts. PASS, else MISSED.
- RK-B (top-5 reported, no bar): count per scale.
- Void if the scale-24 top-1 token equals the scale-0 top-1 token for all 20 facts (implants inert).

What a PASS means: first-token top-1 is reachable at one scale on 20 keyed facts with the exact prompt. It does not say the multi-token answer is recalled, nor paraphrases, nor k = 1000. The next step after a PASS is a recall run at that scale and needs its own prereg. A MISSED closes the facts line as null for this builder at layer 16. 20 facts, one template, one layer.
