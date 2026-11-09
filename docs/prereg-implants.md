# Prereg G0: neuron implants (committed before G1 to G4 produce any result, 2026-11-09)

Plan section G of `docs/plans/2026-11-09-neuron-v2.md`. The verdict is printed against the bars below and claims nothing beyond them. A missed bar is reported
as missed. No bar is re-cut after a result; a later note may annotate a verdict and must say it was written later.

## What an implant is

Base weights stay frozen. An implant set for layer l holds k rows `(key_i, gate_i, value_i)`. After the FFN of layer l the engine adds
`sum_i gate_i * relu(cos(x, key_i) - tau) * value_i` to the residual. `x` is that token's FFN input (the post-norm row that A1 captures as field `x`), `cos` is
cosine similarity, `tau` is one threshold per set. `implants.bin` follows `docs/format-implants.md` (G1 writes it). With k = 0 the engine is unchanged.
Implants switch on and off per request.

The builder (G2) takes the key from the A1 capture of the FFN input on a mistake. It takes the value from the unembedding direction (the lens) toward the right
token minus the wrong token. It fits the gate in closed form so the right token wins by a margin on that example. No gradient step anywhere.

## Fixed setup

- Model and engine: Qwythos-9B Q4_0 pack on engine.mojo (baro neuron2 build), greedy decoding, at most 8 generated tokens, answer parsed from the text.
- Layer: chosen once on the dev split among 12, 16, 20 and 24 (best held-out-mistakes gain at k = 100), then fixed for every test number.
- Splits per family, seeded, committed with the data file: `train` (mistakes the builder may see), `dev` (choose layer and tau, never in a bar), `test` (the only
  split a bar is computed on). A mistake is a problem whose greedy answer from the unmodified model is wrong. Train, dev and test mistakes are disjoint problems.
- Families. T1: two-digit addition with at least one carry (pool of 20000 problems, seed 1). T2: country to capital, every country in `tests/data/capitals.tsv`,
  4 prompt templates. T3: one code idiom family, 40 templated Python one-liner requests ("an expression that returns ...") scored by running the model's
  expression on 5 unit inputs (pool of 4000 instances, seed 3). If a family yields fewer than 2k mistakes, k is capped at the train-mistake count and the cap is
  reported. A family with fewer than 30 test mistakes is reported as UNDERPOWERED and gets no verdict on G-A.
- Unrelated set: 200 fixed items (100 trivia questions in `tests/data/unrelated.tsv`, 100 items from the other two families that the unmodified model answers
  right), plus perplexity on `tests/data/ppl.txt` (fixed text, 2000 tokens).
- k sweep: 1, 10, 100 and 1000 implants, one per train mistake, capped as above. Control: for each k an equal-size random implant set (random unit keys, random unit
  values scaled to the same norms, the same tau and gate distribution). Every number is reported for implants and for the control.
- Statistics: paired exact McNemar test on the same test items. A gain counts when it meets the bar and p < 0.05 (one-sided, implants against unmodified).

## Verdict 1: generalisation, per family

| bar | statement | pass |
|---|---|---|
| G-A | held-out-mistakes accuracy, implants built from train mistakes only, against unmodified | gain >= +10 pp at some k, McNemar p < 0.05 |
| G-B | the same gain minus the random-implant control's gain at that k | >= +10 pp, and the control's own gain within +-3 pp of 0 |
| G-C | unrelated damage at that k | unrelated accuracy drop <= 1 pp and perplexity increase <= 1 % |
| G-D | train mistakes fixed at k = train size (the builder works) | >= 90 % of train mistakes answered right with implants on |

A family passes only if G-A, G-B and G-C hold at the same k. G-D is a precondition: if it fails, the implants did not fix what they were built from, and G-A to G-C
are reported as "builder failed".

## Verdict 2: memorisation as cheap knowledge injection

Facts: 1000 invented facts with new proper names and no prior knowledge ("The capital of Zorvia is Plinth."), in `tests/data/facts.tsv`. Each has the exact question
and 3 templated paraphrases. A fact the unmodified model already answers right (more than 5 % of them) is dropped. Implants are built from the exact question
only, with k equal to the number of facts: 1, 10, 100, 1000.

| bar | statement | pass |
|---|---|---|
| M-A capacity | facts recalled at k = 100, exact prompt and mean over the 3 paraphrases | exact >= 90 %, paraphrase >= 50 %. The same numbers at k = 1000 are reported, no bar |
| M-B interference | unrelated accuracy and perplexity at k = 100 with all 100 facts loaded | accuracy drop <= 1 pp, perplexity increase <= 1 %, and the random control of equal size moves neither more than the implants do |
| M-C cost | wall seconds per fact (builder including capture, k = 100) against one LoRA step of the Qwythos trainer | <= 5 s per fact. The LoRA step time comes from baro's train log (path cited in the report) or is marked UNMEASURED, never guessed |

## G4: prompt to implant compiler (context distillation), per skill

Three skills, each with a deterministic scorer:

1. JSON output format. 200 extraction prompts. Score: the output parses, has exactly the keys asked, and matches the gold values.
2. A style rule: never use the character U+2014, and end every answer with the word "Done.". 200 prompts, scored by a linter.
3. Mojo 1.2 syntax rules (`def` not `fn`, `comptime` not `alias`, `from std.` imports). 100 small function requests. Score: the output compiles with the repo's
   Mojo 1.2 (`mojo build` of a file with the answer inside a main).

Per skill the conditions are: `s_with` (skill prompt in the context), `s_0` (no prompt, no implants), `s_impl` (no prompt, compiled implants), `s_ctrl` (no prompt,
random implants of equal size) and `s_vec` (no prompt, one steering vector from the same captures). The compiler takes A1 captures of the same tasks with and
without the prompt and the per-layer residual delta at the positions that matter (the last prompt token and the first 8 generated), and compiles them into implants
on the layers chosen in verdict 1. Each skill has 50 train tasks; the rest is dev and test as above.

| bar | statement | pass |
|---|---|---|
| D-A headroom | `s_with - s_0` on test | >= 0.20, else the skill is "no headroom" and gets no verdict |
| D-B retained | `(s_impl - s_0) / (s_with - s_0)` on test | >= 0.60 |
| D-C control | the same ratio for `s_ctrl` | <= 0.20 |
| D-D damage | unrelated accuracy drop and perplexity increase with the implants on | <= 1 pp and <= 1 % |
| reported | tokens and prefill ms saved per request (skill prompt length, measured engine prefill) and the retained ratio of `s_vec` | no bar |

## Reporting rules

One table per verdict with every bar, its value and PASS or MISSED, failures first. Every run calls rlog: a normal line with the headline numbers, an anomaly line for
NaN or Inf, a missed bar, or an implant that changes an unrelated item's answer. A run whose own check fails (k = 0 not bit-equal to no implants) is void, not a
verdict. Mistakes in the process go to `docs/rules.md`. Verdicts go to the maintainer unchanged.

## Amendments written before any G3 result run (2026-11-09, after the harness was written, no result seen)

1. T1 mining pool is every two-digit addition with at least one carry, 3645 distinct problems (not 20000: that many distinct problems do not exist; the 6000 first planned could not be generated either). Splits (train 50 %, dev 20 %, test 30 % of the mistakes) and bars are unchanged. If this
   leaves fewer than 2k mistakes, k is capped as the prereg already says.
2. Perplexity is measured as the exp of the mean next-token NLL change over 400 fixed 24-token windows of `tests/data/ppl.txt` (implants on against implants off in the same process, engine `slens_tgt` dNLL),
   minus 1; the bar (<= 1 %) is unchanged. The engine has no teacher-forced scorer over a full text.
3. The unrelated set is the 10 trivia questions of the harness plus non-carry two-digit additions the unmodified model answers right, 100 items, mined in the same pass (not 100 hand-written trivia).
4. Families T2 (capitals) and T3 (code idiom) and the three G4 skills have no harness yet; they are reported NOT RUN, never PASS.
