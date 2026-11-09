# Prereg G4 tasks: the concrete skills, scorers and compiler settings behind the G0 bars (written 2026-11-09 before any G4 run)

G0 (`docs/prereg-implants.md`, 0604f51 + amendments) fixes the G4 bars (D-A headroom, D-B retained, D-C control, D-D damage). This file fixes what G0 left open, before anything is run; it changes no bar.
Fold-in from `docs/prereg-papers-fold.md` that applies here: the shortened-prompt baseline `s_short` (first sentence of the skill prompt only), reported next to the bars; implants count as useful for a skill only if `s_impl - s_short >= 0.10` as well.

## Skills (prompt text is prepended to the task; the task text is the same in every condition)

1. JSON format. Prompt: `Answer with a single JSON object with exactly the keys "name" and "age" and nothing else.` Task: `Text: <sentence>` then `Extract the person's name and age.` then `Answer:`; sentence from 3 templates over 40 names, ages 18 to 90, 30 cities.
   Score = 1 when the generation (cut at the first blank line) parses as JSON, is an object with exactly the keys name and age, name equals the gold name and age equals the gold integer. 200 tasks: 50 train, 50 dev, 100 test. Generation cap 40 tokens.
2. Style rule. Prompt: `Style rule: never use the character U+2014 (em dash) and always end the answer with the word Done.` Task: `Question: Describe <topic> in one sentence.` then `Answer:` over 50 topics x 4 phrasings. Score = 1 when the generation contains no U+2014 and its stripped text ends with `Done.` 200 tasks: 50/50/100. Cap 80 tokens.
3. Mojo 1.2 syntax. Prompt: `Mojo 1.2 rules: declare functions with def (not fn), use comptime (not alias), import from std. (for example from std.math import sqrt). Write only code.` Task: `Task: write a Mojo function named f that <description>.` then `Code:` over 20 descriptions x 5 parameter variants.
   Score = 1 when the code (markdown fences stripped, a `def main(): pass` appended when absent) contains `def f` and `mojo build` of it with the repo's Mojo 1.2 exits 0. 100 tasks: 50 train, 20 dev, 30 test. Cap 160 tokens.

## Conditions (greedy, same engine, same cap per skill)

`s_with` prompt + task; `s_0` task only; `s_short` first sentence of the prompt + task; `s_impl` task only with the compiled implants; `s_ctrl` task only with random implants of equal size (same norms, gates, tau); `s_vec` task only with one steering vector.
Headroom `s_with - s_0` is read on the test split; a skill below 0.20 is reported "no headroom".

## Compiler (G4 deliverable, in `tools/g4-harness.py`; no gradient step)

Layer = the layer chosen on dev for the T1 verdict (`select.json`). For every train task: `O` = the first 8 tokens of the `s_with` generation. Two teacher-forced runs with A1 capture at that layer (fields x and r): with = prompt + task + O, without = task + O.
Positions of interest = the last task token and the 8 tokens of O. For each position j: key = `x` of the WITHOUT run at j; value = `r_with - r_without` at the aligned positions (the residual push the prompt causes); gate = 1 / (1 - tau) so that a perfect key match reproduces the push; tau = 0.9.
One row per (task, position): 450 rows per skill. Real and control rows share one file (control rows after the real ones, random keys and values with the real rows' norms); row ranges per skill are selected per request (`implants_lo`, `implants_k`).
Gate scale is chosen on DEV from {1, 3, 9} (the scale that maximises dev score of `s_impl`); the steering-vector baseline: the mean push over all rows of the skill, scale chosen on dev from {0.5, 1, 2}, added at the same layer through `steer_file`.
Unrelated damage (D-D): the 100 unrelated items and 400 windowed-perplexity contexts of the T1 harness, implants on against off, at each skill's chosen scale.
Report: per skill one row with s_with, s_0, s_short, s_impl, s_ctrl, s_vec, retained ratios, D-A to D-D PASS or MISSED, tokens saved per request (prompt tokens) and measured engine prefill seconds with and without the prompt on 10 test tasks. A skill whose test split is under 30 tasks is flagged UNDERPOWERED.
