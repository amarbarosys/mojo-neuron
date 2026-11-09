# Prereg G5: implants as localized context distillation (written 2026-11-09 before any G5 run; the maintainer's pick, plan section G follow-up)

Idea. G4 compiles the prompt into implants at fixed positions (the last task token and the first 8 generated). G5 asks the teacher where it matters: teacher = the model with the skill prompt in context, student = the same model on the task alone. Distil only the spans where the two disagree about the next token,
and keep going until the student with implants matches the teacher on the training tasks or a row budget ends. Same skills, tasks, scorers, splits, layer, tau and engine ops as G4 (`docs/prereg-g4-tasks.md`); G0's bars for generalisation and damage still apply to every number.

## Compiler (no gradient step)

Per train task, the teacher's greedy continuation O (up to the skill's generation cap) is replayed teacher-forced twice: with the prompt (teacher) and without (student). The engine's steer lens gives, per position j, KL(teacher next-token distribution || student's) (`slens_ctx` base = teacher run, `slens_k` = student run).
Round 1: for every task take the positions with KL > 0.5 nats, at most 12 per task (the largest first). One implant row per chosen position: key = FFN input `x` of the student run at that position, value = `r_teacher - r_student` at the aligned positions (as G4), gate 1/(1 - tau), tau 0.9, layer as G4.
Round 2: run the student with the round-1 implants teacher-forced again; add rows (same rule, same cap, positions not already used) where KL is still > 0.5. Stop there (two rounds, a fixed budget; no further tuning). Gate scale chosen on dev from {1, 3, 9} as in G4. Random control: same rows with random keys and values of the same norms.

## What is measured (per skill, test split, greedy)

Task score `s_impl5` against `s_with`, `s_0`, `s_short` (G4's numbers are reused when the run's G4 receipts exist, else re-measured), retained ratio `(s_impl5 - s_0) / (s_with - s_0)`, rows per task and total, and the mean teacher-student KL over the test tasks' teacher continuations with no implants, with G5 implants, with random implants.
Unrelated damage and perplexity change as in G4.

## Bars (any miss reported as missed)

| bar | statement | pass |
|---|---|---|
| H-A | the skill has headroom (`s_with - s_0 >= 0.20` on test) | else no verdict |
| H-B | retained ratio with G5 implants | >= 0.60 |
| H-C | random-implant control retained ratio | <= 0.20 |
| H-D | unrelated accuracy drop and perplexity change | <= 1 pp and <= 1 % |
| H-E | mean teacher-student KL on test continuations falls with G5 implants, against none, and against random implants of equal size | by >= 30 % and by more than the random control does |
| H-F | economy against G4: G5 retained ratio minus G4's at its row count, or G5's rows for the same retained ratio | G5 is >= 0.10 higher at <= G4's rows, or reaches >= G4's retained ratio with <= 50 % of G4's rows |

A skill passes G5 if H-A to H-E hold; H-F says whether asking the teacher where to put the rows beats the fixed positions of G4, and may fail while G5 passes. G4 and G5 verdicts are reported side by side. Not claimed: anything beyond these three skills, this layer and this model.
