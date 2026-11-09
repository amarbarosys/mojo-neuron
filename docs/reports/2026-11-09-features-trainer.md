# Trainer investigation: why the Qwythos L16 transcoder sat above a linear map (2026-11-09, lane features)

## VERDICT
The transcoder was undertrained and unstable, not overfit and not short of features, and fixing that moved K=64 held-out FVU from 0.850 to 0.658
(F=8192, 100k tokens). The 0.55 bar is not reached, and it is a data limit now: at K=256 the held-out curve turns up after step 4000, and halving
the tokens costs 0.15 FVU. A power-law fit through the 50k and 100k points puts the K=64 crossing of 0.55 between about 180k and 420k tokens
(three points, one seed each: UNVERIFIED beyond 100k). The linear ridge on the same split reaches 0.520.
-> `.work/tc-s*/` logs, sweeps listed below, prereg bar in `docs/prereg-b1.md` (0.30 stays missed), the 0.55 target set by the boss.

## What was wrong
1. **Reconstruction scale at init.** With unit decoder rows and K active features the first prediction had FVU about 8. Rescaling the encoder so the first
   batch's reconstruction has variance 0.5 x vtot (`--init-var 0.5`, a scalar of 0.28) took K=64, F=16384, lr 5e-4, 3000 steps from 0.850 to 0.697.
2. **A learning-rate ceiling near 3e-4 that no warmup removes.** Peak lr 1e-3 and 2e-3 diverged around step 750 with a 300-step warmup. Peak 5e-4 over 9000
   steps diverged at step 2000 (held-out FVU 10.4, ended 1.11) while the same lr over 3000 steps was stable, because the faster decay had cut the lr below
   about 3e-4 by step 1500. Peak 7e-4 with beta2 0.99 and cosine over 6000 steps diverged at step 1000 (5.15, aborted by the guard); over 1500 steps it survived,
   again because the cosine had dropped the lr. beta2 0.99 does not change the ceiling much; the lr in force around steps 500 to 1500 does.
3. **Not K, not dead features.** At equal steps K=256 (0.883) and K=1024 (0.902) trained slower than K=64 (0.850) from the same init; the dead share was 0 to 0.7 %
   in every run, 0.18 % in the final K=64 run. Smaller F converged faster (F=8192 0.794, F=4096 0.748 at 3000 steps).

## Final runs (Qwythos L16, 99,840 tokens captured, 90,112 train and 9,728 held-out, B=512, F=8192, init rescale 0.5, warmup 300, cosine to 10 %, beta2 0.99, lr 3e-4, 6000 steps, 33 ms per step)
Held-out FVU every 500 steps.

| K | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 | active per token | dead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | .939 | .816 | .762 | .729 | .705 | .686 | .676 | .668 | .663 | .660 | .659 | **.658** | 64.0 | 0.18 % |
| 256 | .863 | .738 | .697 | .671 | .652 | .638 | .633 | .633 | .634 | .636 | .640 | **.643** | 255.8 | 0 |

K=256 reaches 0.633 at step 3500 to 4000 and then rises: the data is the limit there. For reference, a K=256 run at lr 7e-4 (beta2 0.99) survived with a noisy curve
(1.38, .95, 1.68, 2.65, 1.11, .85 ..., ended .682) and a ridge warm start at F=8192 gave K64 .720, K256 .678, K1024 .604, K4096 .538 (3780 active per token, so dense: reference only).
Linear ridge, same split: 0.520 (train 0.447).

## Data scaling (K=64, same recipe, `--ntrain`, normalisation statistics from the full train split)
| training tokens | 25,088 | 50,176 | 90,112 (the full train split) |
|---|---:|---:|---:|
| held-out FVU after 6000 steps | 1.840 (rises after step 500: 122 epochs) | 0.809 | 0.658 |

Crossing of 0.55 from the 50k and 100k points: no floor, power law exponent 0.30, 182k tokens; floor 0.45, 253k; floor 0.50, 328k; floor 0.52 (the ridge), 418k.
Caveats: three points, one seed, 6000 steps at every size (more epochs on less data), the 25k point is an overfit point and is not used in the fit.

## Capture speed for the data lever
Engine capture today (one position per step): 97 to 108 tokens/s (prefill_s 4.74 to 5.26 for 511 rows). llama.cpp prompt processing on the same GGUF (Q4_K_M, ROCm):
2,213 tokens/s at 512 and 2,834 at 2,048 (`llama-bench`), a ceiling for a batched capture; the engine's own batched rate is lane A's to measure.
A 500k-token extra capture is about 78 min at 107 tokens/s, about 3 min at the llama.cpp rate; 16 KiB per token on disk (fields xy), 8 GB for 500k.

## GPU time and failures
`gpu-wait stats --days 1` (whole box): 1105 jobs, 951 ok, 125 failed, 29 cancelled, GPU busy 432 min, 83 min lost in failed or cancelled jobs.
Lane features today: 57 runs (35 ok, 20 failed, 2 cancelled) in 35 job kinds, 131 GPU wall-minutes; the 20 failures include the planted-defect mutant builds of `scripts/tc-check.sh`, which fail by design.
Two jobs of mine were cancelled on the boss's instruction (sweeps 6 and 7); sweep 6 was in fact nearly finished and its logs were used here.

## Mistakes
A test fixture cut a capture at half the record size and tcceil printed `inf`; tcceil printed `nan` on an empty held-out split (both fixed, 101eb3f, gate `scripts/tcceil-check.sh`).
A 9000-step run at a learning rate that only a 3000-step schedule had survived: the stable lr is a property of the lr trajectory, not of the peak value; run any long config first at 1500 steps
with the same schedule shape scaled to the long length, or watch the first 1500 steps of the long run and abort (the `--abort` guard now does).

## Next
1. A 400k to 500k-token Qwythos capture (batched path from lane A, or 78 min at the current rate as one gpu-wait job), retrain this recipe at 100k, 200k, 400k tokens, check the 0.55 crossing.
2. Splice KL with the improved transcoder (`tcsplice`), the bar of `docs/prereg-b1-splice.md` (0.25) stays.

## Update: the 400k capture (batched prefill, cos >= 0.9967 against per-position), K=64, same recipe
Capture `mojo-neuron/.work/capture/l16-400k`: 400,384 tokens in 782 requests of 512; no 16-gram of the old held-out appears in it (0 of 9,713, `python3 -I` check in the session), so the old held-out is clean.
Train = the first N tokens, held out = the last 5 % of the set (19,968 tokens), second score = the old 100k capture's held-out (9,728 tokens, per-position numerics), both with this run's normalisation,
F=8192, K=64, lr 3e-4, beta2 0.99, init rescale 0.5, warmup 300, cosine, 6000 steps (35 ms per step, so 8 epochs at 380k and 31 at 100k), one seed. 13 GPU-min for the job including the statistics pass.

| train tokens | held-out FVU, new set last 5 % | held-out FVU, old 100k held-out | dead |
|---:|---:|---:|---:|
| 100,352 | 0.778 | 0.730 | 0.4 % |
| 200,192 | 0.720 | 0.715 | 2.1 % |
| 380,416 | 0.695 | 0.681 | 1.2 % |

Curves (new/old held-out, every 500 steps) end 0.778/0.730, 0.720/0.715, 0.695/0.681 and are still moving at the end for 380k (0.698 to 0.695 over the last 500 steps, cosine-limited).
Fit `tools/scaling_fit.py` on the new-set points: FVU = 0.670 + 3.2e4 * N^-1.095, floor 0.670, so 0.55 is never reached on this curve. The old-held-out points improve faster at the end
(0.015 then 0.034), no power-law fit exists for them; they stay above 0.68.
**Retraction:** the earlier estimate "0.55 crossing at 180k to 420k tokens" (from the 50k and 100k points of the old capture) is refuted: at 380k the K=64 held-out FVU is 0.695 on the new set and 0.681 on the old one.
Reading: more data helped (0.778 to 0.695) with strongly diminishing returns; at K=64, F=8192 and 6000 steps the curve flattens near 0.67. The bar of 0.55 at K=64 is missed. Not tested, and the caveat of this fit: 380k got only 8 epochs and
a cosine that ends the run, so part of the flattening may be steps, not data (K=256 at 100k turned up after step 4000, which says data limits it there; K=64 at 380k has not shown that yet).
The old-capture-trained model scored 0.658 on the old held-out; the new-capture models score 0.73 to 0.68 on it, so the new set is a harder distribution or the batched numerics cost a little. UNVERIFIED which.

## Update 2: steps versus data, and the end of the 0.55 chase (380,416 train tokens, F=8192, same recipe, one seed)
The job was re-queued after a box reset (21:41) wiped the queue; nothing of the first attempt had finished (K=256 had reached step 500), so both runs below are complete runs of one job, 13.3 GPU-min (6000 steps 3.4 min, 18000 steps 9.9 min, 33 to 34 ms per step).

| run | held-out FVU, new set last 5 % | old 100k held-out | active per token | dead |
|---|---:|---:|---:|---:|
| K=64, 6000 steps (Update 1) | 0.695 | 0.681 | 64.0 | 1.2 % |
| K=64, 18000 steps | **0.641** | 0.633 | 63.9 | 0.06 % |
| K=256, 6000 steps | **0.624** | 0.611 | 255.7 | 0 |

K=64 curve at 18000 steps (new/old held-out, every 1000 up to 18000, selected points): 6000 0.718/0.699, 10000 0.671/0.659, 14000 0.647/0.637, 16000 0.642/0.634, 17000 0.642/0.633, 18000 0.641/0.633.
It flattens: 0.001 over the last 2000 steps. K=256 at 6000 steps is still moving (0.630, 0.626, 0.624 over the last three checkpoints) and neither run turns up, so at 380k tokens the data is not yet the limit that it was at 100k.
**Correction to Update 1:** the flattening of the 100k/200k/380k curve at 6000 steps was partly the step budget. Three times the steps took K=64 at 380k from 0.695 to 0.641 (0.054), as much as the whole 200k to 380k data gain at fixed steps (0.025) twice over.
So the fitted floor of 0.670 and the sentence "the K=64 curve flattens near 0.67" are withdrawn; the K=64 floor on this recipe at 380k is about 0.64, K=256 is lower still (0.62 and falling).
**Verdict on the 0.55 target:** missed at K=64 (0.641) and at K=256 (0.624); the chase stops here as the boss decided. A fixed bar was missed and is reported; the ridge control on this split is still unmeasured (0.520 on the old split), which the skip prereg (`docs/prereg-b1-skip.md`) takes as its step 0.
The linear ridge stays below every sparse run here by about 0.10 FVU on the old split, which is what motivates the skip arm.
