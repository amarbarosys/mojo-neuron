# prereg B1-skip: a transcoder with a linear skip path (written 2026-11-09, no run behind it; bars, controls and arms are fixed here)

## Why
On Qwythos L16 a closed-form ridge x -> y reaches held-out FVU 0.520 (old 100k capture), and no TopK run has: K=64 F=8192 reaches 0.658 (old 100k, 90k train), 0.695 (new 400k set, 380k train),
K=256 0.643 at 100k; the data-scaling fit puts the K=64 floor near 0.67. Most of what a layer-16 FFN does to this data is linear-predictable from its input, and a ReLU TopK code spends its
sparse features on that linear part. A skip transcoder (Dunefsky et al.) gives the linear part to a dense `W_skip x` and leaves the TopK features the remainder:
`y_hat = b_d + a(TopK(relu(x_c W_e + b_e))) W_d + x_c W_skip`, with `x_c = (x - xm) * sx` the trainer's normalised input.

## Arms (all on the 400k capture `mojo-neuron/.work/capture/l16-400k`, train = the first 380,416 tokens, held out = the last 5 % of the set, second score = the old 100k held-out)
| arm | what | trained |
|---|---|---|
| C0 control | closed-form ridge x -> y on the same train split (`tcceil` on the new capture, CPU, run before any skip arm; its number is the denominator of the bar) | none |
| C1 control | TopK only, K=64, F=8192, the sweep10 recipe | already run: 0.695 (new held-out), 0.681 (old held-out) at 6000 steps; the 18000-step run of today if it ends lower |
| A | skip (W_skip initialised from C0) + TopK path, trained jointly | W_skip, W_e, b_e, W_d, b_d |
| B | skip frozen at C0 + TopK path trained on the residual | W_e, b_e, W_d, b_d |
| C2 control | skip with a random init (same scale as C0), trained jointly with the TopK path | tells whether the ridge init matters |

Common settings (chosen from the investigation, not tuned on these arms): F=8192, K=64, B=512, peak lr 3e-4, beta2 0.99, warmup 300, cosine to 10 %, 6000 steps,
TopK path initialised to variance 0.05 x vtot (the skip carries the first fit), abort when held-out FVU > 5 after step 1500, one seed 1, held-out FVU every 500 steps.
W_skip has 16.8 M parameters, so the skip arms are not parameter-matched to C1; this is stated, not controlled.

## Bars (primary)
1. **The TopK path adds to the ridge**: arm A or arm B held-out FVU on the new set's last 5 % <= FVU(C0 on the same rows) - 0.02. Margin 0.02 is 4x the seed spread measured on the 0.5B rehearsal (0.697 vs 0.692);
   the Qwythos seed spread is not measured, so the winning arm gets a second seed (seed 2) and the claim needs both seeds under the bar.
2. **Not worse on the old held-out**: the same arm's FVU on the old 100k held-out <= FVU(C0 on those rows) - 0.01.
3. **Controls behave**: C2 is not better than A by more than 0.01 (else the ridge init was not the point and A is not a ridge story); A and B within 0.02 of each other is reported, not barred.

## Secondary (only if bars 1 and 2 hold)
4. Spliced KL through the engine splice op (`tcsplice`, same 60 positions as `docs/prereg-b1-splice.md`): KL(arm) / KL(zero splice) <= 0.25, identity KL <= 1e-3. One seed. Splice uses the whole `y_hat`, skip included.
5. Report: active features per token, dead share, FVU of the TopK path alone on the residual `y - x_c W_skip`, and the variance share the TopK path removes beyond C0.

## Predictions, committed before any run
P1: B reaches C0 - 0.03 to C0 - 0.05 on the new held-out (the remainder is partly nonlinear but small). P2: A within 0.01 of B. P3: C2 within 0.01 of A. P4: if P1 fails by landing between C0 - 0.02 and C0, the finding is "the sparse path adds under 0.02 on this data" and the transcoder
for interpretation stays TopK-only. A bar miss is reported as a miss.

## What this does not claim
The features of a skip transcoder explain the part of the FFN that is not linear in its input; any label or steering experiment on them (B2, B4) must use that definition, and none is run here.
The 0.55 target of the TopK-only chase is closed (floor about 0.67 on this recipe); this prereg is a different question, whether a sparse path adds to a linear one.

## Implementation, not yet written
`tc.mojo`: a `W_skip` buffer (d x d), its Adam moments, forward `yh += x_c W_skip`, gradient `gW_skip = x_c^T dy`; `tc_train`: `--skip ridge.f32 [--skip-frozen] [--skip-random]` (ridge matrix from `tcceil DUMP`; `W_skip = M^T / sx`, because `x_c = (x - xm) sx`);
the saved `tc.bin` folds the skip into one extra d x d block for `tcsplice`/`tc_splice`. Gate before any result run: `tc-check.sh` gains a skip case against the f64 reference (3 steps, 1e-4) and two mutants that must fail (skip gradient missing, skip sign flipped).
Cost estimate, UNMEASURED: about 40 ms per step (the skip adds three 512 x 4096 x 4096 products to the TopK step), 4 GPU-min per arm, arms A, B, C2 plus a second seed of the winner about 18 GPU-min, the splice job about 10, ridge C0 on the CPU.
