# prereg B1: transcoder trainer (written before any result run, 2026-11-09)

Transcoder: MLP input `x` (post `ffn_norm`, d_in) -> `z = x W_enc + b_enc` -> TopK (K largest, kept if > 0) -> `y_hat = a W_dec + b_dec`
predicts the MLP output `y = W_down act` (d_out). Loss = SSE / SS_total (the FVU itself), Adam, W_dec rows renormalised to unit norm each step.
Tie rule (GPU and CPU reference alike): active = `z >= t_K` and `z > 0`, with `t_K` the K-th largest value of the row (ties may give more than K).

## Gates
1. **Step parity**: one training step on random tiny data (T=64, d=24, F=96, K=8): GPU loss, `a`, and every updated parameter vs
   the f64 CPU reference, max |err| <= 1e-4 * max|ref| per tensor. Mutants that must FAIL it: K+1, W_dec renorm off, sign of db_dec.
2. **Determinism**: two same-seed runs write byte-identical weights.
3. **Rehearsal (Qwen2.5-0.5B layer 12, F=4096, K=32, ~100k tokens, 90/10 split by window)**: held-out FVU <= 0.60; the trained
   transcoder spliced into the CPU forward changes held-out NLL by <= 50 % of what zero-ablating that MLP does.
   Pipeline rehearsal only; no claim about Qwythos.
4. **Qwythos layer 16 (F=16384, K=64, A1 capture, ~10M tokens)**: held-out FVU <= 0.30 and spliced NLL delta <= 25 % of the
   zero-ablation delta. Not reached = reported as not reached, bars are not moved.

## Reported with each run
tokens, params, tokens/param, steps, step time, peak VRAM, FVU per 10 % of training, fraction of dead features (never active on
held-out), mean active count per row.
