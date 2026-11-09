# Prereg tuned lens b: supersedes `docs/prereg-tlens.md` before any run (written 2026-11-09, lane agent)

`prereg-tlens.md` stays committed and never ran. Reason for b: while writing the fit I costed it. (1) The vocabulary S taken from the teacher's top-32 needs a
full W_U pass over 3000+ train positions (3e12 multiply-adds on CPU) before training starts; (2) Adam at lr 1e-4 on a 4096 x 4096 matrix moves `A r` by tens of
residual rms units per step, so the old learning-rate grid could not have fitted anything. Bars 0 to 6 of the old doc are unchanged and repeated below.
What changes:
- S = the 4096 most frequent next tokens (true text tokens) of the train requests. Teacher and student are renormalised over S in training. The coverage of the
  teacher's mass by S on the held-out positions is reported (no bar). Held-out metric stays the full-vocabulary KL(teacher || student).
- Input scale-free: per position `s = rms(r)`, `x = r / s`, `u = r + s * (A x + b)`; A and b start at 0 (step 0 is the logit lens).
- Optimiser Adam (0.9, 0.999). `b`: lr 0.01 (units of s). `A`: lr from {3e-6, 1e-5, 3e-5}, chosen by validation KL at layer 19 only, then used for every layer.
- Sample: 8000 train positions plus 800 validation positions (both from requests with id % 5 != 0, positions >= 1, xorshift 0x9E3779B97F4A7C15 shuffle), 256 held-out
  positions (id % 5 == 0). Batch 64, 2 epochs; the kept translator is the best of epoch 0 (logit lens), 1, 2 by validation KL, so it is never worse than the logit lens on
  validation. Layers 7, 11, 15, 19, 23, 27.
- Baselines per layer as before: logit lens and `b-only` (A frozen at 0, same lr for b, same schedule).
Data, teacher definition, split rule and the six bars: as `prereg-tlens.md`, repeated:
0. Teacher top-1 equals the next text token on >= 25% of held-out positions that have a next token.
1. Logit-lens KL at layer 31 < 1e-4.
2. Held-out KL(tuned) <= 0.8 x KL(logit lens) at >= 4 of 6 layers, and top-1 agreement with the teacher not lower than the logit lens at >= 5 of 6.
3. KL(tuned) <= 0.9 x KL(b-only) at >= 4 of 6 layers.
4. Gradient oracle (tiny dims, CPU): analytic gradients of A, b equal central finite differences, relative error < 1e-3.
5. Mutant: fit against teacher distributions of shuffled positions must FAIL bar 2 in the gate.
6. One normal rlog line per layer with train, validation and held-out KL; a non-finite loss stops the fit and logs an anomaly.
Report only: train-minus-held-out gap, KL per layer for all three lenses, S coverage, which layers fail, wall time.
Receipts: `.work/tlens/<run>/{L<l>.bin, result.json, log.txt}`, `tests/tlens.sh`.
