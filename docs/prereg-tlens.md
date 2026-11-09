# Prereg tuned lens: per-layer affine translator fitted with KL to the final logits (written before the result runs, 2026-11-09, lane agent)

Source: papers synthesis (logit lens is biased at middle layers; tuned lens fixes it). Model Qwythos-9B (32 layers), captures from A1.
Translator for layer l: `u = r + A r + b` (A h x h, b h, both start at 0 so step 0 is the logit lens), student logits `z = W_U (rmsnorm(u) * output_norm)`.
Teacher = same lens of the final residual (layer 31, field r) at the same position. Loss = KL(teacher || student), Adam, batch 64.
Data: `.work/cap/qwythos-lie` (layers 3,7,...,31, 45712 tokens, statement text from a local model: a narrow domain, said in the report). Positions >= 1 only.
Split by request id: id % 5 == 0 is held out (never seen in fitting or choosing), the rest is train. Train sample: 3000 positions; a fixed 10% of the train
sample is the validation part used for early stopping and for the learning rate. Held-out sample: 256 positions, fixed seed (xorshift 0x9E3779B97F4A7C15).
Training loss runs on a restricted vocabulary S = the 4096 tokens that occur most often among the teacher's top-32 over the train sample (teacher and student
renormalised over S). Held-out metric = full-vocabulary KL(teacher || student), mean over held-out positions, no restriction.
Learning rate chosen from {1e-4, 3e-4, 1e-3} by validation KL at layer 19 only, then used for every layer; 3 epochs; no other tuning.
Layers fitted: 7, 11, 15, 19, 23, 27. Baselines per layer: logit lens (same u = r) and `b-only` (A = 0, b fitted, same recipe: what a constant bias alone buys).

Bars (all must hold, a miss = FAIL, no post hoc change):
0. Teacher plumbing: top-1 of the teacher equals the next token of the text on >= 25% of held-out positions (else r at layer 31 is not the final residual and nothing below counts).
1. Identity: logit-lens KL at layer 31 is < 1e-4 (the same hidden state through the same code).
2. Tuned beats logit lens: held-out KL(tuned) <= 0.8 x KL(logit lens) at >= 4 of the 6 layers, and top-1 agreement with the teacher not lower than the logit lens at >= 5 of 6.
3. Affine part does work: KL(tuned) <= 0.9 x KL(b-only) at >= 4 of the 6 layers (else the gain is a unigram bias, say so).
4. Gradient oracle (CPU gate, tiny dims): analytic gradient of A, b and the RMSNorm path equals central finite differences, relative error < 1e-3.
5. Mutant: fitting against teacher distributions of shuffled positions must FAIL bar 2 in the gate (the check can fail).
6. One normal rlog line per layer with train, validation and held-out KL; a non-finite loss stops the fit and logs an anomaly.
Report only (no bar): train-minus-held-out gap, KL per layer for all three lenses, which layers fail.

Receipts: `.work/tlens/<run>/{<l>.bin, result.json}`, `tests/tlens.sh`.
