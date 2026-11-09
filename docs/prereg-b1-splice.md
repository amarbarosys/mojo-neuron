# prereg B1 splice: transcoder spliced into Qwythos-9B layer 16 through the engine (written before any splice run, 2026-11-09)

Replaces the "spliced NLL delta" half of prereg-b1 bar 4 on Qwythos, which could not be run without an engine op. Lane A's splice op
(`splice_layer`, `splice FILE` f32 [n][4096] or `splice_zero`, baro neuron2b 13d6ea8) and its measured effect scale: mean NLL moves by only 0.0015 under a
zero splice, so the metric is KL of the next-token distribution against the unspliced run (the `kl` field of the steer-lens line); floors on lane A's
40 positions: identity splice 1.1e-5, zero splice 3.4e-2, shifted rows 4.1e-2.

## Setup
4 held-out requests of the Qwythos capture (the last 10 % of records, requests 17 to 20), positions 32, 64, ..., 480 of each (15 per request, 60 in all),
prompt = the request's tokens before the position, target = the token at the position. Three arms per position: the transcoder output (`tc_train` weights,
run on the capture's x rows), the true FFN output from the capture (identity, the floor), `splice_zero`.

## Bar
KL(transcoder) / KL(zero) <= 0.25 (the earlier "25 % of the zero-ablation delta", restated in KL), and identity KL <= 1e-3 (the op acts as designed).
The transcoder's held-out FVU is printed beside it; a miss is reported as a miss.
