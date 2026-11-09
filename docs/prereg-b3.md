# prereg B3: cross-model map (written before any result run, 2026-11-09)

Map = ridge regression (`ridge` of src/rlens.mojo, lambda 1e-3 of the mean Gram diagonal) from the residual stream after block LA of model A
to the residual after block LB of model B, on the same tokens (same windows), fitted on the first 90 % of the windows.
Score on the held-out 10 %: R2 = 1 - SSE / SS_tot (SS_tot around the held-out mean of B's residual). Control: the same fit with B's rows permuted
across positions (seeded), scored on the same held-out rows. "Same neuron" score: write direction of an A neuron (unit W_down column of A,
layer LA) pushed through the map, unit-normalised, cosine against every unit W_down column of B layer LB; top-1 cosine per neuron.

## Bars
1. Qwen2.5-0.5B layer 12 (24 blocks) -> vibethinker-1.5B layer 14 (28 blocks), one shared tokenizer, 800 windows of 128 tokens:
   held-out R2 >= 0.50; shuffled control R2 <= 0.05.
2. Neuron match, 512 sampled A neurons: mean top-1 cosine through the real map >= 1.5 x the mean through the shuffled-control map.
3. Qwythos-9B (A1 capture, `r` field) vs the Qwen pair: only after bars 1 and 2 are reported; needs a token alignment (tokenizers differ),
   bars written here before that run: held-out R2 >= 0.30, shuffled control R2 <= 0.05.
A missed bar is reported as missed; the bars do not move.
