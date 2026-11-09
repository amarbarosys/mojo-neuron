# Prereg E1: baro-serve `steer` and `capture` passthrough (written before the result run, 2026-11-09)

Setup: baro-serve (neuron2 build) in front of engine.mojo (Qwythos-9B Q4_0 pack), started with BARO_STEER_SERVE=1, BARO_CAPTURE=DIR,
BARO_CAPTURE_LAYERS=16. Prompt = the 13-token pack prompt, greedy, `/v1/completions`, `logits_topk` 20.
Bars (any miss = FAIL):
1. `steer: [{layer:16, neuron:N, scale:0}]` over HTTP: the 20 raw logits of the first generated token equal the engine's own steered run
   (direct serve-mode `gain 16:N:0`, BARO_DUMP_LOGITS_DIR row) at the same ids within 1e-3.
2. The same request without `steer` differs from the steered one by more than 1e-3 in at least one of the 20 logits (the field acts; mutant).
3. `capture:"t1"`: the response carries `capture.path`; DIR/t1/L16.f16 has (prompt + generated - 1) records of 3*4096 f16; its first 13 records
   are byte-equal to the A1 capture of the same prompt from a direct engine run.
4. A request without `steer` and `capture` sends the engine a line without the new keys (cargo unit tests `steer_tests`).
Not claimed: feature steering (needs B's transcoder bank: the server refuses `feature` items), streaming responses carrying `capture.path`
(only the non-stream response does), `steer.bank`/`steer.file` live (unit-tested mapping only; engine side is the existing steer_bank/steer_file).
