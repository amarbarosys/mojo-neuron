# Engine lane: parity rows (plan steps 2 to 4 plus steer lens, reverse lookup, ablation)

Rows 5 to 10 of the shared docs/parity.md (lane app owns that file; fold these in at merge). Gate = script in `scripts/`,
all captured and run on 2026-11-09.

| # | feature | gate | result |
|---|---|---|---|
| 5 | steer lens, next-token dlogprob and KL at +-dose, random controls | `slens-check.sh` | PASS: top-8 vs llama.cpp 8/8, 6/8, 7/8; KL within 3 %; neighbour mutant rejected |
| 6 | reverse lookup, persona contrast, cosine rank of all layers | `reverse-check.sh` | PASS: top and bottom 10 of 116736 equal an f64 oracle; B - A mutant rejected. Not done: gne's `standardize` (per-dimension residual std), `pca1`, `--verify` rollouts |
| 7 | sweep neurons x doses x prompts, resident | `sweep-check.sh` | PASS: 10 records equal independent one-shot runs, neighbour neurons differ |
| 8 | BARO_STEER add vector per layer on GPU | `steer-check.sh` | PASS on spark archs qwen2, llama, granite (matrix below); zero vector byte-identical; decode cost 0.7 % |
| 9 | cvec export, llama.cpp loads it | `steer-check.sh` | PASS: `.cvec.gguf` equals the f32 vector layer by layer (python gguf reader) and steers like llama.cpp |
| 10 | ablation: neuron gain `L:N:k` | `gain-check.sh` | PASS vs HF fp32 hook (rel err 0.057, mutant 1.16). Directional ablation: existing host-side `BARO_ABLATE` (mojo-ablate gate), not re-gated here |

## Steering matrix (BARO_STEER, serve steer_bank, batched steer lens)

| model | arch / engine | steer-check | slens | note |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct bf16 | qwen2, spark | PASS | PASS | all gates |
| Llama-3.2-1B-Instruct Q4_K_M | llama, spark | PASS | PASS | |
| granite-4.2-3b bf16 | granite, spark | PASS | PASS | |
| Qwythos-9B Q4_0 pack | qwen35, engine.mojo | PASS (`engine-steer-check.sh`) | UNSUPPORTED(spark op only) | one-shot `BARO_STEER`; megakernel off while steering |
| OrcaSAQ-2-Cyber-27B (--q4 pack) | qwen35, engine.mojo | PARTIAL: 1 of 2 recipes passes the noise-aware rule | UNSUPPORTED | pack's base logits 22 % from llama.cpp, see report |
| RegesCore 35B, Qwen3.8-Flash-Next | MoE | UNSUPPORTED(MoE experts out of scope) | | |
| Spark-X2.5-4B, MiniCPM5-2B, vibethinker, Qwen2.5-7B | spark | UNVERIFIED(no gate run; same code path as the PASS rows) | | |
