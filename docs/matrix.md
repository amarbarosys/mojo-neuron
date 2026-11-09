# Model x feature matrix (lane app)

Cells: PASS, or UNSUPPORTED(reason). Lens = `lens` golden (f64 brute force of sampled neurons + neighbour mutant) and a `lens.bin` for the shared store.
Rows measured 2026-11-09 with `scripts/lens-matrix.sh` (seconds = golden + bin for one layer, CPU).

| model | arch | lens layer | lens golden | lens.bin | seconds | fires (CPU forward) |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct q4_K_M | qwen2 | 12 | PASS | PASS | 11 | PASS (bf16 file vs llama.cpp) |
| vibethinker-1.5b q4_K_M | qwen2 | 14 | PASS | PASS | 26 | UNVERIFIED (same code path as qwen2) |
| llama-3.2-1b-instruct q4_K_M | llama | 8 | PASS | PASS | 30 | PASS |
| minicpm5-2b q4_K_M | llama | 20 | PASS | PASS | 26 | UNVERIFIED (minicpm scales not modelled) |
| granite-4.2-3b bf16 | granite | 20 | PASS | PASS | 36 | PASS |
| spark-x2.5-4b q4_K_M | spark2_5 | 18 | PASS | PASS | 67 | UNSUPPORTED(arch spark2_5 has no CPU forward yet) |
| ornith-1.5-9b q4_K_M | qwen35 | 16 | PASS | PASS | 193 | UNSUPPORTED(hybrid linear attention, no CPU forward) |
| qwen3.8-27b obliterated q4_K_M | qwen35 | 32 | skipped (27B check needs 5 GB of W_U in RAM) | PASS | 175 | UNSUPPORTED(hybrid linear attention) |
| lily-cybersecurity-7b IQ1_S | llama | 16 | skipped | PASS | 15 | UNVERIFIED |
| TriLM-1.5B TQ2_0 | llama | 12 | UNSUPPORTED(ggml type 35 TQ2_0 has no dequantizer in mojo-gguf) | UNSUPPORTED(same) | 0 | UNSUPPORTED(same) |
| RegesCore 35B, qwen3.8-flash-next (MoE) | qwen35moe, qwen4exp | n/a | UNSUPPORTED(MoE experts out of scope) | UNSUPPORTED | n/a | UNSUPPORTED |
| bge-m3 | bert | n/a | UNSUPPORTED(encoder, no output head) | UNSUPPORTED | n/a | UNSUPPORTED |
