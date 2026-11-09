# mojo-neuron plan

Find, rank and steer single MLP neurons of GGUF models on mojo-baro, and export them as llama.cpp
control vectors. About 600 LOC of Mojo, because GGUF reading, dequantization and the GPU forward
already exist in our libraries. Not a trainer. MoE experts are out of scope for now.

Ideas come from gguf-neuron-explorer (`gne`, MIT) and neuron-explorer (MIT). Written clean-room in
Mojo; no code copied.

## Reuse (surveyed 2026-11-04)

| need | reuse | state |
|---|---|---|
| GGUF read, dequant of all quant types | mojo-gguf `gguf.model.Gguf`, `gguf.dequant.deq_rows` | done |
| thread pool | mojo-gguf `gguf.par.parallelize` | done |
| GPU forward, 6 archs | mojo-baro `serve/spark.mojo` (llama, qwen2, granite), `serve/engine.mojo` (qwen35) | done |
| per-layer residual hook | baro branch `ablate`: `BARO_DUMP_LAYERS`, `BARO_ABLATE` (host-side, spark only) | partial |
| logits dump | baro `BARO_DUMP_LOGITS` | done |
| TPE, KL | mojo-ablate `src/tpe.mojo`, `src/kl.mojo` | done |
| UI | gne `gne/ui/static` (vanilla JS) served as static files | later |

## Steps

Each step lands as its own commit, with its gate written first.

| # | item | LOC | performance target | oracle |
|---|---|---|---|---|
| 1 | `lens`: logit lens for every neuron of a layer, CPU | S | Qwen2.5-0.5B layer in < 5 s | f64 brute force on sampled neurons, plus a mutant |
| 2 | baro `BARO_STEER`: add one f32 vector per layer into X after the FFN, GPU kernel, no host sync | S | decode cost < 1 % | zero vector byte-identical; greedy equals llama.cpp `--control-vector` |
| 3 | `cvec`: write `.cvec.gguf` (`direction.L`, arch `controlvector`) from a recipe `L:N:dose,...` | S | n/a | llama-cli loads it; greedy equals step 2 |
| 4 | `sweep`: one resident baro process runs neurons x doses x prompts | M | engine startup paid once | same tokens as one-shot runs |
| 5 | `rank`: word-cluster mass against random-direction controls | S | < 1 s per layer, host-side | hand-built rollout golden |
| 6 | serve the gne UI and an MCP server over steps 1 to 5 | M | n/a | UI looked at in the browser |

Step 2 is measured first. If the hook costs more than 1 % of decode, it moves into the FFN down
gemv epilogue.

## Lens math

For neuron n of layer L: g = unit(W_down[:, n]) * output_norm_weight, logits = W_U g. Keep the top
and bottom k tokens. One pass over W_U per layer: each vocab row is dotted with every neuron while
the neuron block stays in L3.

## Gates

`scripts/check.sh` runs every test capped with
`systemd-run --user --scope -p MemoryMax=4G -- timeout 300`. GPU work goes through `gpu-wait run`.
