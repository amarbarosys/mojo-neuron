# Benchmark against gne (gguf-neuron-explorer)

Our resident steer lens runs the same workload 4.3 times faster than gne's `lens steer`, and our rollout sweep
1.4 times faster than gne's `sweep`. All numbers below are from one machine, one afternoon (2026-11-09), and the
commands that produced them are listed so anyone can rerun them.

## Setup

- Machine: Ryzen 7 7800X3D, Radeon RX 7900 XTX (gfx1100), 64 GB RAM, ROCm. Every GPU job through `gpu-wait`.
- Model: Qwen2.5-0.5B-Instruct, bf16 GGUF (`~/Models/qwen2.5-0.5b-instruct-import/qwen2.5-0.5b-instruct-bf16.gguf`), layer 12,
  both tools on the same file.
- gne 0.1.0 (MIT, `~/Projects/imports/gguf-neuron-explorer-main`) on `llama-cpp-python 0.3.36` built from source with
  `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`; all 24 layers on the GPU (`GPU 24 / CPU 0` in `gne info`).
- Ours: `mojo-baro` spark engine (branch `ablate`) on a q8 pack of the same weights, one resident process per run,
  `scripts/build-sweep.sh` binaries. Process start (pack load, GPU init) is 1.6 s and is included in every number.
- `rho` (the residual norm that `rel` doses scale by) differs a little: gne measures it over a corpus
  (`gne stats resid`), ours from one prompt (`tools/rho.py`). Our median KL at +0.5 is 0.342, gne's 0.388.

## Steer lens: the next-token change when every neuron of a layer is pushed

Workload: all 4864 neurons of layer 12 plus 64 random-direction controls, dose +0.5 and -0.5 (relative), gne's four
default contexts (three plain texts and one chat turn of 36 tokens), top 12 tokens.

| tool | run 1 | run 2 | run 3 | run 4 | per unit (both signs) |
|---|---|---|---|---|---|
| gne `lens steer` | 119.4 s | 128.9 s | | | 24 to 26 ms |
| ours `sweep mode=slens` | 28.8 s | 29.0 s | 29.0 s | | 5.9 ms |

```
gne -m MODEL --root ROOT lens steer --layer 12 --dose 0.5 --random-controls 64 --yes     # under gpu-wait
sweep MODEL OUT mode=slens layer=12 neurons=all doses=0.5,-0.5 prompts=gne-ctx.txt ctrl=64 k=12 chat=0 rho=rho.f32
```

How the speed came about: gne evaluates the four contexts as one batch per dose sign. Our first version sent one
request per context and was 3 times slower than gne on this model (12 ms per request, 4 contexts, one sign). The
fix was to stop treating a neuron as a request: the steered residual after layer L is the cached base residual plus a
vector, so layers L+1 to the end run on a batch of up to 28 neurons x 36 tokens as plain GEMMs (new kernels:
micro-sequence attention and rope, no KV cache). Weights are dequantised once per batch instead of once per neuron.

Agreement with gne on the same neurons (every neuron, dose +0.5): KL correlation 0.987 (+) and 0.990 (-), top-12 up-token
overlap 0.86. Against llama.cpp's own full-vocabulary answer on three sampled neurons: top-8 overlap 8/8, 6/8, 7/8,
KL within 3 % (`scripts/slens-check.sh`). The differences come from q8-pack versus bf16 arithmetic and the rho above.

## Rollouts: generated text under a pushed neuron

Workload: gne's `quick` design (who +0.8, orange +0.8, free +3.0 relative), neurons 0 to 31 plus 8 random controls,
32 tokens each, chat template on: 120 rollouts.

| tool | run 1 | run 2 | run 3 |
|---|---|---|---|
| gne `sweep --design quick` | 18.8 s | 15.3 s | 15.3 s |
| ours `sweep` (two batches, two process starts) | 11.1 s | 11.1 s | 12.2 s |

gne's first run includes the HIP kernel warm-up. Ours is 1.4 times faster including both process starts;
one batch would save another 1.6 s. A rollout is bound by token-by-token decode (about 1.6 ms per token on this model,
600 tokens/s single stream); gne batches several sequences per decode step, which is why the gap is small here.

## Decode cost of steering (BARO_STEER)

Qwen2.5-0.5B, the worst case because its layers are tiny; 6 interleaved runs of 200 tokens each way:

| | tokens/s (median) |
|---|---|
| no steering | 606.5 |
| BARO_STEER set | 602.0 |

0.7 % decode cost. The vector add is fused into the next RMSNorm launch, so there is no extra launch and no host sync.

## What these numbers do not show

- One small model. Bigger models are not measured yet (no 7B or 27B run of either tool). gne also batches its
  contexts, so the ratio there is open.
- gne steers all positions of its contexts through llama.cpp's control vector; so do we, and the gates compare against
  the same llama.cpp control vector semantics.
- Our steer lens contexts here are gne's own, 4 to 36 tokens. Very long contexts (96+ tokens) are refused.
