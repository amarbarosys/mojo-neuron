# Capture format v1 (engine op `BARO_CAPTURE`, lane causal owns this file)

Per-token activations of the baro engine (engine.mojo, dense qwen35 such as Qwythos-9B), written for transcoder
training (lane features), the live heat strip (lane app) and attribution graphs (A4). Readers: `numpy.fromfile`,
`mmap`, Mojo `read_bytes`. A change that breaks a reader bumps `v`.

## Layout

```
~/Projects/mojo/mojo-neuron/.work/cap/<model>/      <model> = GGUF stem, lower case (shared through the MAIN checkout)
  meta.json     written by the engine at start
  L<l>.f16      one file per captured layer l (0-based), records appended per token
  tokens.i32    the token fed to the model at that position, i32 LE, one per record
  req.i32       request id of that record, i32 LE (a new request restarts at position 0)
```

Record `i` of every file is the same token: `L<l>.f16` holds `nf * h` f16 LE values per record
(`nf` = length of `fields`), `tokens.i32` and `req.i32` hold one i32. Number of tokens = size of `tokens.i32` / 4.

## meta.json

```json
{"model":"<name or pack path>","dtype":"f16","h":4096,"n_layers":32,"fields":"xyr","layers":[16]}
```

`fields` lists the rows of a record in order; letters from `x y r`:

| letter | row | where it is taken |
|---|---|---|
| `x` | FFN input | output of the FFN RMSNorm (bf16 in the engine, widened to f32 then f16) |
| `y` | FFN output | down-projection result before the residual add (f32 -> f16) |
| `r` | residual | the residual stream after the layer (after steering if `BARO_STEER` is on), the position `BARO_DUMP` slot `2l+1` and `BARO_DUMP_LAYERS` read |

`r(l) - y(l)` is the residual after attention/SSM of layer l. Values beyond f16 range saturate to +-65504; the
engine prints `capture: pos P saturated N values` for every token that saturates (none seen on the gate prompt).

## Engine knobs

`BARO_CAPTURE=DIR` (turns capture on; DIR must exist; files are appended, so start from an empty DIR),
`BARO_CAPTURE_LAYERS=16,20`, `BARO_CAPTURE_FIELDS=xyr` (default), `BARO_CAPTURE_MODEL=name` (meta only).
Capture runs the layer loop one position at a time: batched prefill, speculation, the megakernel and the
checkpoint reuse are off for the whole process (the engine refuses a step that advances more than one position).
Every prompt position and every generated position except the last sampled token is recorded. Dense models only
(`IS_MOE` raises; MoE router capture is item A5).

## Streaming

The files are opened, appended and closed once per token, so a reader that tails them (`size / (nf*h*2)` records)
sees a token as soon as its step finished. The app (C1) polls `tokens.i32` size.

## Size

`nf * h * 2` bytes per token per layer: Qwythos-9B, `xyr`, one layer: 24 KiB per token, 10 M tokens = 245 GB.
Use `BARO_CAPTURE_FIELDS=xy` (16 KiB per token, what a transcoder reads) and fewer tokens when disk matters.

## Batched prefill (engine neuron2b a1a608a)

With `BARO_STEER_SERVE` unset the capture also runs through the batched prefill path (chunks of up to 1024 rows): same files and record layout, about 24 times faster (2563 against 106 tokens/s on 2048 tokens). y is the residual after the FFN add minus the residual before it (computed in f32 on the host side
of the chunk), the last prompt token still goes through the one-position path. Against the one-position capture of the same text: median relative L2 x 0.0047, y 0.0079, r 0.0034; residual cosine at least 0.99665 (5 of 2048 positions under 0.999). Numerics differ slightly (WMMA GEMMs against gemv), so a set
captured batched is not byte-equal to one captured per position; do not mix them inside one training set without saying so.
