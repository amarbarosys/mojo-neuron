# mojo-neuron data format v1

Lane engine owns this file. It adopts the lane app proposal (`.work/briefs/format-proposal.md`) for
`lens.bin`, `sweep.jsonl`, `steer.bin` and the recipe string, and adds `model.json`, the steering
unit math and the cvec spec. A change that breaks a reader bumps the version. All numbers
little-endian, all text UTF-8. Token text is the display form after `unbyte`.

## Layout

```
.work/neurons/<model>/           <model> = GGUF file stem, lower case
  model.json                     once per model
  <layer>.lens.bin               logit lens of every neuron of layer L (lane app writes)
  <layer>.sweep.jsonl            rollouts, append only (lane engine writes, lane app ranks)
  <layer>.steer.bin              steer lens at one dose (lane engine writes)
  <layer>.dossier/<neuron>.json  optional neuron record
.work/cvec/<name>.cvec.gguf      exported control vectors
```

`.work/neurons/` is shared through the MAIN repo (`~/Projects/mojo/mojo-neuron/.work/neurons`).
A unit is `(model, layer, neuron)`, layers 0-based.

## model.json

```json
{"v":1,"model":"qwen2.5-0.5b-instruct-q4_k_m","gguf":"/abs/path.gguf","arch":"qwen2",
 "n_layer":24,"hidden":896,"ffn":4864,"vocab":151936,
 "rho":[0.0],"rho_src":"median residual norm after each layer over N prompts"}
```

`rho[L]` = median L2 norm of the residual stream after layer L (one f32 per layer). It is the
unit of a `rel` dose.

## lens.bin

`magic "MNLN" | u32 version=1 | u32 layer | u32 ffn | u32 hidden | u32 k`, then `ffn` records:
`top_id i32[k] | top_z f32[k] | bot_id i32[k] | bot_z f32[k]`. Top = most positive logit-lens
score, bot = most negative, best first. z = dot(W_U row, unit(W_down[:, n]) * output_norm_weight).

## sweep.jsonl

One JSON object per line:

```json
{"model":"m","layer":14,"neuron":8949,"prompt":"Who are you?","dose":0.8,"text":"..","tokens":[1,2]}
```

- `neuron` is an int, `"ctrl0"`, `"ctrl1"` ... for random-direction controls (same prompts and
  doses as real neurons), or `"base"` for the unsteered row (dose 0, one per prompt).
- `"dose_unit":"gain"` rows (ablation): `dose` is the factor k that multiplies the neuron's own activation before the
  down projection (`0` removes it, `2` doubles it, negative inverts), at every position; recipe form `L:N:xK`.
- Engine adds optional keys the app may ignore: `"dose_unit":"rel"|"gain"`, `"prompt_id":int`,
  `"first_diff":int` (index of first token that differs from base, -1 if none), `"ms":int`.
- Greedy decoding unless the record carries `"temp"`.

## steer.bin

`magic "MNST" | u32 version=1 | u32 layer | u32 ffn | u32 k | f32 dose`, then `ffn` records:
`up_id i32[k] | up_dlp f32[k] | dn_id i32[k] | dn_dlp f32[k]`. The change is the natural-log probability change of
the next token at the last position of each context, going from dose 0 to `+dose` (a `rel` dose, see below), taken as
the exact mean over the contexts. `up` = the k tokens with the largest mean rise (best first, dlp > 0), `dn` = the k
with the largest mean fall (most negative first, dlp < 0). Unused slots hold id -1 and 0. Neurons that were not
swept are all -1. One file per (layer, dose): `<layer>.steer.bin` when a single dose was run, else
`<layer>.steer.<dose>.bin`. The contexts are the prompts of the run, listed in `<layer>.steer.json`
(`{"dose":0.5,"contexts":["..."],"null_kl":{"n":16,"p50":0.69,"p95":1.2,"max":1.2}}`).
`<layer>.steer.kl.f32` (same dose suffix rule) is `f32[ffn]`: KL(base || steered) of the next-token distribution,
mean over the contexts, -1 for neurons not swept. `null_kl` = the same KL for the `ctrl=N` random unit directions at the
same dose (the null a neuron's KL is ranked against), `null` when no controls ran.

## Steering unit and recipe

Recipe string `L:N:dose,L:N:dose`, e.g. `14:8949:+1.0,13:7956:-1.0`. Each unit adds a vector to the
residual stream after layer L, the llama.cpp `--control-vector` position (`direction.L` is added
to the output of 0-based layer L, L >= 1):

```
rel (default)  vec = dose * rho[L] * unit(W_down[:, N])
raw            vec = dose * W_down[:, N]
```

Units of one layer add up.

## neuron record (dossier)

Every section optional; a reader skips what is missing. Engine owns `steer`; app owns the rest.

```json
{"v":1,"model":"m","layer":14,"neuron":8949,
 "lens":{"top":[["tok",1.0]],"bottom":[["tok",-1.0]]},
 "steer":{"slens":[{"dose":0.8,"up":[["tok",1.5]],"down":[["tok",-1.2]]}],
          "rollouts":[{"dose":0.8,"prompt":"..","text":".."}]},
 "rank":{"score":0.0,"control_mean":0.0,"control_std":0.0},
 "connections":{"up":[],"down":[]},"fires":[]}
```

## cvec.gguf

GGUF, `general.architecture = "controlvector"`, `controlvector.layer_count = n_layer`,
`controlvector.model_hint = <arch>`, tensors `direction.L` (f32, `[hidden]`) for each layer L in
1 .. n_layer-1 with a non-zero vector. Layer 0 cannot be steered through llama.cpp.
