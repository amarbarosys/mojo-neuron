---
name: mojo-neuron
description: "Use when you must explore, steer, ablate or probe the neurons and features of a GGUF language model with mojo-neuron, a Mojo neuron explorer built on the mojo-baro inference engine: logit lens of every neuron, CPU forward that shows where a neuron fires, steering vectors and llama.cpp control vectors, ablation scans and surgery, activation capture, transcoders, tuned lens, truth probes, implants, a research log, an app and an MCP server. Says which model kinds each tool supports, what was verified, what FAILED, and the traps."
---

# mojo-neuron

Interpretability tools for GGUF models, written in Mojo. They answer four questions about a model's MLP neurons
and learned features:

| question | tool family |
|---|---|
| what does a neuron write? | logit lens, tuned lens, R-style ridge lens (CPU, weights only) |
| where does it fire? | CPU forward `fires`, engine activation capture (GPU) |
| what does it do? | steering (vectors, banks, personas, llama.cpp control vectors), ablation scan, surgery |
| what is a feature? | transcoders, probes, implants, labels |

Every tool was built against a written pre-registered bar and a gate with a planted mutant. Many bars FAILED.
Those are listed as FAILs below. Read "Verified FAILs" before you build on a result.

Status words used in this file: **PASS** (gate ran and passed, receipt in the repo), **FAIL** (bar missed, stays
missed), **UNVERIFIED** (no gate ran), **UNSUPPORTED(reason)** (does not work for that case). Numbers come from
`docs/reports/*.md`, `docs/matrix.md`, `docs/parity.md`, `docs/bench.md`, `docs/experiments.md`. If this file and
the repo disagree, the repo wins.

## When to use, when not

Use it when you have a GGUF and want per-neuron or per-feature evidence: "which neurons promote the token X",
"does neuron N fire on this sentence", "what happens to the text when I push neuron N", "how much does the output
change if I remove neuron N", "can I fit a sparse transcoder of layer L".

Do not use it for: encoder models (no output head), mixture-of-experts layers (experts are out of scope),
attention-head analysis, models whose quantisation has no dequantiser in the GGUF reader (`TQ2_0` ternary).
It is not a training framework and not a chat server; the chat and server parts only call a running engine.

## What it needs

| need | detail |
|---|---|
| Mojo toolchain | Builds use `mojo build`. The gates were run with the Mojo that the `mojo-gguf` checkout pins (Mojo 1.2.0 when this file was written). Other versions: UNVERIFIED. Use `def` and `comptime` syntax; `fn`/`alias` are gone in 1.x. |
| source dependencies, by local path | `mojo-gguf` main (import name `ggufx`; needs commit `f5da9cf` or later, see trap 1), `mojo-json`, `mojo-log` (research log), `mojo-imkit` source equal to tag `v0.8.0` (app only), `mojo-subprocess` (engine tools). Pass each as `-I <checkout>/src`. |
| engine | The `mojo-baro` engine (github.com/amarbarosys/mojo-baro), built and baked separately. Needed for steering at the GPU, `sweep`, `reverse`, ablation scan, capture, implants, live chat. Not needed for `lens`, `fires`, `rank`, `rlens`, `cvec`, `connections`, `recipe`, tuned lens fit, probes on existing captures. |
| model | a GGUF file. For the engine, a pack baked from it (`engine-pack.py` in mojo-baro). The Qwythos-class hybrid (`qwen35`) engine reads a pack baked from a **Q4_0-pure** GGUF, not `Q4_K_M`. |
| GPU | Everything GPU-side was tested on one AMD Radeon RX 7900 XTX (gfx1100, ROCm), 24 GB. Other GPUs: UNVERIFIED. The CPU tools need no GPU, but see trap 14: every `mojo build` may open the GPU device. |
| host llama.cpp | `LLAMA_BIN`, used as the numeric oracle and the tokenizer by gates and by `fires`. Use `llama-completion` (the one-shot binary); `llama-cli` waits for chat input even with `-p`. |
| python3 | Only as oracle code in `tools/` (numpy, a gguf reader, PIL for the image gate). Run it as `python3 -I`. |
| RAM | 27B lens checks need about 5 GB of unembedding matrix in RAM (the lens gate is skipped there). Cap builds and runs, for example `systemd-run --user --scope -p MemoryMax=6G`. |

Models used in the matrix (all from public releases): Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct,
granite-4.2-3b, vibethinker-1.5b, minicpm5-2b, spark-x2.5-4b, a Qwen3.5-family 9B hybrid ("Qwythos-9B",
the main engine target, hidden 4096, 32 blocks plus one MTP head), a 27B qwen35 build.

## Build

Binaries land in `.work/` of the checkout. A failed build must not leave an old binary: delete it first.

```
M=<path to mojo>; GG=<mojo-gguf>/src; J=<mojo-json>/src; LG=<mojo-log>/src
mkdir -p .work
rm -f .work/lens .work/nx
$M build src/lens.mojo -I src -I $GG -I $J            -o .work/lens
$M build src/nx.mojo   -I src -I $GG -I $J -I $LG     -o .work/nx     # nx needs all four dirs
$M build src/mcp.mojo  -I src -I $GG -I $J -I $LG     -o .work/mcp
$M build src/rank.mojo -I src -I $J                   -o .work/rank
$M build src/rlens.mojo -I src -I $GG -I $J -I $LG    -o .work/rlens
scripts/build-sweep.sh      # sweep, reverse: needs the engine checkout via BARO=... and mojo-subprocess
scripts/build-app.sh        # the window app, needs mojo-imkit v0.8.0 and a display
```

Every script reads its paths from environment variables with defaults for the authors' machine; override them:
`MOJO`, `GGUF_SRC`, `JSON_SRC`, `LOG_SRC`, `SUBPROC_SRC`, `IMKIT`, `BARO`, `LLAMA_BIN`, model paths `QWEN`, `LLAMA`, `QWEN_BF16`.
A new import in a shared module means every build line of its importers needs a new `-I` (trap 9).

## Run-time environment

| variable | meaning |
|---|---|
| `MNEURON_DIR` | the store (see next section). Default: the `.work/neurons` of the main checkout. |
| `MNEURON_BRAIN` | directory for exported neuron notes (`export_brain`). Default is the authors' note vault; set it. |
| `MNEURON_SYNC` | optional command that indexes exported notes (only when `export_brain` gets `"sync":1`). |
| `MNEURON_BIN` | directory holding the tool binaries `nx surgery` runs. Default `./.work`. |
| `NX_RLOG` | research-log directory; `NX_RLOG=off` disables the log. |
| `NX_CVEC`, `NX_TCCAP` | paths of the `cvec` and `tccap` binaries used by `nx persona` and `nx fingerprint`. |
| `PERSONA_DIR`, `REPLAY_DIR`, `FP_DIR` | stores of `nx persona`, `nx replay`, `nx fingerprint`. |
| `MNEURON_LEDGER` | optional file that `nx live` appends a wall-time line to. |
| `LLAMA_BIN` | host llama.cpp directory (oracle, tokenizer). |

## Store, units, doses

A **unit** is `(model, layer, neuron)`; layers are 0-based. `<model>` is the GGUF file stem in lower case.

```
$MNEURON_DIR/<model>/
  model.json                      arch, n_layer, hidden, ffn, vocab, gguf path, rho[] (one f32 per layer)
  <L>.lens.bin                    logit lens of every neuron of layer L ("MNLN")
  <L>.rlens.bin / rlens.bin       R-style readout / ridge maps ("MNRL")
  <L>.sweep.jsonl                 steered rollouts, append only
  <L>.steer.bin, .steer.json, .steer.kl.f32   steer lens of one dose
  <L>.dossier/<neuron>.json       neuron record
  live.json, live/                engine paths and the last live chat run
cvec files: <out>.cvec.gguf  (llama.cpp)   and   <out>.f32  (the engine's BARO_STEER file)
```

Formats are in `docs/format.md` (lens.bin, sweep.jsonl, steer.bin, recipe, cvec), `docs/format-capture.md`,
`docs/format-implants.md`, `docs/app-formats.md`. A change that breaks a reader bumps the version.

**Recipe string**: `L:N:dose,L:N:dose`, for example `14:8949:+1.0,13:7956:-1.0`. Each unit adds a vector to the residual
stream after block L (the llama.cpp `--control-vector` position; block 0 cannot be steered through llama.cpp and the
cvec file skips it; the engine can). Units of one layer add up.

- `rel` dose (default): `vec = dose * rho[L] * unit(W_down[:, N])`, where `rho[L]` is the median residual norm after layer L.
  A rel dose of 1 already wipes the text of a 0.5B model; useful range is about 0.1 to 1. Positive and negative doses are not
  mirror images (experiment X2: the effect is far from antisymmetric at ±0.5), so run both signs.
- `raw` dose: `vec = dose * W_down[:, N]` (used when `model.json` has no `rho`).
- gain form `L:N:xK`: multiplies the neuron's own activation by K before the down projection (0 removes it, 2 doubles it,
  negative inverts) at every position. In `sweep.jsonl` such rows carry `"dose_unit":"gain"`.

`rho` is measured, not computed by `lens`: `tools/rho.py` (one prompt) or `scripts/rho-cap.sh` (median residual norm of an unsteered
capture). `lens --out` merges into `model.json` and keeps an existing `rho` (gate `tests/modeljson.sh`).

## Which path a model takes (decide first)

```
GGUF --> lens (CPU, unembedding only)        every model in the matrix with a dequantiser
     |-> fires (CPU forward)                 dense llama, qwen2, granite only
     '-> engine (GPU): steer, steer lens, gain/ablate, capture, chat
           qwen35 hybrid (Qwythos 9B, 27B) -> engine.mojo + window.mojo   (capture, scan, implants, splice live here)
           small dense (0.5B..4B)          -> spark.mojo                  (sweep, slens, reverse, BARO_STEER)
           MoE (qwen35moe, qwen4exp)       -> UNSUPPORTED except as a plain residual add
```

Before you edit an engine op, grep for the binary that the target model's architecture really runs. A plan once named
`spark.mojo` for an op that Qwythos runs in `engine.mojo`.

### Model x feature matrix (`docs/matrix.md`, measured with `scripts/lens-matrix.sh`)

| model | arch | lens layer | lens gate | lens.bin | s | fires (CPU forward) |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct q4_K_M | qwen2 | 12 | PASS | PASS | 11 | PASS (bf16 file vs llama.cpp) |
| vibethinker-1.5b q4_K_M | qwen2 | 14 | PASS | PASS | 26 | UNVERIFIED (same code path as qwen2) |
| llama-3.2-1b-instruct q4_K_M | llama | 8 | PASS | PASS | 30 | PASS |
| minicpm5-2b q4_K_M | llama | 20 | PASS | PASS | 26 | UNVERIFIED (minicpm scales not modelled) |
| granite-4.2-3b bf16 | granite | 20 | PASS | PASS | 36 | PASS |
| spark-x2.5-4b q4_K_M | spark2_5 | 18 | PASS | PASS | 67 | UNSUPPORTED(no CPU forward for this arch) |
| 9B qwen35 hybrid q4_K_M | qwen35 | 16 | PASS | PASS | 193 | UNSUPPORTED(hybrid linear attention) |
| 27B qwen35 q4_K_M | qwen35 | 32 | skipped (5 GB of unembedding in RAM) | PASS | 175 | UNSUPPORTED |
| lily-cybersecurity-7b IQ1_S | llama | 16 | skipped | PASS | 15 | UNVERIFIED |
| TriLM-1.5B TQ2_0 | llama | 12 | UNSUPPORTED(ggml type 35 has no dequantiser) | UNSUPPORTED | 0 | UNSUPPORTED |
| 35B MoE builds | qwen35moe, qwen4exp | n/a | UNSUPPORTED(experts out of scope) | UNSUPPORTED | n/a | UNSUPPORTED |
| bge-m3 | bert | n/a | UNSUPPORTED(encoder, no output head) | UNSUPPORTED | n/a | UNSUPPORTED |

"lens gate" = f64 brute-force of sampled neurons plus a neighbour-neuron mutant. Seconds = gate + `lens.bin` for one layer, CPU.

### Steering matrix (`docs/engine-parity.md`)

| model | engine | BARO_STEER gate | steer lens (`slens`) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct bf16 | spark | PASS | PASS |
| Llama-3.2-1B-Instruct Q4_K_M | spark | PASS | PASS |
| granite-4.2-3b bf16 | spark | PASS | PASS |
| 9B qwen35 hybrid, Q4_0 pack | engine.mojo | PASS | UNSUPPORTED(spark op only) |
| 27B cyber qwen35, `--q4` pack | engine.mojo | PARTIAL: 1 of 2 recipes passes the noise-aware rule (pack's base logits are 22 % from llama.cpp) | UNSUPPORTED |
| MoE 35B builds | | UNSUPPORTED | |
| spark-x2.5-4b, minicpm5-2b, vibethinker, Qwen2.5-7B | spark | UNVERIFIED (same code path as the PASS rows) | |

## Reference: CPU tools

All print a final `PASS ...` or `FAIL ...` line when given a check argument; failures exit non-zero.

| tool | usage | does |
|---|---|---|
| `lens` | `lens MODEL LAYER K [CHECK] [--out DIR] [--rmap rlens.bin] [--tc DIR NF]` | Logit lens of every neuron of one layer: `z = W_U · (unit(W_down[:,n]) * output_norm_weight)`, keep top and bottom K tokens. Text output: one line per neuron, tab separated; inside a field the tokens are joined by U+241F. `CHECK` = n: verify n sampled neurons against an f64 brute force (rank order of both poles, plus reversed and neighbour mutants). `--out DIR` writes `DIR/<model>/<L>.lens.bin` and merges `model.json`. `--rmap` reads through ridge maps (R-style lens). `--tc DIR NF` reads transcoder features (8 features took 7.6 s). |
| `rank` | `rank SWEEP.jsonl [N]` | Word-cluster mass of the steered rollouts of each unit against the `ctrl*` random-direction control rows; prints the top N units by cluster mass next to the control statistics. Needs a sweep file. |
| `rlens` | `rlens MODEL CORPUS [--layers a,b] [--windows N] [--holdout N] [--out DIR] [--dump DIR]` | Per-layer ridge map `M_l` from the residual after block l to the residual after the last block (before the final norm), fitted on a text corpus with the CPU forward. Writes `rlens.bin`. Ours, **not** the Jacobian R lens and not J++. |
| `cvec` | `cvec MODEL RECIPE OUT [RHO.f32]` | Writes `OUT.cvec.gguf` (llama.cpp `--control-vector`) and `OUT.f32` (engine `BARO_STEER`). Gate: gguf equals the f32 vector layer by layer and llama.cpp loads it. A bad control vector logs `common_contr: failed to load` yet exits 0: grep the log. |
| `conn_check`, `exp_*` | `exp_steer`, `exp_next`, `exp_conn` | experiment drivers behind `docs/experiments.md`. |
| `tccap` | `tccap MODEL.gguf CORPUS LAYER OUT_DIR [--windows N] [--seg S] [--chunk W]` | CPU capture of one layer (x and y rows) for small models. |

## Reference: `nx CALL JSON`

`nx CALL '{"model":...}'` runs one call and prints one JSON object. `-` reads the arguments from stdin. A failure
prints `FAIL nx: <reason>` and exits 1. `nx chat` is different (a terminal chat). `nx persona` and `nx replay` take
a sub-command word before the JSON: `nx persona save '{...}'`. `<model>` is the store name (GGUF stem, lower case).

The same calls are an MCP server (`.work/mcp`, JSON-RPC lines on stdio, protocol versions 2024-11-05 and 2025-03-26
tested; read fd 0 with `read(2)`, not `input()`; `tools/list` returns the schemas). `live` is not offered over MCP.

### Weights-only calls (need only a `lens.bin` in the store; all model kinds that have one)

| call | arguments | returns |
|---|---|---|
| `models` | none | list of `{model, arch, n_layer, ffn, lens_layers}` for every store entry with `model.json` |
| `lens` | `model, layer, neuron, kind?:"r"` | `{top:[[token, z],..], bottom:[..]}`; `kind:"r"` reads the R-style `<L>.rlens.bin` |
| `search_lens` | `model, layer, token, k?=50` | up to k hits `{neuron, pole:"top"|"bottom", rank, z}` whose lens holds that token (case-insensitive, stripped), in neuron order, not by strength |
| `connections` | `model, layer, neuron, span?=3, k?=12` | `{up:[{layer,neuron,role,cos}], down:[..]}`: weight cosines of this neuron's write direction against the gate/up rows of the next `span` layers (and the reverse). A statement about weights, not behaviour (experiment E4, null) |
| `recipe` | `model, recipe` | the recipe in normal form; refuses layer 0, neuron out of range, bad shape |
| `rank` | `model, layer, k?=50` | top k units of `<L>.sweep.jsonl` with the control statistics; error if no sweep exists |
| `dossier` | `model, layer, neuron, span?` | writes and returns `<L>.dossier/<neuron>.json`: lens, R lens if present, connections, rank if a sweep exists |
| `export_brain` | `model, layer, neuron, sync?=0` | writes a markdown note `L<layer>-N<neuron>.md` under `$MNEURON_BRAIN/<model>/` with the lens words and `[[links]]` to connected neurons; returns `{path}`. `sync:1` runs `$MNEURON_SYNC` (a note-index command, optional) |
| `cvec` | `model, recipe, out?` | like the `cvec` tool through the store: `{layers, recipe, mode:"rel"|"raw", gguf, f32}`; default `out` = `<store>/<model>/dial` |

### CPU forward (dense llama, qwen2, granite only; UNSUPPORTED elsewhere)

| call | arguments | returns |
|---|---|---|
| `fires` | `model, layer, text, k?=20, neuron?` | with `neuron`: `{tokens, activations}` (the SwiGLU activation of that neuron at each token). Without: the k neurons with the highest activation over the text, with the token where each peaks. Tokenisation adds BOS like the engine driver. |

### Engine-backed calls (need engine binaries and a GPU)

| call | arguments | returns / effect |
|---|---|---|
| `live` | `model, layer, neuron, prompt, dose?=0.5 (>0), n?=24` | runs one resident engine sweep of the neuron at -dose and +dose plus the unsteered row; returns `{neg, base, pos, dose, wall_s}`. Needs `<store>/<model>/live.json` = `{"sweep":BIN,"gguf":FILE,"spark":BIN,"pack":DIR,"baro":DIR,"rho":F32?,"runner":CMD?}`. Rows go to `live/<L>.sweep.jsonl`, never the ranked sweep. 3 s for 0.5B on a free GPU; 66 s measured while another job held the card. |
| `surgery` | `model, out, layer, neuron, op, k?, vec?, via?` | runs the `surgery` binary (`$MNEURON_BIN/surgery`), returns its text. See Surgery. |

| `ablate-scan` | `model, out, layer, neurons, prompts, k?, brain?, ...` | runs the ablation scan binary (`$MNEURON_BIN/scan`, `src/causal/scan.mojo`), returns its text. `nx scan` is the live capture scan, a different call. `tests/causal.sh` calls it through a stub. |

### Capture-derived and product calls (read a capture dir, `docs/format-capture.md`)

| call | arguments | returns / effect | gate |
|---|---|---|---|
| `scan` | `cap, tokens?` | live heat strip of a capture: per token the 6 strongest channels of the FFN output; `{tokens, layer, sig}` | `tests/scan.sh`, `tests/scan-real.sh` |
| `sound` | `cap, out, start?=0, tokens?=0` | WAV, 24 kHz mono 16 bit, one note per token (0.12 s), one voice per captured layer, pitch `110 Hz * 2^(3L/n_layers)`, 6 strongest FFN channels set the timbre; deterministic | `tests/sound.sh` |
| `image` | `turns:[{text,words}], out, dry?, runner?` | one 768x768 picture per turn through a local ComfyUI runner; `dry:1` returns only the prompts | `tests/image.sh` |
| `replay` | `record {cap, req, id}`, `frame {id, k, layer?}` | copy a request of a capture, then read record k (token and f16 rows widened to f32) | `tests/replay.sh` |
| `persona` | `list`, `save {name, recipe, gguf|model, rho?, labels?, overwrite?}`, `apply {name}` | stores recipe plus both vectors (`steer.f32`, `steer.cvec.gguf`) plus lens and labels; `apply` checks the sha and returns the engine env and the llama.cpp flag | `tests/persona.sh` |
| `fingerprint` | `op: enroll|match, name, gguf, layer, dir?` | a model fingerprint from the spectrum of a sonified fixed text; `match` returns cosine distances, `unknown` beyond 3x the largest self-distance | `scripts/fingerprint-check.sh` |
| `radar` | `file?, gap_s?, week?, report?, brain?` | clusters `anomalies.jsonl` of the research log (>= 2 events by tool, kind, unit, time gap) into a weekly markdown digest | `tests/radar.sh` |
| `wiki` | `model, dir?, thr?=4.5, k?` or `query` | OPTIONAL, needs a local note-search tool (`brain-ask --net wiki`); links neuron notes to Wikipedia pages by label words; reverse form maps a query back to units | `tests/wiki.sh` |

### Research calls (Qwythos-class captures, CPU)

| call | arguments | returns | verdict |
|---|---|---|---|
| `concept` | `gguf|model, expr, scale?=1, layer?=0, k?=10, out?` | concept arithmetic. Terms `L:N`, `tok:TEXT@L` (`_` = space), `vec:FILE@L`, `mean(CAP,L,a-b)`; unit-normalised, summed per layer; writes `<out>/concept.f32` (steer file, layer count from the model's real block count, not counting an MTP head) and returns the lens of each touched layer. `A - A` is the exact zero vector | CPU and engine gates PASS; language-swap bar FAIL |
| `probe` | `cap? or text?, dir?, layer?, req?, gguf?` | live lie score per token (logit of P(true)) at the best layer, from the probes in `.work/probe/lie/` | built; bars partly FAIL (see FAILs) |
| `lens-diff` | `gguf|model, cap, layer, a, b, k?=10, null?=20, out?` | lens of mean(h(a_i) - h(b_i)); `a`, `b` are 1-based request ranges of equal length; also 20 null lenses | FAIL twice |
| `tlens` | `gguf|model, cap, out, layers?="7,11,15,19,23,27", ntrain?=8000, nval?=800, nheld?=256, lr?` fits; `op:"lens"` with `gguf, cap, map, layer, req, back?=0, k?=10` applies | per-layer affine "tuned lens" | PASS bars 0 to 5 |
| `cast` | `op: fit|gate|score` | request-level condition gate: apply a persona vector only when a condition direction fires | FAIL bar 5 |

## Reference: engine tools and knobs

### Standalone engine tools (need engine sources via `-I`, a pack, a GPU)

| tool | usage | does | gate |
|---|---|---|---|
| `sweep` | `sweep MODEL.gguf OUTDIR layer=.. neurons=.. doses=.. prompts=FILE [mode=slens] [gains=..] [ctrl=N] [k=12] [chat=0] [rho=F]` | one resident engine process runs neurons x doses x prompts. `mode=sweep` writes rollouts (`<L>.sweep.jsonl`); `mode=slens` writes the batched steer lens (`<L>.steer*.bin`, `.json`, `.kl.f32`); `gains=` runs gain ablation; `ctrl=N` adds N random unit-direction controls | `sweep-check.sh`, `slens-check.sh`, `gain-check.sh` |
| `reverse` | `reverse MODEL.gguf OUTDIR prompts=FILE sys_a=TEXT sys_b=TEXT [top=100 swap=0 chat=1]` | persona contrast: residual difference of two system prompts, cosine rank of every neuron of every layer | `reverse-check.sh` |
| `scan` | `scan MODEL OUT layer=.. neurons=.. prompts=FILE [k=0] [ctl=0.99] baro=.. spark=.. pack=.. runner=..` | ablation scan: zero each listed neuron inside one resident engine over 64 prompts, rank by mean KL and by target-token NLL change. `ctl=0.99` adds the noise-floor pass and ranks by excess over it. Prints the ablation mode and metric | `scan-check.sh` |
| `surgery` | `surgery MODEL OUT layer= neuron= op= [k vec via]` | writes a modified GGUF. `op=zero` zeroes the neuron's `ffn_up` row (exact in every ggml type, only that row's bytes change). `via=down` and scale/replace/add print the residual error instead of claiming equality | `tests/causal.sh`, A3 engine gate |

### Engine environment knobs (documented in the engine's settings list)

| knob | effect |
|---|---|
| `BARO_STEER=FILE` | `f32[N_LAYERS * H]`; row L is added to the residual after block L. Fused into the next RMSNorm; decode cost 0.7 %. Turns off batched prefill, speculation and the megakernel (engine.mojo path). |
| `BARO_STEER_SERVE=1` | allocates the steering buffer for serve mode (needed for per-request steer and gain). Turns off batched prefill and speculation. |
| serve request fields | `steer_bank` + `steer_layer` + `steer_row` + `steer_scale`, `steer_file`, `gain "L:N:k,.."`, `op:"slens"`, `implants:0`, `capture`, `steer:[{layer,neuron,scale}]` (E1: `scale` is the gain on FFN activation N, 0 ablates, 1 unchanged) |
| `BARO_CAPTURE=DIR` `BARO_CAPTURE_LAYERS=16,20` `BARO_CAPTURE_FIELDS=xyr` `BARO_CAPTURE_MODEL=name` | activation capture (below) |
| `BARO_IMPLANTS=path` | rank-k side FFN per layer (below) |
| `BARO_PACK`, `BARO_SERVE=1`, `BARO_SPEC=0`, `BARO_TMAX` | pack path, request-per-line serve mode, speculation off, max positions |

Engine ops live on the `ablate` branch (spark: `BARO_STEER`, `sweep`, `slens`, gain) and on `neuron2`/`neuron2b`
(engine.mojo: capture, scan floor, serve passthrough, implants, splice). At the time of the reports the `neuron2` line
had a green `run-tests.sh` but had not passed the engine's merge check, and `neuron2b` carried unverified ops
(`slens_foil`, MoE routing-weight plane, splice). Check that your engine checkout accepts each knob before you plan on it.

## Workflows

### 1. Lens: what does a neuron write?

```
.work/lens $GGUF 12 10 8                   # layer 12, top/bottom 10 tokens, f64-check 8 neurons -> PASS line
.work/lens $GGUF 12 10 --out $MNEURON_DIR  # writes <model>/12.lens.bin and model.json
.work/nx models '{}'
.work/nx lens '{"model":"qwen2.5-0.5b-instruct-q4_k_m","layer":12,"neuron":5}'
.work/nx search_lens '{"model":"...","layer":12,"token":"snake","k":20}'
```

Read it with the evidence below: the weights-only lens is **not** a good guide in the middle of a model. On a
0.5B model (24 layers) the next token at a neuron's strongest firings is in its lens top about as often as chance up to
layer 12 (lift 0.84 to 1.14), 1.7x at layer 16, 3.0x at layer 20, 3.5x at layer 22 (E2). Use it as a pre-filter for the
late third, or fit the tuned lens (workflow 8). Against steering: 5.8 % of layer-12 neurons share a token between the
lens top-12 and the steer-up top-12, 15.9 times a shuffled control, so it ranks neurons only a little (X1, E1).

### 2. Where does a neuron fire? (`fires`, dense llama, qwen2, granite)

```
.work/nx fires '{"model":"...","layer":12,"text":"The snake slithered away","neuron":777}'
.work/nx fires '{"model":"...","layer":12,"text":"The snake slithered away","k":10}'
```

For qwen2 bf16, every activation was compared with an independent numpy forward; for llama, qwen2 and granite the
SwiGLU values were compared with llama.cpp's own `ffn_swiglu` tensor on prompts of at most 6 tokens (llama.cpp prunes the
last layer and elides tokens beyond 6; oracle skips the last layer). Models without a CPU forward need engine capture.

### 3. Capture: per-token activations from the engine

One request line per prompt into one resident engine (see `scripts/capture-qwythos.sh`, `src/features/tcreq.mojo` writes request lines from a corpus):

```
BARO_PACK=<pack> BARO_SPEC=0 BARO_SERVE=1 BARO_TMAX=1088 \
BARO_CAPTURE=<EMPTY existing dir> BARO_CAPTURE_LAYERS=16 BARO_CAPTURE_FIELDS=xy \
BARO_CAPTURE_MODEL=<name> <engine> < requests.jsonl > engine.log
```

- Files: `meta.json`, `L<l>.f16` (one per layer), `tokens.i32`, `req.i32`. Record i is the same token in every file.
  Number of tokens = size of `tokens.i32` / 4. Files are appended, so start from an **empty** directory and never reuse a
  directory across designs.
- Fields (letters of `BARO_CAPTURE_FIELDS`): `x` FFN input (after the FFN RMSNorm), `y` FFN output (down projection before
  the residual add), `r` residual after the layer (after steering if steering is on). `r(l) - y(l)` is the residual after attention/SSM.
- Disk: `nf * hidden * 2` bytes per token per layer. Hidden 4096: `xyr` 24 KiB, `xy` 16 KiB per token per layer.
  10 M tokens of `xyr` for one layer = 245 GB. Use `xy` for transcoders.
- **One position per step (default path)**: batched prefill, speculation, megakernel and checkpoint reuse are off for the
  process; about 97 to 108 tokens/s (100k tokens = 16 min). The engine refuses a step that advances more than one position.
- **Batched path** (when `BARO_STEER_SERVE` is unset): chunks of up to 1024 rows, about 24x faster (2563 against 106 tokens/s on
  2048 tokens). Against the one-position capture of the same text: median relative L2 x 0.0047, y 0.0079, r 0.0034, residual
  cosine at least 0.99665 (5 of 2048 positions under 0.999). **Do not mix batched and one-position captures inside one training set
  without saying so** (rule 15). A 400,384-token batched capture of 782 requests of 512 tokens passed an integrity check (no
  non-finite values, no shared 32-token run with the 100k set).
- Saturation: values beyond f16 range clip to ±65504 and the engine prints `capture: pos P saturated N values`.
- Readers: `numpy.fromfile`, `mmap`. Gate: capture residual rows are bit-equal to the engine's own `BARO_DUMP` at layers 0, 16, 31;
  a layer-shift mutant is rejected (`scripts/capture-check.sh`). Dense models only (an MoE raises).
- Tail a capture live: `nx scan`, the app's scan view, `nx sound`, `nx replay`.

### 4. Steering: vector, bank, persona, control vector for llama.cpp

```
# 1. a recipe, checked
.work/nx recipe '{"model":"...","recipe":"14:8949:+0.5,13:7956:-0.5"}'
# 2. vectors: llama.cpp control vector + engine steer file
.work/nx cvec '{"model":"...","recipe":"14:8949:+0.5","out":".work/dial"}'
llama-completion -m $GGUF --control-vector .work/dial.cvec.gguf -p "Who are you?" -n 32
# 3. engine, one-shot:   BARO_STEER=.work/dial.f32  <engine> ...
# 4. engine, many neurons x doses x prompts in one process:
.work/sweep $GGUF out layer=14 neurons=8949,7956 doses=0.5,-0.5,0 prompts=prompts.txt ctrl=8 ...
.work/rank out/14.sweep.jsonl 20        # which units move word clusters more than the random controls
# 5. a named persona (recipe + both vectors + labels), applied by name
.work/nx persona save '{"name":"x","recipe":"14:8949:+0.5","model":"..."}'
.work/nx persona apply '{"name":"x"}'
```

- Always run the **random-direction controls** (`ctrl=N`) and the **unsteered row** (`"neuron":"base"`) next to every steered row.
- Run a **prompt-only baseline**: in the dream experiments, putting the lens tokens in the prompt (rate 0.396) beat every
  non-collapsed steered run.
- Sweep a coefficient and choose on held-out units. The lens-overlap rate rises with the dose while distinct tokens fall in
  step: part of that "gain" is collapse into repetition (a 128-token greedy run with 4 to 30 distinct tokens).
- A steering oracle needs the engine's exact f32 vector: it rounds activations to bf16, so a vector that differs in the last
  float32 bit shifts logits by 1e-3 to 1e-2, and a tolerance tighter than one bf16 step means nothing (rule 4).
- Layer 0 cannot be steered through llama.cpp. The steer lens op needs a plain dense profile (no gate, no sliding window),
  contexts up to 96 tokens, f32 activations; `gain` needs f32 activations and runs token by token.
- Concept arithmetic: `nx concept '{"gguf":F,"expr":"mean(CAP,23,a-b) + tok:_Paris@16"}'` writes a steer file and the lens.
- Persona from the live engine over HTTP (`baro-serve`, request fields above) equals the direct `steer_file` logits to 1.8e-15
  (bar 1e-3; `scripts/persona-serve-check.sh`).
- Chat: start a baro server on a BARO-baked pack, then `nx chat [--url U] [--steer L:N:gain,..] [--persona NAME] [--max N] [--script FILE]`.
  `steer` is a **gain** on the FFN activation, not a dose.

### 5. Ablation scan and surgery (Qwythos-class engine)

```
scan $GGUF OUT layer=16 neurons=3,1031,2059 prompts=tests/data/eval64.txt ctl=0.99 baro=<engine src> spark=<engine bin> pack=<pack> runner=none
.work/nx surgery '{"model":"...","out":"edited.gguf","layer":16,"neuron":1031,"op":"zero"}'   # ffn_up row zeroed
```

- **Read scan scores only as the excess over the `ctl=0.99` pass.** On the 9B hybrid, every ablation of a sampled layer-16 neuron
  scored KL **7.7e-6 to 1.2e-5**: the engine's bf16 numeric noise floor. Any change that survives bf16 rounding re-rolls the
  rounding downstream and lands on that floor, whatever its size. Gain 0.999 scores exactly 0.0 (the change rounds away), gain 0.99
  already scores about 7e-6. Floor (median of the gain-0.99 pass): 7.671e-6. A neuron is "detectable" only above twice the floor:
  none of 16 sampled layer-16 neurons was.
- Damage does not grow like (k-1)^2 near zero. Neuron 3: 7.8e-6 at gain 0, 9.0e-6 at -1, 1.9e-5 at 10, 1.4e-3 at 100.
- Cost: 57 ms per 10-token request, 3.7 s per neuron at 64 prompts. A full 12,288-neuron layer is about 12.6 h, so scans take a
  neuron list. Mode printed with every scan: zero ablation, clean prompts only, metric KL plus target-token NLL.
- Pre-rank by a cheap linear direct effect (activation x down_proj to the target logit via the lens, then exact gain only on
  the top slice): pre-registered (`docs/prereg-papers-fold.md`) but **not built**.
- Surgery: `via=up` is an exact zero. Zeroing a down column of a K-quant is not exact (0.0122 left against 0.0698 on a 0.5B test).
  The engine reads a pack baked from the Q4_0-pure GGUF, so a surgery file for the engine must be edited from that GGUF and
  re-baked with `engine-pack.py`. Gate: engine pack logits from the surgery file equal engine `gain 16:1031:0` on the original
  pack (max diff 0.000); llama.cpp loads the file and generates. The surgery files are reflinks on btrfs: size deletes with
  exclusive-bytes tools, not `du`.
- Group ablation (A2c-group): groups picked by predicted direct effect move the next-token distribution 10 to 50 times more
  than random groups, but the predicted **sign** did not hold at layer 16 (a late layer is the cheapest next try). MISSED.
- Splice op (engine replaces one layer's FFN output with rows from a file, plus teacher-forced NLL): works; use **KL against the
  unspliced run** as the effect metric, not NLL (the layer's whole effect on mean NLL was 0.0015 and 0.029 nats).

### 6. Transcoders (sparse features of one layer)

Capture `xy` (workflow 3), then train a TopK transcoder x -> y:

```
tc_train DATA LAYER OUT NF K B STEPS LR [--seed S] [--bar F] [--init-var 0.5] [--fp32] [--abort F]
tc_splice ...    # CPU spliced NLL on small models      tcsplice ...  # spliced KL through the engine
tcceil ...       # linear ridge ceiling on the same split (run this FIRST)
lens --tc DIR NF # read features through the unembedding   xmap ...  # cross-model ridge map
```

The recipe that worked (Qwythos 9B, layer 16, F = 8192, 99,840 tokens: 90,112 train, 9,728 held out):

| setting | value |
|---|---|
| peak learning rate | **3e-4**; 5e-4 survives 3000 steps, diverges over 9000; 7e-4 and above diverge |
| Adam beta2 | 0.99 |
| warmup / schedule | 300 steps linear, cosine down to 10 % |
| init | centre and scale inputs (mean norm sqrt(d), folded into We, be on save); rescale the encoder so the first reconstruction has variance 0.5 x total (`--init-var 0.5`) |
| batch | 512 |
| precision | fp16 WMMA default, `--fp32` available; 33 to 35 ms per step |
| guard | held-out FVU every 500 steps; abort above 5 |

The stable learning rate is a property of the **trajectory** (the lr in force around steps 500 to 1500), not of the peak value:
run any long config first at 1500 steps with the same schedule shape, or watch the first 1500 steps.

FVU (fraction of variance unexplained, held out; lower is better):

| run | held-out FVU |
|---|---|
| plain init, F 16384, K 64, lr 5e-4, 3000 steps (first run) | 0.850 |
| + init rescale 0.5 | 0.697 |
| recipe above, K 64, F 8192, 6000 steps, 100k tokens | **0.658** |
| same, K 256 | 0.643 (best 0.633 at step 3500 to 4000, then rises: data-limited at 100k) |
| K 64, 50k tokens / 25k tokens | 0.809 / 1.840 (overfit) |
| 380k tokens, K 64, 6000 steps | 0.695 (new held-out) / 0.681 (old held-out) |
| 380k tokens, K 64, **18000 steps** | **0.641** / 0.633 |
| 380k tokens, K 256, 6000 steps | **0.624** / 0.611 |
| **dense linear ridge**, same 100k split (control) | **0.520** (train 0.447); an independent numpy least squares gave 0.5186 |
| Qwen2.5-0.5B layer 12 rehearsal, F 4096, K 32 | 0.697 (ridge 0.409) |

The sparse transcoder stayed above the dense linear map in every run. The 0.55 target is missed (FAIL, see below). More steps
helped as much as more data between 200k and 380k tokens, so a "floor near 0.67" fitted from fixed-step runs was withdrawn.
Report **seed stability** (0.5B: share of features with a partner at cosine >= 0.7 across two seeds was 0.003) and the
neuron-feature share (cosine >= 0.5 to a `W_down` column, 0.001). Spliced KL on Qwythos L16 (60 positions): identity 1.2e-5 (floor),
zero splice 1.93e-2, transcoder 1.34e-2 = 0.696 of the zero splice against a bar of 0.25 (FAIL).

### 7. Lens variants

- **R-style (ridge) lens**: `rlens` then `lens --rmap`. It turns the layer-12 null of the logit lens into 1.3x to 1.4x lift on
  held-out text (E3, E5), is slightly worse than the logit lens at layer 22, and is **not** the Jacobian R lens or J++ (UNSUPPORTED:
  needs a backward pass and calibration tasks; parity against published Qwen3.5-4B maps UNSUPPORTED).
- **Tuned lens**: `nx tlens` fits a per-layer affine map on captures, held-out full-vocabulary KL(teacher || student), 256 positions
  (PASS bars 0 to 5):

  | layer | logit lens | b-only | tuned |
  |---|---|---|---|
  | 7 | 6.66 | 5.16 | 3.94 |
  | 11 | 6.73 | 5.12 | 3.79 |
  | 15 | 6.63 | 5.15 | 3.70 |
  | 19 | 6.83 | 4.26 | 2.48 |
  | 23 | 6.46 | 3.14 | 2.07 |
  | 27 | 4.36 | 1.77 | 1.19 |

  Top-1 agreement with the teacher lens 0.04 to 0.27 rises to 0.27 to 0.57. Caveats: fitted on a narrow capture (45.7k tokens of
  English encyclopedic statements), the learning rate 3e-5 is the top of the grid, a shuffled-teacher mutant still beats the logit
  lens by 0.8 at 2 of 6 layers (it learns the token prior), so the control that matters is the b-only column. Apply with
  `nx tlens '{"op":"lens",...}'`; at layer 27 it prints ` The`, ` These`, ` A` where the logit lens prints CJK tokens.
  Check any gradient by **directional derivatives**, not entry-wise finite differences (f32 noise grows as the step shrinks).
- **lens-diff** (`nx lens-diff`): FAIL twice (below). Markers must be matched case-insensitively and chosen after looking at the
  tokens a known sample puts on each pole (the lens reads the NEXT token, a capitalised sentence starter after a full stop).

### 8. Probes (truth / lie)

`nx probe '{"cap":DIR}'` or `'{"text":T}'` gives a per-token lie score (logit of P(true)) at the best layer from a logistic probe fitted
per captured layer with a shuffled-label control (`lie_fit`, `lie_steer`, `lie_eval`; `scripts/probe-run.sh` rebuilds them from pairs).
Qwythos 9B, 1921 verified pairs, layers 3,7,...,31, field `r`. Understand the limits before use:

- It reads whether the claim **inside** the sentence is true. It survives paraphrase, hedging, assertion and a sycophantic prefix
  (AUROC 0.87 to 0.91). A negation frame ("It is false that <true fact>") turns it around: AUROC **0.33** (bar 0.70, FAIL).
- Report AUROC **and** accuracy, with a **per-context threshold**: set S5 had AUROC 0.871 but accuracy 0.537 at the probe's own zero threshold.
- A truth probe that must survive negation needs negated families in its training split and a held-out family in the test.
- Steering along the probe direction moves the next layer's score as stated (+28.2 / -28.4 against a class gap of 48.8).
- Control on twin data (a statement and its one-word negation) is bimodal: judge it with a 30-seed null of |metric - 0.5|, never a fixed band.
- On a quantised model the probe and the model's own output agree (probe says true while output says false: 0 to 2.3 %).

### 9. Implants (rank-k side FFN per layer, engine.mojo)

`BARO_IMPLANTS=file` loads `implants.bin` (`docs/format-implants.md`): per layer a key matrix, gates and values;
`delta = sum_i gates_i * max(c_i - tau, 0) * values_i` with `c_i = dot(x, key_i) / |x|`, `x` = the FFN input row, added after the layer's FFN.
Base weights are never touched. No file, or a request with `"implants":0`, is bit-equal to the unmodified engine. A loaded file turns batched
prefill, speculation and the megakernel off. Gate (PASS): k=0 and `implants:0` bit-equal; a keyed prompt moves 4.3 logits, an unrelated
prompt 0, a random key 0. **No implant result (generalisation, memorisation, prompt compiler) exists**: the builder and harness are written,
the result runs were not made. Key an implant at the position that predicts the first wrong token (prompt plus the correct prefix), not the last
prompt position, and run a regression set (locality, general ability, earlier implants) after every implant.

### 10. Research log (Rlog)

Every new op ends with `Rlog(tool, args, model).done(result, receipt, vals, lo, hi)` or `.event(kind, result, receipt)` (`src/mneuron/rlog.mojo`).
Lines go to `<NX_RLOG>/<date>.jsonl`; out-of-range or non-finite values become `anomaly` lines and also go to `anomalies.jsonl`.
`NX_RLOG=off` writes nothing. Misses of pre-registered bars were logged as anomalies. `nx radar` clusters the anomalies into a weekly digest.
The model argument of Rlog is the GGUF **file path** (it hashes it), not a server's model id. Gate `tests/rlog.sh`.

### 11. App and MCP

`scripts/build-app.sh` then `.work/app` (store from `MNEURON_DIR`; theme `MNEURON_THEME=glass|neon|flat`). Views: **map** (layers x 40 neuron buckets),
**list** (table, rank by lens or sweep, token search), **neuron** (lens, connections, fires on your text, recipe buttons, dossier, steered chat),
**dial** (up to 8 units, slider -1 to +1 in rel doses, live blend preview without the engine, export cvec, play capture), **scan** (tails a capture), **replay**
(scrubber over a recorded request). Steered chat has three columns (-dose, none, +dose); **generate** runs `nx live`; tokens are not streamed (one batch per click).
Token text is sanitised to what the font atlas holds; UTF-8 falls back to `IMKIT_FALLBACK` (a CJK font). Scripted checks: `MNEURON_SCRIPT=file .work/app`
with lines `frame`, `wait N`, `click X Y`, `text S`, `wheel D`, `shot PNG`; at exit it prints one `STATE ...` line. It needs a display and the GPU for the UI composition.
MCP: `.work/mcp` is the same calls on stdio; gate with a real client, not only a pipe (a pipe passed once while a real client got "Connection closed").

### 12. Agents on the engine

`scripts/agents/baro-serve.sh up|down` starts the Rust `baro-serve` on an engine plus pack, then `scripts/agents/{qwen-code,omp,hermes}.sh` run a least-privilege
smoke task (the random first heading of a README they can only read through a tool call). Run agent CLIs without auto-approve flags, in a throwaway directory, with
**absolute** paths in the task; judge two verdicts, what the tool call read and what the answer says. `baro serve` rebuilds its engine from sources embedded in the
GGUF and failed on this toolchain (`existing function with conflicting signature` on `read`): start `baro-serve --engine ENGINE --pack PACK` directly.

## Measured numbers and their caveats

`gne` below is gguf-neuron-explorer (MIT, a Python/llama.cpp tool of the same kind), the comparison target; no code was copied from it or from neuron-explorer (`docs/provenance.md`).

All speeds: one machine (Ryzen 7 7800X3D, RX 7900 XTX, 64 GB), one afternoon, single runs unless a table says more.

| claim | number | caveat | source |
|---|---|---|---|
| steer lens, all 4864 neurons of a layer + 64 controls, ±0.5 | 28.8 to 29.0 s vs gne's 119.4 to 128.9 s (4.3x); 5.9 ms vs 24 to 26 ms per unit | one 0.5B model, gne's own 4 contexts (4 to 36 tokens); contexts over 96 tokens refused; q8 pack vs bf16 | `docs/bench.md` |
| agreement with gne | KL correlation 0.987 (+) and 0.990 (-); top-12 up-token overlap 0.86 | rho from one prompt vs a corpus (median KL 0.342 vs 0.388) | `docs/bench.md` |
| vs llama.cpp on 3 sampled neurons | top-8 overlap 8/8, 6/8, 7/8; KL within 3 % | | `docs/engine-parity.md` |
| rollouts, 120 | 11.1 to 12.2 s vs gne 15.3 to 18.8 s (1.4x) | decode-bound (about 1.6 ms/token); process start included | `docs/bench.md` |
| `BARO_STEER` decode cost | 0.7 % (606.5 vs 602.0 tokens/s, median of 6) | worst case: tiny layers | `docs/bench.md` |
| lens of a layer | 11 s (0.5B, gate + bin) to 193 s (9B) | CPU; includes the f64 golden unless skipped | `docs/matrix.md` |
| reverse lookup | top and bottom 10 of 116,736 neurons equal an f64 oracle; B - A mutant rejected | gne's `standardize`, `pca1`, `--verify` rollouts not built | `docs/engine-parity.md` |
| gain ablation vs HF fp32 hook | rel err 0.057 (bar 0.10), neighbour mutant 1.16 | directional ablation is host-side `BARO_ABLATE`, not re-gated | `docs/engine-parity.md` |
| persona dial (0.5B) | rel err 0.0482; top-1 785/785; llama.cpp loads the gguf and changes its text | negated vector response is not linear (mutant cosine +0.005, its bar missed) | `docs/reports/2026-11-09-app.md` |
| capture | 24 KiB (xyr) or 16 KiB (xy) per token per layer at hidden 4096; 97 to 108 tokens/s one-position, 24x batched | batched numerics differ slightly | `docs/format-capture.md` |
| ablation noise floor (9B, layer 16) | KL 7.7e-6 to 1.2e-5 | a floor, not an effect | causal report |
| tuned lens | KL 3.9 to 1.2 vs logit lens 6.7 to 4.4 | narrow data, 256 held-out positions | agent report |
| lie probe | AUROC 0.87 to 0.91 plain, paraphrase, hedge, assertion, sycophantic | 0.33 on negation frame | probe report |
| logit lens predicts steering | 4.67 % of neurons share a token, 0.26 % chance (18x); 15.2 % in the strongest 5 % of lens scores | 95 % share nothing; layer 12 only | E1 |
| logit lens predicts next token at firings | lift 1.05 (layer 12), 1.73 (16), 3.03 (20), 3.53 (22) | one model, a 13 KB corpus; part of the late lift is token repetition | E2 |
| R lens vs logit lens | layer 12 lift 1.44 (held out 1.33), layer 22 3.29 vs 3.53 | helps mid-model only | E3, E5 |
| connections predict co-firing | mean signed correlation 0.002 to 0.005 vs random about 0; positives 49.5 to 52 % | null at every layer measured | E4 |
| model fingerprint | self distance 1e-16, second nearest 0.0225, 0.0225, 0.0398; held-out model flagged unknown (nearest 0.0314); shuffled-band control 0.976 | small margins | product report |
| radar | 3 planted clusters recovered exactly | a tool-only mutant needed a planted counter-case | product report |

## Verified FAILs and missed bars (they stay missed; do not quote the "post hoc" numbers as passes)

| item | verdict | numbers | what it means |
|---|---|---|---|
| **A2** ablation scan, first pre-registered run | **FAIL** | its 1e-6 absolute tolerance was wider than the spread of the scores (24 neurons at 7.7e-6 to 1.2e-5); the single re-runs were bit-identical to the batch | passes only as A2b with a floor pass and relative tolerance (one run, 16 neurons, none detectable) |
| **A2c-group** | **MISSED** | floor 3.0e-3 on logit difference; supporters beat it at k=32 but random groups are as large, opposers have the wrong sign, dose not monotone | magnitude pre-rank works, sign at layer 16 does not |
| splice NLL bars | **FAIL** | the layer's effect on mean NLL (0.0015 and 0.029 nats) is below the guessed 0.05 | KL bars written after run 1 pass: POST HOC, not independent |
| **A5** MoE router capture | not gated | engine code builds for qwen35moe; no MoE pack on the test machine | UNVERIFIED, parked |
| **F24** lie probe | bar 1 (accuracy >= 0.80) **MISSED** 0.777 at L19; bar 2 (shuffled control in [0.42,0.58]) **MISSED**: 0.32 0.33 0.33 0.34 0.66 | the band assumed independent statements; twin data is bimodal; surface-only probes score 0.50 to 0.58 so there is no surface leakage | bars 3, 3e, 4 PASS (steering +28.2/-28.4, vector applied, base equals capture) |
| **F24b** negation set S4 | bar 3 (>= 0.70) **MISSED**: 0.331 | probe follows the embedded proposition | bars 1, 2, 4, 5 PASS (S0 0.913; S1 0.882, S2 0.904, S3 0.913, S5 0.871) |
| **F3** language-swap lens | **MISSED**: L27 6 French markers on top, 0 English in the bottom 50 | marker list lacked capitalised and subordinator tokens; a broader list (17 French on top, 10 English at the bottom) is POST HOC | `A - A` exact zero and engine bit-equality PASS |
| **F26 / F26b / F26c** dreams | **FAIL x3** | prereg scale 3/8 (bar 2 missed); 2.5x 4/8 and distinct 36.1 < 40; coefficient sweep: held-out rate 0 at the peak m=1 | monotone: rate up, distinct tokens down (collapse); prompting the lens tokens beats every non-collapsed steered dream (0.396). 5x (8/8) is post hoc |
| **lens-diff** | **FAIL x2** | run 1 L27 top 6 FR / 0 EN (miss bar 2); run 2 top 7 FR / 0 EN, bottom 13 EN, consistency 1.0, but 1 of 20 null lenses held 7 French markers (max 3) | the pole is real; the sign-flip null is the weak part |
| **CAST** gate | **FAIL bar 5** | false-fire rate 0.123 (37 of 300 English statements) vs limit 0.05; bars 1 to 4 hold (held-out accuracy 1.0, shuffled control 0.469, 40/40 gate outputs byte-equal, AUC 1.0) | condition direction is not specific to French at layer 3 |
| **B1** transcoder bars | **FAIL** | Qwythos L16: held-out FVU 0.850 vs 0.30; spliced KL ratio 0.696 vs 0.25; 0.5B rehearsal 0.697 vs 0.60; seed stability 0.003 vs 0.30 | trainer machinery PASS (3 steps vs f64 reference, WMMA cosine >= 0.995, 3 planted defects fail, byte-identical determinism) |
| **B1 0.55 chase** | **FAIL** | K 64: 0.641 at 380k tokens and 18000 steps; K 256: 0.624; the 180k-420k-token crossing estimate was **retracted** | ridge control on the new split unmeasured |
| B2 feature steer | **FAIL** | diag overlap 0 / 0.02 / 0.125 vs off 0.009 / 0.036 / 0 | features were poor |
| B3 cross-model map | **FAIL** | held-out R² -0.312 (bar 0.50), match ratio 1.06 (bar 1.5); position 0 of each window dropped (post hoc): R² 0.201 | |
| B4 labels | **FAIL** | kept 0 of 12; AUROC 0.517 vs 0.516 shuffled, recall 0 | pipeline refuses labels for featureless units; it was never shown to accept a good one |
| F30b labels vs wiki | **MISSED** | recall 5/10 (bar 8), flip rate 12/30 (bar 6) | null units 0/6 false links held |
| C2 dial mutant | one bar **missed** | mutant cosine +0.005 (bar 5) | mutant caught by rel err 1.078 |
| X2 antisymmetry | falsified | 0.2 % of neurons share >= 6 of 12 up/down tokens; KL ratio in [0.5,2] for 63 % | steer both signs |
| X3, X3b persona neurons | falsified, no power | 0 of 20; then 2 of 100 vs 3 of 400, p 0.26 | |

Not built or not run: A4 attribution graph (needs transcoders that pass), G2 gate, G3, G4 (implant results), B5, F2 engine half (pin format only),
F5, F25, F27, F29, CAST tier 2, gne's `standardize` / `pca1` / `--verify`, gain steering `xN` LoRA export, OpenAI-compatible endpoint with `when`,
J++ and Jacobian R lens.

## Traps and rules (each cost a run; generalised from `docs/rules.md`)

1. **After any `mojo-gguf` sync, run `scripts/check.sh` before trusting a build.** A sync once reverted a thread-context fix: `deq_rows` returned zeros, `lens` printed the same tokens for every neuron, and the gate said "mutant passed". A model output identical for all units: dump the raw `z` and the norm weights before suspecting the math.
2. **Stale binaries pass failed builds.** Delete the binary you are about to build. One gate passed for every contributor because an old `.work/nx` sat there. A receipt names the binary's build commit; verify by rebuilding from the claimed commit in a clean checkout. One lane's engine binaries predated the commit they were said to verify.
3. **Pre-register before the result run.** Write the bar, the data, the control and the mutant in a committed file first. A missed bar is reported and stays missed. A rule written after the data may annotate a verdict, never flip it. Label post hoc numbers POST HOC. Replace an unrun prereg by a new one; do not edit it.
4. **Compare tools' inputs before suspecting the engine**: tokenise with BOS like the driver (a no-BOS oracle made a PASS look like 0/8 overlap). A steered oracle uses the engine's exact f32 vector.
5. **bf16 floor.** Any ablation that survives bf16 rounding of the FFN activation scores about 7e-6 to 1.2e-5 KL on the 9B. Set a gate tolerance from a pilot measurement, relative not absolute, and make a mutant at least 10x the tolerance away.
6. **A gate must compare what the user sees, in full** (rank order and both poles, every activation), and carry a mutant that only the stronger comparison can see (reversed order, transposed stride, zeroed interior neurons). A sum cannot tell a reversed-token mutant apart. Plant the exact case that tells a mutant from the original. A mutant must differ from the original at every tested index, including the boundary (a layer-shift mutant once mapped the last layer onto itself).
7. **Every argument and every sentence of a tool description gets a gate that fails when it is untrue.** (`k` parsed and ignored; "indexes the note" without indexing; the protocol version echoed.)
8. **A writer of a shared file owns named fields and merges the rest** (`model.json`: `lens --out` once overwrote `rho`). Write to a temp file and rename.
9. **Every build line of every importer of a shared module names all source dirs** (`nx`, `mcp`, `app` need `-I mojo-log`). A new import means grep every build line in the same commit.
10. **Capture needs one position per step.** "Prefill off" is not enough: a replay window fed four prompt rows per step. A mode that needs m=1 forces every batching knob off in code and raises on a step that advances more than one.
11. **Keep capture sets uniform**: never mix batched and one-position captures, nor reuse a capture dir across designs (the engine appends). Report FVU on a held-out slice of each set when two sets exist.
12. **A tool with a divisor checks the divisor** and exits non-zero with `FAIL` (an empty held-out split printed `nan` and exit 0; a half-length file printed `inf`). A metric that does not exist for a set is `none`, never `0.0`. Size fixtures from the format doc's record size.
13. **Job scripts run with a bare environment.** A job scheduler may not pass `PATH` or your variables: name every binary by absolute path, export each variable inside the command, and check `[ -f MODEL ]` (a symlink target) in the shell **before** queueing a long GPU job. Run every CLI branch of a helper on saved inputs before queueing. Never wrap a queued job in an outer `timeout`: it loses its slot while waiting.
14. **Every `mojo build` may open the GPU device**, even for CPU code. A hung GPU can abort a CPU build.
15. **Do not reuse one name for a package and a build root** (`src/app/` beside `src/app.mojo` is not found): sensory modules live in `src/sense/`. Run outputs go in `.work/<tool>s/<run>`, binaries stay `.work/<tool>`.
16. **Mojo**: `out` is a keyword (use `nout`, `root`, `dir`); `var xs = [a]` is a fixed Array (use `List[T]()`); `ptr.unsafe_offset(n)`; `print` stays buffered when stdout is a pipe/FIFO, so a process that talks to a peer writes with `write(2)`; clamp byte slices with `min(len)` (an out-of-range slice aborts with no message); keep a `String` alive across `unsafe_ptr()`; a stand-in engine must print the real protocol's exact compact JSON bytes; two FIFO peers must not both open their read end first.
17. **llama.cpp** prunes the last layer to the output positions and elides tokens past 6, so an oracle uses prompts of at most 6 tokens and skips the last layer. `llama-cli` waits for chat input; use `llama-completion`. Read ggml tensors through `nb[]` strides (`ffn_moe_topk` is a view): when only position 0 matches, suspect the reader.
18. **Rank by the right thing.** A lens, a probe and a steer can agree on tokens and still be wrong about behaviour; read scan scores as excess over a control; quote probe accuracy only with a per-context threshold; judge a control on twin data by a 30-seed null.
19. **Prompting beats steering, probes and edits in several results here.** Always run the prompt-only baseline.
20. **Marker lists and bands are written after looking at the data's shape** (case, leading space, subordinators; twin bimodality), not before.
21. **A gradient check on an f32 forward uses directional derivatives**, not entries.
22. **A rerun never deletes receipts; it moves them.** A generator that wants N distinct items asserts N <= population first. Gate scripts count with `|| true` or test the file first, run once with `bash -x` before trusting silence, never negate a pipeline's exit, never `producer | grep -q` under `pipefail`.
23. **Never `ln -sf` onto a path that already holds a file** (it replaced a real engine binary with a dangling link).
24. **Size an engine job from a measurement before you queue it**: capture 12.5 ms per record on one engine; scan 3.7 s per neuron at 64 prompts; 10M tokens at 107 tokens/s is 26 h.
25. **Heavy parallel builds can exceed the CPU thermal limit**: one box reset with seven parallel build sessions. Cap concurrent builders and use one job per builder.
26. **A smoke task for an agent CLI names absolute paths** and proves itself in a clean checkout; a gate proven only in the checkout that wrote it is not proven.
27. **Plans are not authority.** A plan said the wiki network was `--net text` (it is a separate net), a capture op lived in `spark.mojo` (Qwythos runs `engine.mojo`), the layer count was `blk.*` tensors (the MTP head is block 32; the engine wants `f32[32 * 4096]`). Probe the named store or file once before building on it.

## How to verify

`scripts/check.sh` (CPU only, about 3 minutes, exits non-zero on the first failure, ends with `PASS check`). It builds `lens`, `conn_check`, `fires_check`, `nx`
and runs, in order:

| step | what it proves |
|---|---|
| lens on Qwen2.5-0.5B (layer 12) and Llama-3.2-1B (layer 8) | rank order of both poles equals an f64 brute force on sampled neurons; reversed and neighbour mutants differ |
| `tests/rank.sh` | rank golden plus two mutants |
| `tests/rlog.sh` | normal and anomaly lines; forced Inf becomes an anomaly in both files; `off` writes nothing; a blinded build is caught |
| `lens --out` and `tools/check_lens_bin.py` | binary format valid |
| `tests/modeljson.sh` | `model.json` merge keeps `rho`, `gguf` absolute |
| `conn_check` | connections golden plus mutant |
| `fires_check` (qwen2 bf16, llama) | activations vs llama.cpp `ffn_swiglu`, plus every activation vs a numpy forward for qwen2 bf16 |
| `tests/rlens.sh` | ridge maps vs numpy, held-out beats identity |
| `tests/calls.sh` | the `nx` calls against independent paths, recipe refusals, dossier JSON, note export |
| `tests/sound.sh`, `tests/chat.sh`, `tests/tap.sh` | golden WAV and FFT pitch; chat against a mock server; agent tool call really read the file |
| `tests/radar.sh`, `persona.sh`, `wiki.sh`, `replay.sh` | product features (wiki needs a note-search tool) |
| `lie`, `lie-steer`, `lie-eval`, `concept`, `circuit` | probe CPU halves on synthetic data with mutants |

What needs a **GPU and the engine** (not in `check.sh`): `scripts/steer-check.sh` family (`engine-steer-check.sh`, `engine-slens-check.sh`, `slens-check.sh`,
`sweep-check.sh`, `gain-check.sh`, `reverse-check.sh`), `scripts/capture-check.sh`, `scan-check.sh`, `implants-check.sh`, `persona-engine-check.sh`,
`persona-serve-check.sh`, `e1-check.sh`, `tc-check.sh` and `tc-det.sh` (transcoder: 3 steps vs f64 reference, 3 mutants, WMMA), `probe-run.sh`, `concept-engine-check.sh`,
`concept-lang-check.sh`, `tests/live.sh`, `tests/app.sh` (display and GPU), `tests/mcp_client.sh` (a real Claude client, spends tokens), `tests/image.sh`,
`tests/agents-live.sh`, `tests/dial.sh`. The last clean-tree receipt of `scripts/check.sh` in the repo is PASS at commit `306f806`; later merges ran their own gates, listed in their reports.

A gate is proven only when it passes in a clean checkout, fails with its planted mutant, and the binary it ran was built from the commit it names.
Judge every claim by what a user would meet: a UI is looked at, a call is run through the same path the client uses, a fix reproduces the symptom first.

## Where the truth lives

`README.md` (tools and gates), `docs/matrix.md`, `docs/parity.md`, `docs/engine.md`, `docs/engine-parity.md`, `docs/bench.md`, `docs/format*.md`, `docs/app.md`,
`docs/experiments.md` (E1 to E5, X1 to X3b), `docs/rules.md`, `docs/prereg-*.md` (every bar, written before its run), `docs/reports/*.md` (verdicts),
`docs/plans/` (plans), `.work/verify/*.md` (independent re-verification receipts).
