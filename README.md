# mojo-neuron

mojo-neuron reads the neurons of a GGUF language model in 11 seconds on a CPU (one layer of a 0.5B model, gate included) and steers them through the
[mojo-baro](https://github.com/amarbarosys/mojo-baro) engine 4.3 times faster than gguf-neuron-explorer. Most of the research we built on top of those
tools missed its pre-registered bar: transcoders, a lie probe, model dreams, a condition gate. This README lists the misses next to the passes.

It is a neuron explorer written in Mojo. It computes the logit lens of every neuron of a layer, shows where a neuron fires on your text (a CPU forward
for dense llama, qwen2 and granite, checked against llama.cpp), steers and ablates neurons on the GPU, captures activations for transcoders and probes,
and offers the same calls as a command line (`nx`), an MCP server and a small window app. Every tool has a gate with a planted defect.

## What it needs

| need | notes |
|---|---|
| Mojo | the toolchain pinned by the `mojo-gguf` checkout (Mojo 1.2.0 when this was written); other versions are untested |
| source dependencies | `mojo-gguf` (import name `ggufx`), `mojo-json`, `mojo-imkit` v0.8.0 (app only): from prefix.dev or from source. `mojo-log` and `mojo-subprocess` (engine tools): source checkouts. Each is passed with `-I <checkout>/src` |
| engine (GPU tools only) | [mojo-baro](https://github.com/amarbarosys/mojo-baro), built and baked separately. Needed for steering on the GPU, `sweep`, `reverse`, ablation scan, capture, implants, live chat |
| a model | a GGUF file; for the engine, a pack baked from it |
| GPU (engine tools only) | tested on one AMD RX 7900 XTX (gfx1100, ROCm), 24 GB. Other GPUs are untested |
| llama.cpp | `LLAMA_BIN`: oracle and tokenizer for the gates |

The weights-only tools (`lens`, `rank`, `rlens`, `cvec`, `connections`, `recipe`, `fires`) need neither the engine nor a GPU. Note that `mojo build`
may open the GPU device even for CPU code.

## Build

```
M=mojo; GG=../mojo-gguf/src; J=../mojo-json/src; LG=../mojo-log/src
mkdir -p .work
$M build src/lens.mojo -I src -I $GG -I $J        -o .work/lens
$M build src/nx.mojo   -I src -I $GG -I $J -I $LG -o .work/nx
$M build src/mcp.mojo  -I src -I $GG -I $J -I $LG -o .work/mcp
scripts/build-sweep.sh     # sweep, reverse (needs BARO=<mojo-baro checkout>)
scripts/build-app.sh       # window app (needs mojo-imkit v0.8.0 and a display)
```

The scripts read their paths from `MOJO`, `GGUF_SRC`, `JSON_SRC`, `LOG_SRC`, `SUBPROC_SRC`, `IMKIT`, `BARO` and `LLAMA_BIN`; set them for your machine.

## Quickstart: the logit lens of one layer on the CPU

```
.work/lens qwen2.5-0.5b-instruct-q4_K_M.gguf 12 10 8
```

This reads layer 12 and keeps the top and bottom 10 tokens of every neuron. The last argument checks 8 sampled neurons against an f64 brute force:
rank order of both poles must match, and a reversed and a neighbour-neuron mutant must differ. Add `--out DIR` to write `DIR/<model>/12.lens.bin` and
`model.json`, then ask questions through `nx`:

```
export MNEURON_DIR=DIR
.work/nx models '{}'
.work/nx lens        '{"model":"qwen2.5-0.5b-instruct-q4_k_m","layer":12,"neuron":5}'
.work/nx search_lens '{"model":"qwen2.5-0.5b-instruct-q4_k_m","layer":12,"token":"snake","k":20}'
.work/nx fires       '{"model":"qwen2.5-0.5b-instruct-q4_k_m","layer":12,"text":"The snake slithered away","neuron":777}'
```

Lens plus gate took 11 s for this model and layer (`docs/matrix.md`). Read the output with the caveat under Results: the weights-only lens is at
chance in the middle of a model.

## Tools

| tool | what | gate |
|---|---|---|
| `lens MODEL LAYER K [CHECK] [--out DIR] [--rmap F]` | logit lens (or R-style readout) of every neuron of a layer; text or `lens.bin` | `scripts/check.sh`: f64 golden + mutants |
| `rank SWEEP.jsonl [N]` | word-cluster mass of steered rollouts against `ctrl*` random-direction controls | `tests/rank.sh` |
| `rlens MODEL CORPUS ...` | per-layer ridge transport maps (our R-style lens) | `tests/rlens.sh` against numpy |
| `cvec MODEL L:N:dose,... OUT` | llama.cpp control vector plus the engine's steer file | `scripts/steer-check.sh` (GPU) |
| `nx CALL JSON` | `models lens search_lens rank connections fires dossier recipe cvec export_brain live` and the calls below | `tests/calls.sh` |
| `nx fires` | activation of a neuron per token (dense llama, qwen2, granite) | `src/fires_check.mojo` vs llama.cpp `ffn_swiglu`; qwen2 bf16 also vs a numpy forward |
| `.work/mcp` | the same calls as an MCP server on stdio | `tests/mcp.sh`, `tests/mcp_client.sh` |
| `.work/app` | window: map, list, neuron view, steered chat, dial, scan, replay (`docs/app.md`) | `tests/app.sh`, `tests/live.sh` (GPU) |
| `sweep`, `reverse` | engine: neurons x doses x prompts in one resident process; steer lens; gain ablation; persona contrast | `scripts/sweep-check.sh`, `slens-check.sh`, `gain-check.sh`, `reverse-check.sh` (GPU) |
| `ablate-scan`, `surgery` | engine: ablation scan with a noise-floor pass; exact zeroing of a neuron's `ffn_up` row | `scripts/scan-check.sh`, `tests/causal.sh` (GPU) |
| capture (`BARO_CAPTURE`) | per-token FFN input, FFN output and residual as f16 (`docs/format-capture.md`) | `scripts/capture-check.sh` (GPU) |
| `tc_train`, `tc_splice`, `tcceil`, `xmap` | TopK transcoder trainer, spliced NLL, linear ridge ceiling, cross-model map | `scripts/tc-check.sh`, `scripts/tc-det.sh` (GPU) |
| `nx concept`, `nx probe` | concept arithmetic; live lie score per token | `tests/concept.sh`, `tests/lie.sh` (CPU); `scripts/probe-run.sh` (GPU) |
| `nx lens-diff`, `nx tlens`, `nx cast` | contrastive lens, tuned lens, request-level condition gate | `tests/lensdiff.sh`, `tests/tlens.sh`, `tests/cast.sh` |
| `nx sound`, `scan`, `replay`, `persona`, `fingerprint`, `radar` | capture to WAV, live heat strip, replay, named persona store, model fingerprint, anomaly digest | `tests/sound.sh`, `scan.sh`, `replay.sh`, `persona.sh`, `scripts/fingerprint-check.sh`, `tests/radar.sh` |
| implants (`BARO_IMPLANTS`) | rank-k side FFN per layer in the engine (`docs/format-implants.md`) | `scripts/implants-check.sh` (GPU) |

`scripts/check.sh` runs the CPU gates in about 3 minutes and ends with `PASS check`. `skills/mojo-neuron/SKILL.md` is the long guide: every call with
its arguments, step-by-step workflows, traps, and what each gate covers.

## Which models work

From `docs/matrix.md` (lens) and `docs/engine-parity.md` (steering); the full tables are in those files.

| model kind | lens | fires (CPU forward) | steering on the engine |
|---|---|---|---|
| qwen2, llama, granite, dense up to 4B | PASS | PASS (qwen2, llama, granite); UNVERIFIED for their siblings | PASS (Qwen2.5-0.5B, Llama-3.2-1B, granite-4.2-3b) |
| qwen35 hybrid, 9B and 27B | PASS | UNSUPPORTED (hybrid linear attention) | PASS on 9B; PARTIAL on 27B (1 of 2 recipes) |
| `TQ2_0` ternary | UNSUPPORTED (no dequantiser) | UNSUPPORTED | UNSUPPORTED |
| MoE, encoders | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED (a plain residual add only) |

## Results

Pre-registered bars are in `docs/prereg-*.md`, written before each result run. Verdicts and receipts are in `docs/reports/`. A missed bar stays missed;
numbers marked post hoc in the reports are not passes.

### The tools pass their gates

| item | result | source |
|---|---|---|
| lens, f64 golden plus mutants | PASS on qwen2, llama, granite, spark2_5 and qwen35 rows | `docs/matrix.md` |
| steer lens speed | 4864 neurons plus 64 controls at ±0.5 in 28.8 to 29.0 s against 119.4 to 128.9 s for gguf-neuron-explorer (4.3x); one 0.5B model, its own contexts | `docs/bench.md` |
| steering decode cost | 0.7 % (606.5 against 602.0 tokens/s) | `docs/bench.md` |
| steer lens against llama.cpp | top-8 overlap 8/8, 6/8, 7/8; KL within 3 % | `docs/engine-parity.md` |
| gain ablation | against a Hugging Face fp32 hook: rel err 0.057 (bar 0.10), neighbour mutant 1.16 | `docs/engine-parity.md` |
| capture | residual rows bit-equal to the engine's own dump at layers 0, 16, 31; shifted-layer mutant rejected | `docs/reports/2026-11-09-causal.md` |
| surgery (`ffn_up` row zero) | engine pack logits equal engine gain 0 (max diff 0.000); llama.cpp loads the file | `docs/reports/2026-11-09-causal.md` |
| implants in the engine | k=0 and `"implants":0` bit-equal to the plain engine; a keyed prompt moves 4.3 logits, an unrelated prompt 0 | `docs/reports/2026-11-09-causal.md` |
| tuned lens (`nx tlens`) | held-out KL to the teacher 3.94 to 1.19 against 6.66 to 4.36 for the logit lens, at 6 of 6 layers; narrow data, 256 positions | `docs/reports/2026-11-09-agent.md` |
| transcoder trainer machinery | 3 steps against an f64 reference, WMMA cosine >= 0.995, 3 planted defects fail, byte-identical determinism | `docs/reports/2026-11-09-features.md` |
| persona store | vector byte-identical to the direct recipe; llama.cpp loads it; engine logits equal | `docs/reports/2026-11-09-product.md` |
| lie probe on plain and paraphrased statements | AUROC 0.87 to 0.91; steering along it moves the score +28.2 / -28.4 against a class gap of 48.8 | `docs/reports/2026-11-09-probe.md` |

### The research on top of them mostly failed

| item | result | source |
|---|---|---|
| lie probe flips on negation | "It is false that <true fact>": AUROC **0.331** against a bar of 0.70. Accuracy bar 0.80 missed (0.777). The shuffled-control band was missed because twin data is bimodal | `docs/reports/2026-11-09-probe.md` |
| transcoder 0.55 bar | held-out FVU **0.641** (K=64, 380k tokens, 18000 steps) and **0.624** (K=256), against a dense linear ridge at 0.520 on the 100k split. First runs: 0.850 against a bar of 0.30. An estimate that 0.55 would be crossed near 180k to 420k tokens was retracted | `docs/reports/2026-11-09-features-trainer.md`, `...-features.md` |
| transcoder splice, features, maps, labels | spliced KL 0.696 of the zero splice against a bar of 0.25; feature steering, the cross-model map (R² -0.312 against 0.50) and labels (0 of 12 kept) all missed | `docs/reports/2026-11-09-features.md` |
| ablation scan, first run | failed its own mutant bar: the absolute tolerance was wider than the spread of the scores. It passes only as the re-run with a floor pass | `docs/reports/2026-11-09-causal.md` |
| ablation effect on the 9B | every sampled layer-16 neuron scores KL 7.7e-6 to 1.2e-5, the bf16 noise floor; none of 16 neurons was detectable above twice the floor | `docs/reports/2026-11-09-causal.md` |
| group ablation | MISSED: the predicted sign did not hold at layer 16 | `docs/reports/2026-11-09-causal.md` |
| model dreams | three pre-registered runs FAIL. The lens-overlap rate rises only as the text collapses into repetition; putting the lens tokens in the prompt (0.396) beats every steered run that did not collapse | `docs/reports/2026-11-09-agent.md` |
| lens-diff | FAIL twice: the French pole is real, the sign-flip null is not clean | `docs/reports/2026-11-09-agent.md` |
| CAST condition gate | FAIL bar 5: 12.3 % false fires on English statements against a limit of 5 % | `docs/reports/2026-11-09-agent.md` |
| concept arithmetic, language swap | MISSED (6 French markers on top, 0 English at the bottom); the broader marker count is post hoc | `docs/reports/2026-11-09-probe.md` |
| the logit lens as a guide | at chance up to layer 12 of 24 and 3.0x chance at layer 20; weight-cosine connections do not predict co-firing (null at every layer measured); a neuron's effect at ±0.5 is not antisymmetric | `docs/experiments.md` |
| MoE router capture (A5) | MISSED: expert ids equal to the llama.cpp router in 198 of 240 cells (bar: all) | `docs/reports/2026-11-09-causal.md` |
| persona neurons | neurons aligned with a persona direction did not raise persona words (the test was underpowered) | `docs/experiments.md` |

Not built or not run: the attribution graph (it needs transcoders that pass), implant generalisation and memorisation results (running), gain
steering `xN` with LoRA export, J++ and the Jacobian R lens. `skills/mojo-neuron/SKILL.md` has the full list.

## Limits

- One machine and one GPU were tested. Engine ops are tested on a 0.5B to 4B dense family and one 9B hybrid; the 27B build is partial.
- The CPU forward (`fires`) exists for dense llama, qwen2 and granite only.
- The R-style lens here is a ridge map, not the Jacobian R lens.
- The transcoder does not beat a linear map at the data sizes tried (up to about 380k tokens, one layer).
- The lie probe reads the claim inside a sentence, not the sentence, and needs a per-context threshold.
- Engine capture runs one position per step unless the batched path is used. The two captures differ slightly and must not be mixed in one training set.
- Engine ops live on engine branches. Check that your mojo-baro checkout accepts a knob before relying on it.
- The steer lens refuses contexts over 96 tokens and needs a plain dense profile.

To check any claim above, run `scripts/check.sh` for the CPU gates, or open the named report: every number here is copied from it, and every bar is in the
matching `docs/prereg-*.md`. What this repository does not show is that any of the failed items cannot work: each report names the untried alternatives.

## Provenance and license

Ideas from gguf-neuron-explorer and neuron-explorer (both MIT); no code copied (`docs/provenance.md`). Licensed under Apache-2.0 (`LICENSE`).
