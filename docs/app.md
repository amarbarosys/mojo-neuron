# App

`scripts/build-app.sh` then `.work/app` (store: `MNEURON_DIR`, default the shared `.work/neurons` of the main repo).
Views: **map** (layers x 40 neuron buckets, colour = best lens score, click a cell), **list** (imkit table of neuron, score, promoted tokens; ranked by lens score or sweep rank, token search over the lens, wheel or scrollbar), **neuron** (lens, connections up/down clickable, fires on your text, recipe buttons,
brain note, dossier file, steered chat panel). Every button runs a call from `src/mneuron/calls.mojo`, the same code as the CLI and MCP.

The steered chat has three columns (- dose, no steering, + dose) for one prompt. It shows the rows of `<layer>.sweep.jsonl` for the
neuron (the engine's `"neuron":"base"` row of the same prompt fills the middle column), or, after **generate**, a live run: `nx live` starts
the engine lane's resident sweep (one spark process) for the neuron at -dose and +dose plus the unsteered row on the prompt in the text box.
It needs `<store>/<model>/live.json` with the engine paths (`sweep`, `gguf`, `spark`, `pack`, `baro`, optional `rho`, `runner`; the values
of the engine's `scripts/sweep-check.sh`). The run goes through `gpu-wait`, is logged in `gpu-ledger.md` (wall time, queue wait included),
and writes to `<store>/<model>/live/<layer>.sweep.jsonl`, not to the ranked sweep file. Not offered over MCP. 0.5B on a free GPU: 3 s;
66 s measured while the engine lane held the GPU. `tests/live.sh` checks the call against the engine's sweep run directly.

Script mode for checks: `MNEURON_SCRIPT=file .work/app` with lines `frame`, `wait N`, `click X Y`, `text S`, `wheel D`, `shot PNG`;
at exit it prints one `STATE ...` line. `tests/app.sh` is the gate, `tests/app_script.txt` the script. Needs a display and the GPU
(imkit composes the UI on the GPU; not a compute job).

## imkit v0.8.0 (C0)

Built against mojo-imkit sources equal to tag v0.8.0 (`scripts/build-app.sh` diffs `src/` against the tag). The UI takes the theme from
`MNEURON_THEME=glass|neon|flat` (default glass). Token text is UTF-8: the font falls back to `IMKIT_FALLBACK` (default Noto Sans CJK)
for codepoints Roboto lacks; `clip` cuts by codepoint and never inside a character (`MNEURON_CLIPTEST=1 .work/app` prints the proof line, checked
by `tests/app.sh`). The fires view plots the activations per token (`Ui.plot`) under the token chips.

## Persona dial, sound, live scan (v2 items C2, C3, C1)

**dial** view: on a neuron view press **add to dial** (up to 8 units, dose starts at +0.25). Each unit has a slider, -1 to +1 in multiples of the layer's
residual norm `rho` (rel doses; raw column units when `model.json` has no rho; +-1 already wipes the text, which is why the range is not wider).
The right panel is the live blend preview: for every token the sum over units of `dose * lens score` (promotes positive, suppresses negative),
no engine involved. **export cvec** calls `cvec` (same `write_cvec` as the CLI, `nx cvec`): `<store>/<model>/dial.f32` (engine `BARO_STEER` file) and
`dial.cvec.gguf` (llama.cpp `--control-vector`). Gate `tests/dial.sh`, bars in `docs/prereg-C2.md`.

**Sound** (`src/sense/sound.mojo`, `nx sound`): a capture becomes one note per token (0.12 s), one voice per captured layer; pitch = 110 Hz * 2^(3 L / n_layers),
timbre = the 6 strongest channels of the FFN output (channel c adds harmonic 1 + c % 8), loudness = row norm. 24 kHz mono 16-bit WAV, deterministic
(gate `tests/sound.sh`: golden sha256 on a synthetic capture, FFT pitch per layer). **play capture** in the dial view renders the first 96 tokens of
`$MNEURON_CAP` (default `<store>/../cap/<model>`) and streams them to the SDL audio device (imkit `Audio`); scripted runs only render.

**scan** view (`src/sense/scan.mojo`, `nx scan`): tails a capture dir (`$MNEURON_CAP`, press **open**) while an engine writes it: per token the 6
strongest channels of the FFN output as a heat strip under the word (red +, blue -), last 60 tokens; click a cell for the channel dossier (the channel over
all tokens as a plot, strongest tokens). A channel is a residual direction, not a neuron (neuron activations need the FFN weights on top of the captured
input: not built). Gate `tests/scan.sh` appends tokens while the app runs and compares the strip with numpy.

Script mode adds `state FILE` (write the STATE line now) and `waitfile FILE` (frames until it exists).

## Chat through baro (E2) and agent CLIs on baro (E3)

`scripts/agents/baro-serve.sh up|down|status` keeps one resident `baro-serve` (OpenAI API, default port 18090, `BARO_PORT`) as a gpu-wait job
(`mneuron-baro-serve-<port>`, 12 GB declared). The Rust server runs directly on `engine-9b` and `baro-serve` built from baro 7c37795 by `scripts/agents/baro-e1.sh` (detached worktree `mojo-baro-lanes/app-e1`, receipt `.work/baro-e1/SHA`) and the Qwythos-9B pack `mojo-baro/.work/e11b-qwythos-pack` (Q4_0-pure of claude-mythos-5-1m) with a `tokenizer.json`
symlink overlay: `tools/baro serve` is not used, its engine rebuild from the embedded sources fails on this toolchain (read(2) signature clash).

`nx chat [--url U] [--steer L:N:gain,..] [--max N] [--script FILE] [--up GGUF]`: terminal chat (history kept, one POST per turn). `--steer` sends the request field
`steer: [{layer, neuron, scale}]`; with the E1 server (baro 7c37795, built by `scripts/agents/baro-e1.sh`) it is the engine's gain: FFN activation N of layer L times `scale`
(0 ablates, 1 leaves it, not an additive dose). A pre-E1 server ignores the field, so `baro-serve.sh` only starts the E1 build, with `BARO_STEER_SERVE=1` (the engine refuses `steer` otherwise; costs batched prefill and speculation, `STEER_SERVE=0` turns it off). Every turn ends with an rlog line (tokens, tok/s);
a reply without `choices` is an anomaly and a non-zero exit. `--script` takes one user line per turn, `--up` starts the server first and stops it after.
Transcript `.work/chat/<run>.jsonl`. Gate `tests/chat.sh` (mock OpenAI server, CPU) and `tests/chat-live.sh` (Qwythos, GPU).

`scripts/agents/{qwen-code,omp,hermes}.sh`: each points its CLI at `http://127.0.0.1:<port>/v1` with the server's model id and runs one headless task, "read README.md and
name its first heading", in a temp dir whose heading is random per run, so a correct answer proves a real file-read tool call. `omp` uses its own isolated profile
`baro-nx`. `BARO_STEER_PROFILE` is refused: the CLIs cannot add the per-request `steer` field and baro has no server-side default.
Each agent script talks to the server through `tools/openai-tap.py` (a pass-through proxy that logs request bodies) and ends with two separate verdicts: **read-path** (`tools/tap-reads.py`: a tool call asked for `<smoke dir>/README.md` or the relative name, no other absolute README.md) and **answer** (the random heading is in the reply). A wrong file and a model misreading the right file are told apart; `tests/agents-live.sh` prints both counts. The tap log replaces parsing hermes `state.db` and the omp and qwen logs: the request history holds every earlier tool call in one format.
