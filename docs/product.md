# Product lane tools (src/product/, section F lane R)

| call | does | gate |
|---|---|---|
| `nx radar {file?, gap_s?, week?, report?, brain?}` | clusters `.work/rlog/anomalies.jsonl` by tool, kind, unit, time gap; digest `docs/reports/radar-<week>.md` + Brain note | `tests/radar.sh` |
| `nx persona list` / `save {name, recipe, gguf or model, rho?, labels?}` / `apply {name}` | store in `$PERSONA_DIR` (default main checkout `.work/personas/<name>/`): `persona.json`, `steer.f32` (BARO_STEER or per-request `steer_file`), `steer.cvec.gguf` (llama.cpp `--control-vector`); `apply` verifies the vector sha | `tests/persona.sh`, `scripts/persona-engine-check.sh` |
| `nx wiki {model, dir?, thr?, k?}` / `{query, model?, k?}` | neuron notes -> wiki neurons (`brain-ask --net wiki`, score >= 4.5 and a shared word); reverse query -> units | `tests/wiki.sh` |
| `nx replay record {cap, req, id}` / `frame {id, k, layer?}` | one capture request -> `$REPLAY_DIR/<id>`; frame k = token + exact f32 rows | `tests/replay.sh` |

| `nx wiki {model, layer, steer:1}` / `{..., neuron, title}` | F30b steer-first: units of `<layer>.steer.bin` labelled by the tokens steering raises; a link needs a title word sharing a stem with them | `scripts/wiki-steer-check.sh`, `tests/wiki-synth.sh` (2 bars missed, see the report) |
| `nx fingerprint {op: enroll|match, name, gguf, layer}` | probe text -> tccap -> C3 sound -> 64-band signature; match by cosine distance | `scripts/fingerprint-check.sh` |
| `nx persona apply` -> `steer`; `nx chat --persona NAME` | persona picked by name through baro-serve (E1 `steer: {"file": ...}`) | `scripts/persona-serve-check.sh`, `tests/chat.sh` |
| app scan view scrubber (`scrub K`, `play` script hooks) | slider over tokens of a capture or replay dir, play one frame | `tests/replay-ui.sh` |

Env: `NX_CVEC`, `NX_TCCAP`, `FP_DIR` (cvec binary), `PERSONA_DIR`, `REPLAY_DIR`, `MNEURON_BRAIN`, `MNEURON_DIR`, `NX_RLOG`. Every call ends with an Rlog line.
Open: F1b strength sweep and CAST (prereg only), F30 labels wait for B4; preregs `docs/prereg-f{1,1b,1c,4,9,10,30,30b}.md`.
