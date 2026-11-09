# Prereg F10: brain-scan replay (waits for C1 and C3; data side needs only the capture format v1)

Record: `nx replay record {cap, req, id}` copies request `req` of a capture dir (`docs/format-capture.md`: `L<l>.f16`, `tokens.i32`, `req.i32`,
`meta.json`) to `.work/replay/<id>/` (same layout, one request, positions renumbered from 0) (token ids stay in `tokens.i32`; the app decodes them with the model store).
`nx replay frame {id, k, layer}` returns the record at token k (heat = the stored rows). The app scrubber (imkit slider, heat strip, C3 sound)
calls `frame` and nothing else.

Bars:
- Scripted: for a capture of 3 requests, every k of the recorded request: `frame k` equals record k of the original request, bit for bit
  (oracle `tools/replay-oracle.py`, numpy, reads the ORIGINAL files); a mutant that shifts k by one must fail the gate.
- The other requests' tokens are absent from the replay; token count = that request's count.
- Rlog line per call. After C1 and C3: the scrubber opens, frame k's strip equals the live strip of token k (C1 receipt) and the sound for
  frame k is the C3 sound of capture record k (golden WAV).
