# Prereg F4: sonification fingerprint (waits for C3)

`nx fingerprint {model, text?}`: a fixed probe text (`tests/fingerprint-probe.txt`, committed) goes through the engine capture, C3 renders the
sound, the spectral signature is the mean log-magnitude spectrum in 64 bands (+ per-layer pitch track), stored in `.work/fingerprints/<model>.json`.
`nx fingerprint match {model}` returns the nearest stored model by cosine distance and its distance.

Bars (3 enrolled models, one held-out 4th):
- Each of the 3 models, fingerprinted again from a fresh run, matches ITSELF (nearest = itself) and its distance to itself is below half
  its distance to the second nearest.
- The 4th model, not enrolled, is flagged `unknown`: its nearest distance exceeds the largest self-distance seen (re-run noise) times 3.
- Re-run noise (same model twice) is reported; a shuffled-band control fingerprint matches no model.
- Rlog line per call; the WAV of each model is kept for the maintainer to hear (rendered to `.work/fingerprints/<model>.wav`).
