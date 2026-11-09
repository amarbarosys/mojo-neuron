"""wiki-seed DIR MODEL: plant three neuron notes in the export_brain format (known topic, nonsense mutant, the real junk label of the 4.18 case)."""
import sys
d, m = sys.argv[1], sys.argv[2]
notes = {(30, 1): "chlorophyll photosynthesis sunlight leaves plants", (30, 2): "zzqxjv blorptangle vkjhqw flarbnog",
         (12, 1933): "erno ertain ilan noreferrer olib olie probante"}
import os
os.makedirs(f"{d}/{m}", exist_ok=True)
for (l, n), t in notes.items():
    toks = " ".join(f"` {w}`" for w in t.split())
    open(f"{d}/{m}/L{l}-N{n}.md", "w").write(f"# mojo-neuron neuron {m} layer {l} neuron {n}\n\nMLP neuron {n} of layer {l} in model {m}. Its output direction promotes the tokens {toks} and suppresses `x` `y` (logit lens).\n\n## Connections\n\n## Related\n- [[mojo-neuron/2026-11-04-start]]\n")
