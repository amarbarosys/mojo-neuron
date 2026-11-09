# Prereg F30: Wikipedia neuron map

Rule (set from calibration on queries NOT used in the gate: 9 hand queries and 79 real qwen2.5-0.5B layer-12 labels): a wiki neuron
from `brain-ask --net wiki` is a confident link when score >= 4.5 AND its title shares a word with the label (first 5 letters).
Calibration: topical labels scored 5.1 to 10.7, nonsense and subword labels <= 1.7, one junk layer-12 label 4.18 (`wiki/Ernő Prónai`),
stop-word labels 3.8 (so stop words are dropped, labels under 2 words are not asked). Layer-22 sample of 50: 1 label above 4.5.
Today's labels are lens tokens: coverage is low by construction, labels improve after B4.

Bars, `tests/wiki.sh` (planted notes in the real note format, `nx wiki`):
- Known unit (promotes chlorophyll photosynthesis sunlight leaves plants): a link titled Photosynthesis or Chlorophyll.
- Mutant label (zzqxjv blorptangle vkjhqw) and the real junk label of the 4.18 case (erno ertain ilan noreferrer olib olie probante):
  no link, no `## Wikipedia` section, recorded `best` below 4.5 (the threshold, not a bug, excludes them).
- Reverse `nx wiki {"query":"photosynthesis"}`: the known unit is in the top 3 units and neither junk unit appears.
- Two forward runs leave exactly one `## Wikipedia` section; every call appends an Rlog line (tool `wiki`).
Reported, not barred: the rate of linked units on real exported neurons.
