# Prereg F9: anomaly radar

Bar, fixed before the result run:
- Seeded `anomalies.jsonl` (tools/radar-seed.py) with 3 planted clusters + 2 singletons (one shares tool and time with cluster 1 at another unit) + 1 event in another ISO week.
  `nx radar` reports exactly 3 clusters, each equal to the planted (tool, kind, unit, n); isolated = 2; the other-week event is excluded.
- Clusters 1 and 3 share tool+kind+unit and sit 3 days apart: they must stay two clusters (time split).
- Digest file `radar-<week>.md` holds 3 table rows; Brain note `<today>-radar-<week>.md` is written under the given Brain dir.
- Mutant (`-D RADAR_MUT=1`, clusters by tool only) must fail the gate.
- Every run appends one Rlog line (tool `radar`).
