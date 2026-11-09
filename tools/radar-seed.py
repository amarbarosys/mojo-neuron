"""Seed an anomalies.jsonl for tests/radar.sh: usage radar-seed.py OUT.jsonl. Prints the planted clusters (tool kind unit n) one per line."""
import json, sys
def line(ts, tool, unit, kind="anomaly", res="fvu=nan | non-finite value at 1"):
    a = json.dumps({"layer": unit[0], "neuron": unit[1]}) if unit else "{}"
    return json.dumps({"ts": ts, "level": "info", "msg": "event", "run": "r", "tool": tool, "args": a, "model": "", "msha": "", "git": "x", "kind": kind, "result": res, "receipt": ""})
rows = []
for m in range(5): rows.append(line(f"2026-11-09T10:0{m}:00.000000Z", "lens", (12, 5)))        # c1: 5 events in 5 minutes
for m in range(4): rows.append(line(f"2026-11-09T11:1{m}:00.000000Z", "fires", (10, 7), res="parity miss 0.4"))  # c2
for m in range(3): rows.append(line(f"2026-11-12T09:0{m}:00.000000Z", "lens", (12, 5)))        # c3: same key as c1, 3 days later
rows.append(line("2026-11-10T03:00:00.000000Z", "rank", (3, 9)))                                # singleton
rows.append(line("2026-11-09T10:02:30.000000Z", "lens", (7, 1)))                                # singleton, same tool+time as c1, other unit (tool-only mutant merges it)
rows.append(line("2026-11-02T08:00:00.000000Z", "lens", (12, 5)))                               # previous ISO week (W45)
open(sys.argv[1], "w").write("\n".join(sorted(rows)) + "\n")
print("lens anomaly L12N5 5"); print("fires anomaly L10N7 4"); print("lens anomaly L12N5 3")
