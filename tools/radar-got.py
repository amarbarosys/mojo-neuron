"""Read `nx radar` JSON on stdin, assert the week/event/isolated counts, print sorted `tool kind unit n` lines."""
import json, sys
d = json.load(sys.stdin)
assert d["week"] == "2026-W46" and d["events"] == 14 and d["isolated"] == 2, d
print("\n".join(sorted("%s %s %s %d" % (c["tool"], c["kind"], c["unit"], c["n"]) for c in d["clusters"])))
