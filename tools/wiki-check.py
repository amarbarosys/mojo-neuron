"""wiki-check FILE [reverse]: map rows (wiki.jsonl) or the reverse answer, against the preregistered bars."""
import json, sys
if len(sys.argv) > 2:
    r = json.load(open(sys.argv[1])); u = r["units"]
    top3 = [(x["layer"], x["neuron"]) for x in u[:3]]
    assert (30, 1) in top3, f"FAIL wiki: known unit not in the top 3 of the reverse query: {u}"
    assert not any((x["layer"], x["neuron"]) in ((30, 2), (12, 1933)) for x in u), "FAIL wiki: a junk unit came back from the reverse query"
else:
    rows = {(r["layer"], r["neuron"]): r for r in map(json.loads, open(sys.argv[1]))}
    assert rows[(30, 1)]["links"] and rows[(30, 1)]["label"].startswith("chlorophyll"), "FAIL wiki: known unit unlinked or label polluted: %s" % rows[(30, 1)]
    for k in ((30, 2), (12, 1933)):
        assert not rows[k]["links"] and rows[k]["best"] < 4.5, f"FAIL wiki: junk unit {k} {rows[k]}"
    print("map:", {k: (v["best"], [l["title"] for l in v["links"]]) for k, v in rows.items()})
