"""wiki-steer-check NX STORE_MODEL_DIR MODEL LAYER SUMMARY.json: F30b section B bars (docs/prereg-f30b.md) on a steer-first run."""
import json, math, subprocess, sys
nx, mdir, model, layer, summ = sys.argv[1:6]; layer = int(layer)
s = json.load(open(summ)); rows = [json.loads(l) for l in open(f"{mdir}/{layer}.wiki.jsonl")]
def conf(n, t): return json.loads(subprocess.run([nx, "wiki", json.dumps({"model": model, "layer": layer, "steer": 1, "neuron": n, "title": t})], check=True, capture_output=True, text=True).stdout)["confirmed"]
u = next(r for r in rows if r["up"])
word = u["up"].split()[0]
assert conf(u["neuron"], word.capitalize() + " article") == 1, "FAIL wiki-steer: confirm() rejects a title built from the unit's own up word (check is vacuous)"
assert "chess" not in u["up"] and conf(u["neuron"], "Chess") == 0, "FAIL wiki-steer: the planted Chess link was confirmed"
lim = math.ceil(0.05 * s["linked"])
print(f"units {s['units']}  linked {s['linked']}  lens_agree {s['lens_agree']}  null hits {s['null_hits']} (limit {lim})")
for r in rows:
    if r["links"]: print("  link", r["neuron"], r["links"], "<- steer up:", r["up"])
assert s["null_hits"] <= lim, f"FAIL wiki-steer: shuffled null confirmed {s['null_hits']} of {s['linked']} links (limit {lim})"
