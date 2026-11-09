"""persona-check PERSONA.json UNIT LABEL: schema fields present, the recipe's lens entry for UNIT has tokens, the label survived."""
import json, sys
p = json.load(open(sys.argv[1])); unit, label = sys.argv[2], sys.argv[3]
for k in ("v", "name", "model", "gguf", "recipe", "labels", "lens", "sha_f32", "sha_cvec"):
    assert k in p, "FAIL persona: persona.json lacks " + k
assert p["recipe"] == "12:5:+1.0,13:7:-0.5", p["recipe"]
l = [e for e in p["lens"] if e["unit"] == unit]
assert l and l[0]["top"], "FAIL persona: no lens tokens for " + unit
assert p["labels"][unit] == label, "FAIL persona: label lost"
