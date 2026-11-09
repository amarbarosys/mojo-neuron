"""tap-reads.py LOG SMOKE_DIR: what the agent's tool calls read. Prints `right N` (a README.md path equal to SMOKE_DIR/README.md, or relative), `wrong N` (any other
absolute README.md path) and `calls N` (tool calls seen). Exit 0 when at least one right path was asked for and no wrong one."""
import json, re, sys
log, smoke = sys.argv[1], sys.argv[2].rstrip("/")
seen = {}
for l in open(log):
    for m in json.loads(l).get("messages", []):
        for tc in m.get("tool_calls") or []:
            seen[tc.get("id") or json.dumps(tc, sort_keys=True)] = tc.get("function", {}).get("arguments", "")
right = wrong = 0
for a in seen.values():
    for s in re.findall(r'"([^"]*README\.md)"', a if isinstance(a, str) else json.dumps(a)):
        if s in ("README.md", "./README.md") or s == smoke + "/README.md": right += 1
        elif s.startswith("/"): wrong += 1
print("right %d wrong %d calls %d" % (right, wrong, len(seen)))
sys.exit(0 if right > 0 and wrong == 0 else 1)
