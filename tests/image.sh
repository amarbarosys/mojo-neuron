#!/usr/bin/env bash
# Gate C4: one picture per turn from the strongest features, rendered by ComfyUI (zimage) inside one resident `with-comfy` under gpu-wait.
# Checks: the prompt carries the feature words (dry run); 2 turns -> 2 distinct, non-blank PNGs with sidecars from ONE server start; rlog line;
# gpu-wait log: the image job (21 GB) and an engine-sized exclusive job queued at the same time never ran together.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p .work/image
M=${MOJO:-$HOME/Projects/mojo/mojo-gguf/.venv/bin/mojo}; GG=${GGUF_SRC:-$HOME/Projects/mojo/mojo-gguf/src}; J=${JSON_SRC:-$HOME/Projects/mojo/mojo-json/src}; LG=${LOG_SRC:-$HOME/Projects/mojo/mojo-log/src}
PY=${PY:-$HOME/Projects/mojo/mojo-baro/.venv/bin/python}; D=$PWD/.work/image; fail() { echo "FAIL image: $1"; exit 1; }
rm -f .work/nx_image; timeout 400 "$M" build src/nx.mojo -I src -I "$GG" -I "$J" -I "$LG" -o .work/nx_image 2>&1 | grep -E "error" -A4 && fail "build nx" || true
[ -x .work/nx_image ] || fail "no .work/nx_image"
export NX_RLOG=$D/rlog; rm -rf $D/out $D/rlog
T='{"turns":[{"text":"Who are you?","words":["ĠCGRect","snake","Snake","ab","slither"]},{"text":"Tell me a story","words":["ocean","storm"]}],"out":"'$D'/out"'
dry=$(.work/nx_image image "$T,\"dry\":1}")
echo "$dry" | grep -q 'Who are you?, cgrect, snake, slither, abstract' && echo "$dry" | grep -q 'ocean, storm' || fail "prompt lacks the feature words (cleaned, de-duplicated): $dry"
echo "ok   prompts: $dry"
# an engine-sized exclusive job submitted first; the image job queues behind or ahead of it, never beside it
gpu-wait run --label fake-engine --vram 22 --timeout 120 -- sleep 15 > $D/fake.log 2>&1 &
sleep 2
t0=$(date +%s); res=$(.work/nx_image image "$T}" 2>&1) || fail "image call: $res"; wait
echo "$res" | grep -q '"images"' || fail "no images in the reply: $res"
for i in 1 2; do [ -s $D/out/turn-$i.png ] && [ -s $D/out/turn-$i.md ] || fail "turn-$i.png or its sidecar missing"; done
python3 -I - $D/out <<'PYEOF' || exit 1
import sys, hashlib, numpy as np
from PIL import Image
d = sys.argv[1]; hs = []
for i in (1, 2):
    im = Image.open("%s/turn-%d.png" % (d, i)).convert("RGB"); a = np.asarray(im, dtype=float)
    print("turn-%d %dx%d std %.1f" % (i, im.width, im.height, a.std())); assert a.std() > 8, "blank picture"; hs.append(hashlib.sha256(a.tobytes()).hexdigest())
assert hs[0] != hs[1], "both turns drew the same picture"
PYEOF
grep -q "snake" $D/out/turn-1.md && grep -q "ocean" $D/out/turn-2.md || fail "sidecar prompts lack the feature words"
$PY - <<'PYEOF' || exit 1
import json, subprocess, sys
jobs = json.loads(subprocess.run(["gpu-wait", "list", "--all", "--json"], capture_output=True, text=True).stdout)["jobs"]
pick = lambda lab: [j for j in jobs if (j.get("label") or "") == lab and j["state"] == "succeeded"]
img, fake = pick("mneuron-image"), pick("fake-engine")
assert img and fake, "gpu-wait has no finished record of the image job and the fake engine job"
i, f = img[-1], fake[-1]
print("image job %s vram %.0f GB ran %.0f..%.0f s; fake engine %.0f..%.0f s" % (i["id"], i["estimated_vram_gb"], i["started_at"] % 1000, i["ended_at"] % 1000, f["started_at"] % 1000, f["ended_at"] % 1000))
assert i["estimated_vram_gb"] >= 21
assert max(i["started_at"], f["started_at"]) >= min(i["ended_at"], f["ended_at"]), "the image job and the engine-sized job overlapped"
assert max(i["submitted_at"], f["submitted_at"]) <= min(i["started_at"], f["started_at"]) + 1, "the test was not meaningful: one job finished before the other was queued"
PYEOF
grep -q '"tool":"image"' $D/rlog/2*.jsonl || fail "no rlog line"
setsid xdg-open $D/out/turn-1.png >/dev/null 2>&1 & setsid xdg-open $D/out/turn-2.png >/dev/null 2>&1 &
echo "PASS image: 2 turns, 2 pictures in $(( $(date +%s) - t0 )) s through one with-comfy, no overlap with the exclusive job, rlog"
