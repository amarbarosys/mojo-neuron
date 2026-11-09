"""sweep-verify OUTDIR LAYER GGUF RHO CVEC BARO SPARK PACK: each sweep record equals an independent one-shot spark run
(BARO_STEER from the cvec tool, same prompt ids). A different neuron's vector must give different tokens somewhere."""
import json, subprocess, sys, tempfile, os
out, layer, gguf, rho, cvec, baro, spark, pack = sys.argv[1:9]
recs = [json.loads(l) for l in open(f"{out}/{layer}.sweep.jsonl")]
reqs = [json.loads(l) for l in open(f"{out}/.req.jsonl")]
assert len(recs) == len(reqs), (len(recs), len(reqs))
tmp = tempfile.mkdtemp(dir=out)

def oneshot(ids, steer):
    p = f"{tmp}/p.ids"
    open(p, "w").write(" ".join(map(str, ids)))
    env = ["env", f"BARO_PROMPT={p}", f"BARO_PACK={pack}", f"BARO_GEN={len(recs[0]['tokens'])}", "BARO_SPEC=0"] + os.environ.get("ONESHOT_ENV", "").split()
    if steer:
        env.append(f"BARO_STEER={steer}")
    r = subprocess.run(["gpu-wait", "run", "--timeout", "60", "--"] + env + [spark], cwd=baro, capture_output=True, text=True)
    for l in r.stdout.splitlines():
        if l.lower().startswith("generated:"):
            return [int(x) for x in l.split()[1:]]
    raise SystemExit("FAIL oneshot: " + r.stdout[-300:] + r.stderr[-300:])

cache, bad, n = {}, 0, 0
for rec, rq in zip(recs, reqs):
    ids = rq["prompt"]
    if rec["neuron"] == "base":
        steer = None
    elif isinstance(rec["neuron"], int):
        vec = f"{tmp}/v{rec['neuron']}_{rec['dose']}"
        if not os.path.exists(vec + ".f32"):
            subprocess.run([cvec, gguf, f"{layer}:{rec['neuron']}:{rec['dose']}", vec, rho], check=True, capture_output=True)
        steer = vec + ".f32"
    else:
        continue
    key = (tuple(ids), steer)
    cache[key] = cache.get(key) or oneshot(ids, steer)
    n += 1
    if cache[key][:len(rec["tokens"])] != rec["tokens"]:  # engine.mojo one-shot may run on past BARO_GEN
        bad += 1
        print("MISMATCH", rec["neuron"], rec["dose"], rec["prompt_id"], cache[key][:6], rec["tokens"][:6])
if bad:
    raise SystemExit(f"FAIL sweep-verify: {bad} of {n} records differ from the one-shot run")
# mutant: swap two neurons' tokens, they must differ somewhere
byk = {}
for r in recs:
    if isinstance(r["neuron"], int):
        byk[(r["neuron"], r["dose"], r["prompt_id"])] = r["tokens"]
ns = sorted({k[0] for k in byk})
diff = sum(byk[(ns[0], d, p)] != byk[(ns[1], d, p)] for (_, d, p) in byk if (ns[0], d, p) in byk and (ns[1], d, p) in byk)
if diff == 0:
    raise SystemExit("FAIL sweep-verify mutant: two neurons give identical tokens, the check cannot tell them apart")
print(f"PASS sweep-verify: {n} records equal their one-shot run; neurons {ns[0]} and {ns[1]} differ in {diff} cases")
