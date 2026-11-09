"""fake-engine OUTDIR LENS.bin MODE: stand-in for the baro engine serve loop (CPU tests of src/agent/dream.mojo).
Speaks the serve protocol on stdin/stdout. A request with steer_row r emits (MODE planted) the lens top-k ids of unit r's neuron,
cycling; unsteered or MODE plain emits a fixed pseudo-random stream. Reads OUTDIR/units.tsv, written by `dream run` before ready."""
import json, struct, sys, time
import numpy as np

out, lensp, mode = sys.argv[1:4]
b = open(lensp, "rb").read()
ffn, k = struct.unpack_from("<II", b, 12)[0], struct.unpack_from("<I", b, 20)[0]
ids = np.frombuffer(b, dtype=np.int32, offset=24).reshape(ffn, 4, k)[:, 0, :]  # part 0 = top ids
t0 = time.time()
while not __import__("os").path.exists(f"{out}/units.tsv") and time.time() - t0 < 120:
    time.sleep(0.05)
time.sleep(0.2)
units = {}
for l in open(f"{out}/units.tsv"):
    r, kind, n, own = l.split()
    units[int(r)] = int(n)
print('{"ready":true,"tmax":256}', flush=True); print('ready sent', len(units), 'units', file=sys.stderr, flush=True)
for line in sys.stdin:
    q = json.loads(line); print('req', q['id'], file=sys.stderr, flush=True)
    row = q.get("steer_row", -1)
    for i in range(q["n"]):
        if mode == "sweep" and row >= 0:  # lens-token share grows with the scale; saturated scales collapse onto the 32 lens ids
            p = min(1.0, q.get("steer_scale", 0) / 400)
            hit = ((q["id"] * 2654435761 + i * 40503) % 1000) / 1000 < p
            t = int(ids[units[row], (len(q["prompt"]) + i) % k]) if hit else (q["id"] * 7919 + i * 104729 + len(q["prompt"]) * 31) % 150000
        elif mode == "planted" and row >= 0:
            t = int(ids[units[row], (len(q["prompt"]) + i) % k])
        else:
            t = (q["id"] * 7919 + i * 104729 + len(q["prompt"]) * 31) % 150000
        print(json.dumps({"id": q["id"], "tok": t}, separators=(",", ":")), flush=True)
    print(json.dumps({"id": q["id"], "done": True, "n": q["n"]}, separators=(",", ":")), flush=True)
