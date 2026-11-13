"""G3 harness (docs/prereg-implants.md 0604f51 + amendments): implants against held-out mistakes (T1 two-digit addition) and as memorised facts, with a random-implant control.

usage: implants-harness.py WORKDIR STAGE   STAGE: data mine capture select selectk eval-real eval-ctl verdict facts-base facts-cap facts-eval (ARM=real|ctl) facts-verdict
One implants file per family holds every train mistake (all layers, nested rows); request fields implants_layer / implants_k / implants_scale pick the layer, the first k rows and a gate scale, so one
engine process covers a whole sweep (every GPU job here waits 10 to 20 minutes in the queue). Scale per k is chosen on TRAIN mistakes only; the layer on DEV. Stages write files in WORKDIR and are skipped when
their output exists. DRY=1 runs every stage against a stub engine (no GPU) to exercise the code after the GPU calls.
Env: ENGINE PACK GGUF TOKENIZER BUILDER LAYERS SCALES KS POOL. Python is the experiment driver and oracle; the builder under test is src/causal/implants (Mojo).
"""
import json, os, random, subprocess, sys, time, math
import numpy as np
from tokenizers import Tokenizer

W, STAGE = os.path.abspath(sys.argv[1]), sys.argv[2]
HOME = os.path.expanduser("~"); BARO = f"{HOME}/Projects/mojo/mojo-baro-lanes/neuron2b"
ENGINE = os.environ.get("ENGINE", f"{BARO}/.work/engine-9b-g"); PACK = os.environ.get("PACK", f"{HOME}/Projects/mojo/mojo-baro/.work/e11b-qwythos-pack")
GGUF = os.environ.get("GGUF", f"{HOME}/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf")  # the e11b pack source_gguf
TOK = Tokenizer.from_file(os.environ.get("TOKENIZER", f"{HOME}/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf/tokenizer.json"))
BUILDER = os.environ.get("BUILDER", os.path.abspath(".work/implants")); DRY = os.environ.get("DRY") == "1"
LAYERS = [int(x) for x in os.environ.get("LAYERS", "12,16,20,24").split(",")]; KS = [int(x) for x in os.environ.get("KS", "1,10,100,1000").split(",")]
SCALES = [float(x) for x in os.environ.get("SCALES", "0.5,1,2,4,8").split(",")]; POOL = int(os.environ.get("POOL", 4000))
os.makedirs(W, exist_ok=True)
enc = lambda s: TOK.encode(s).ids
dec = lambda ids: TOK.decode(ids)
def jl(p): return [json.loads(l) for l in open(p)]
def dump(p, rows): open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
def jload(p): return json.load(open(p))
def exists(n): return os.path.exists(f"{W}/{n}")


# gpu-wait --shared --vram: measured 9.89 GB (9.2 GiB) for engine-9b-g4 under rocm-smi, so 10 GiB; VRAM_GB overrides (boss rule 2026-11-10: correctness evals co-run).
def gw_argv(d, timeout):
    """gpu-wait argv for one engine job (tools/test-gw-argv.py)."""
    return ["gpu-wait", "run", "--shared", "--vram", os.environ.get("VRAM_GB", "10"), "--timeout", str(timeout), "--", f"{d}/run.sh"]


def engine(name, reqs, env=None, timeout=3000):
    """Run reqs (dicts with unique int ids) through one engine process in one gpu-wait job; returns ({id: {"toks","topk","slens","secs"}}, workdir)."""
    d = f"{W}/run-{name}"; os.makedirs(d, exist_ok=True)
    if len({r["id"] for r in reqs}) != len(reqs): raise SystemExit(f"FAIL engine {name}: duplicate request ids")
    if DRY:
        rnd = random.Random(0); out = {}
        for r in reqs:
            out[r["id"]] = {"toks": [rnd.choice([15, 16, 17, 18])], "topk": {1: 1.0, 2: 0.5}, "slens": {"kl": 1e-6, "dnll": 1e-4, "nll": 2.0, "dld": 0.0} if r.get("slens_k", 0) else None, "secs": 0.06}
        return out, d
    open(f"{d}/req.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in reqs))
    e = {"PATH": os.environ["PATH"], "HOME": HOME, "BARO_SERVE": "1", "BARO_STEER_SERVE": "1", "BARO_SPEC": "0", "BARO_PACK": PACK, **(env or {})}
    open(f"{d}/run.sh", "w").write("#!/usr/bin/env bash\n" + "".join(f"export {k}='{v}'\n" for k, v in e.items()) + f"cd '{d}'\nexec '{ENGINE}' < req.jsonl > out.jsonl 2> err.txt\n")
    os.chmod(f"{d}/run.sh", 0o755)
    t0 = time.time()
    subprocess.run(gw_argv(d, timeout), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out, done = {}, 0
    for l in open(f"{d}/out.jsonl"):
        if not l.startswith('{"id":'): continue
        j = json.loads(l); r = out.setdefault(j["id"], {"toks": [], "topk": None, "slens": None, "secs": 0.0})
        if "error" in j: raise SystemExit(f"FAIL engine {name} request {j['id']}: {j['error']}")
        if "tok" in j: r["toks"].append(j["tok"])
        if "logits_topk" in j: r["topk"] = {e["id"]: e["logit"] for e in j["logits_topk"]}
        if "slens" in j: r["slens"] = j["slens"]
        if j.get("done"): done += 1; r["secs"] = j.get("prefill_s", 0) + j.get("decode_s", 0); r["prefill"] = j.get("prefill_s", 0)
    if done != len(reqs): raise SystemExit(f"FAIL engine {name}: {done} done lines for {len(reqs)} requests, see {d}/err.txt")
    print(f"engine {name}: {len(reqs)} requests, {time.time() - t0:.0f}s wall incl. queue, {sum(r['secs'] for r in out.values()):.0f}s engine", flush=True)
    return out, d


class Ids:
    """Unique request ids with a side table of what each one is."""
    def __init__(self): self.n = 0; self.meta = {}
    def new(self, **m): self.n += 1; self.meta[self.n] = m; return self.n
    def find(self, **m): return next(i for i, v in self.meta.items() if all(v.get(k) == x for k, x in m.items()))

N_ANS = int(os.environ.get("N_ANS", 10))  # answer tokens generated (three-digit products have up to 6 digits)
def req(i, text_or_ids, n=None, **kw): return {"id": i, "prompt": enc(text_or_ids) if isinstance(text_or_ids, str) else text_or_ids, "n": N_ANS if n is None else n, **kw}
def answer_ok(gen, gold): return gen.strip().split("\n")[0].strip().startswith(gold.strip())
def ok_of(out, rid, gold): return answer_ok(dec(out[rid]["toks"]), gold)


# ---- data ----
FAMILY = os.environ.get("FAMILY", "mul3")  # add = T1 (zero mistakes), mul = T1b (28 mistakes), mul3 = T1c (docs/prereg-implants-d.md)
def make_t1():
    """T1b: every two-digit product a * b (8100 distinct pairs), shuffled with seed 1, first POOL. FAMILY=add: T1, the carry additions."""
    rng = random.Random(1)
    if FAMILY == "add":
        allp = [(a, b) for a in range(10, 100) for b in range(10, 100) if (a % 10) + (b % 10) >= 10]; rng.shuffle(allp)
        return [{"q": f"Q: What is {a} + {b}? A:", "gold": str(a + b)} for a, b in allp[:POOL]]
    lo, hi = (100, 1000) if FAMILY == "mul3" else (10, 100)
    seen, rows = set(), []
    while len(rows) < POOL:
        a, b = rng.randint(lo, hi - 1), rng.randint(lo, hi - 1)
        if (a, b) not in seen: seen.add((a, b)); rows.append({"q": f"Q: What is {a} * {b}? A:", "gold": str(a * b)})
    return rows

def make_nc(n=300):
    rng = random.Random(5); rows = []
    while len(rows) < n:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if (a % 10) + (b % 10) < 10: rows.append({"q": f"Q: What is {a} + {b}? A:", "gold": str(a + b)})
    return rows

TRIVIA = [("Q: What is the capital of France? A:", "Paris"), ("Q: What is the chemical symbol for water? A:", "H2O"), ("Q: How many days are in a week? A:", "7"), ("Q: What color is the sky on a clear day? A:", "blue"),
          ("Q: What is the largest planet in the solar system? A:", "Jupiter"), ("Q: Who wrote Hamlet? A:", "Shakespeare"), ("Q: What is the freezing point of water in Celsius? A:", "0"),
          ("Q: What is the capital of Japan? A:", "Tokyo"), ("Q: How many legs does a spider have? A:", "8"), ("Q: What is the opposite of hot? A:", "cold")]

def facts(n=1000):
    rng = random.Random(7); syl = ["zor", "plin", "vex", "kal", "mur", "dren", "tov", "qui", "bar", "nel", "sul", "fim", "gax", "hov", "jin", "lum"]
    nm = lambda: "".join(rng.choice(syl) for _ in range(rng.choice([2, 3]))).capitalize()
    out, seen = [], set()
    while len(out) < n:
        c, t = nm(), nm()
        if c in seen or t in seen or c == t: continue
        seen |= {c, t}
        out.append({"c": c, "t": t, "exact": f"Q: What is the capital of {c}? A:", "para": [f"Q: Name the capital city of {c}. A:", f"Q: Which city is the capital of {c}? A:", f"Q: {c}'s capital is called what? A:"], "gold": " " + t})
    return out

def windows(n=400):
    ids = enc(open(f"{W}/ppl.txt").read())
    return [(ids[i - 24:i], ids[i]) for i in range(24, len(ids) - 1, max(1, (len(ids) - 25) // n))][:n]

def unrel_items(): return [(u["q"], u["gold"]) for u in jl(f"{W}/unrel.jsonl")]


# ---- implants files (docs/format-implants.md) ----
def read_sets(path):
    b = open(path, "rb").read(); assert b[:8] == b"NIMP0001"; H, ns = (int(x) for x in np.frombuffer(b[8:16], dtype=np.uint32)); off = 16; sets = []
    for _ in range(ns):
        layer, k = (int(x) for x in np.frombuffer(b[off:off + 8], dtype=np.uint32)); tau = float(np.frombuffer(b[off + 8:off + 12], dtype=np.float32)[0]); off += 12
        keys = np.frombuffer(b[off:off + 4 * k * H], dtype=np.float32).reshape(k, H); off += 4 * k * H
        gates = np.frombuffer(b[off:off + 4 * k], dtype=np.float32); off += 4 * k
        vals = np.frombuffer(b[off:off + 4 * k * H], dtype=np.float32).reshape(k, H); off += 4 * k * H
        sets.append({"layer": layer, "k": k, "tau": tau, "keys": keys, "gates": gates, "vals": vals})
    assert off == len(b), "trailing bytes"; return H, sets

def write_sets(path, H, sets):
    with open(path, "wb") as f:
        f.write(b"NIMP0001" + np.array([H, len(sets)], dtype=np.uint32).tobytes())
        for s in sets: f.write(np.array([s["layer"], s["k"]], dtype=np.uint32).tobytes() + np.float32(s["tau"]).tobytes() + np.ascontiguousarray(s["keys"], dtype=np.float32).tobytes() + np.ascontiguousarray(s["gates"], dtype=np.float32).tobytes() + np.ascontiguousarray(s["vals"], dtype=np.float32).tobytes())

def control_sets(sets, seed=0):
    """Random implants of equal size: random keys and values with the real keys' and values' norms, the real gates and tau."""
    rng = np.random.default_rng(seed); out = []
    for s in sets:
        rk = rng.standard_normal(s["keys"].shape).astype(np.float32); rk *= np.linalg.norm(s["keys"], axis=1, keepdims=True) / np.linalg.norm(rk, axis=1, keepdims=True)
        rv = rng.standard_normal(s["vals"].shape).astype(np.float32); rv *= np.linalg.norm(s["vals"], axis=1, keepdims=True) / np.linalg.norm(rv, axis=1, keepdims=True)
        out.append({**s, "keys": rk, "vals": rv})
    return out

def pairs_from_capture(capdir, items, outp):
    """items in capture order -> builder PAIRS.tsv; rec = the last record of each request in the capture (its last prompt position when n = 1)."""
    if DRY: open(outp, "w").write("".join(f"{i}\t{it['right']}\t{it['wrong']}\t{it['gap']}\n" for i, it in enumerate(items))); return
    r = np.fromfile(f"{capdir}/req.i32", dtype=np.int32); ends = [i for i in range(len(r)) if i + 1 == len(r) or r[i + 1] != r[i]]
    m = len(ends) // max(1, len(items))  # mojo-baro main engine writes the whole capture once per captured layer (identical passes, 2026-11-13): key on pass 1
    if m > 1 and len(ends) == m * len(items) and all((r[:ends[len(items) - 1] + 1] == r[(ends[len(items) - 1] + 1) * p:(ends[len(items) - 1] + 1) * (p + 1)]).all() for p in range(1, m)): ends = ends[:len(items)]
    if len(ends) != len(items): raise SystemExit(f"FAIL capture: {len(ends)} requests in capture, {len(items)} items")
    open(outp, "w").write("".join(f"{ends[i]}\t{it['right']}\t{it['wrong']}\t{it['gap']}\n" for i, it in enumerate(items)))

def build_full(items, cd, layers, kk, name):
    """Run the Mojo builder once per layer (first kk items, scale 1), merge into one multi-layer file W/imp-NAME.bin; returns (path, builder seconds)."""
    pp = f"{W}/pairs-{name}.tsv"; pairs_from_capture(cd, items, pp)
    lines = [l for l in open(pp).read().split("\n") if l][:kk]; open(pp, "w").write("\n".join(lines) + "\n")
    sets, secs, H = [], 0.0, 4096
    for L in layers:
        single = f"{W}/imp-{name}-L{L}.bin"; t0 = time.time()
        if DRY:
            sets.append({"layer": L, "k": len(lines), "tau": 0.95, "keys": np.random.randn(len(lines), H).astype(np.float32), "gates": np.ones(len(lines), np.float32), "vals": np.random.randn(len(lines), H).astype(np.float32)}); continue
        r = subprocess.run([BUILDER, GGUF, cd, str(L), pp, single, "scale=1"], capture_output=True, text=True, env={**os.environ, "NX_RLOG": os.environ.get("NX_RLOG", f"{W}/rlog")}); secs += time.time() - t0
        if r.returncode: raise SystemExit(f"FAIL builder L{L}: {r.stdout[-300:]} {r.stderr[-300:]}")
        H, s1 = read_sets(single); sets += s1
    for s in sets:  # file integrity before any GPU time is spent on it
        assert s["k"] == len(lines) and np.isfinite(s["keys"]).all() and np.isfinite(s["vals"]).all() and np.isfinite(s["gates"]).all() and (s["gates"] > 0).all(), f"bad set at layer {s['layer']}"
    out = f"{W}/imp-{name}.bin"; write_sets(out, H, sets); return out, secs


def capture(name, texts, golds, layers):
    """Run texts through the engine with A1 capture of field x at the layers; right = first token of gold, wrong = model's top-1, gap from the top-20 (the right token's logit
    when it is in the top 20, else the 20th logit: a proxy the scale sweep absorbs). Returns items, capture dir, engine seconds."""
    cd = f"{W}/cap-{name}"; os.makedirs(cd, exist_ok=True)
    out, d = engine(f"cap-{name}", [req(i + 1, t, n=1, logits_topk=20) for i, t in enumerate(texts)], {"BARO_CAPTURE": cd, "BARO_CAPTURE_LAYERS": ",".join(map(str, layers)), "BARO_CAPTURE_FIELDS": "x"})
    secs = sum(r["secs"] for r in out.values()); items = []
    for i, g in enumerate(golds):
        tk = out[i + 1]["topk"]; right = enc(g)[0]; top = max(tk, key=tk.get)
        items.append({"right": right, "wrong": top, "gap": 0.0 if top == right else tk[top] - tk.get(right, min(tk.values()))})
    return items, cd, secs

def capture_ids(name, prompts, rights, wrongs, layers):
    """A1 capture of field x at the layers, one request (n = 1) per prompt id list; the LAST position of each request is the one keyed. right/wrong = token ids; gap = engine top-1 logit minus the right token's logit
    (the right token's logit when it is in the top 20, else the 20th logit: a proxy the scale sweep absorbs). Returns items, capture dir, engine seconds."""
    cd = f"{W}/cap-{name}"; os.makedirs(cd, exist_ok=True)
    out, d = engine(f"cap-{name}", [req(i + 1, p, n=1, logits_topk=20) for i, p in enumerate(prompts)], {"BARO_CAPTURE": cd, "BARO_CAPTURE_LAYERS": ",".join(map(str, layers)), "BARO_CAPTURE_FIELDS": "x"})
    secs = sum(r["secs"] for r in out.values()); items = []
    for i, (r_, w_) in enumerate(zip(rights, wrongs)):
        tk = out[i + 1]["topk"]; top = max(tk, key=tk.get)
        if w_ is None: w_ = top if top != r_ else sorted(tk, key=tk.get)[-2]  # facts: the model's own top token is the wrong one
        items.append({"right": r_, "wrong": w_, "gap": max(0.0, tk[top] - tk.get(r_, min(tk.values())))})
    return items, cd, secs

def imp_kw(L, k, s): return {"implants": 1, "implants_layer": L, "implants_k": k, "implants_scale": s}
def ppl_reqs(ids, win, cfgs):
    """slens requests: base contexts with implants off, then for every config the same windows with implants on; the last window's line carries that config's mean KL and dNLL."""
    rs, last = [], {}
    for i, (c, t) in enumerate(win): rs.append({"id": ids.new(kind="pplbase"), "prompt": c, "n": 1, "implants": 0, "slens_ctx": i, "slens_k": 0})
    for cfg in cfgs:
        for i, (c, t) in enumerate(win):
            rid = ids.new(kind="ppl", cfg=list(cfg)); rs.append({"id": rid, "prompt": c, "n": 1, **imp_kw(*cfg), "slens_ctx": i, "slens_k": 1, "slens_tgt": t, "slens_flags": (1 if i == 0 else 0) + (2 if i == len(win) - 1 else 0)})
            if i == len(win) - 1: last[tuple(cfg)] = rid
    return rs, last

def pick_scale(fit):
    """fit: {scale: fixed fraction on train}; smallest scale reaching 0.95, else the best fraction (smallest scale on ties)."""
    for s in sorted(fit):
        if fit[s] >= 0.95: return s
    m = max(fit.values()); return min(s for s in fit if fit[s] == m)

def diff_info(m):
    """(gold token ids, model token ids, index of the first wrong token or None): an implant is keyed at the position that predicts that token (prompt + the correct prefix)."""
    gi, go = enc(" " + m["gold"]), enc(" " + m["got"])
    d = next((i for i in range(min(len(gi), len(go))) if gi[i] != go[i]), None)
    return gi, go, d

def t1_sets():
    mis = [m for m in jl(f"{W}/mistakes.jsonl") if diff_info(m)[2] is not None]  # a mistake with no differing token (output cut short) cannot be keyed
    return [m for m in mis if m["split"] == "train"], [m for m in mis if m["split"] == "dev"], [m for m in mis if m["split"] == "test"]


# ---- stages ----
if STAGE == "data":
    if not exists("t1.jsonl"): dump(f"{W}/t1.jsonl", make_t1())
    if not exists("nc.jsonl") and not exists("unrel.jsonl"): dump(f"{W}/nc.jsonl", make_nc())
    if not exists("ppl.txt"): open(f"{W}/ppl.txt", "w").write(open(f"{HOME}/Projects/mojo/mojo-neuron/docs/plans/2026-11-09-neuron-v2.md").read()[:9000])
    print("data:", len(jl(f"{W}/t1.jsonl")), "t1 problems")

if STAGE == "mine" and not exists("mistakes.jsonl"):
    pool = jl(f"{W}/t1.jsonl"); have_unrel = exists("unrel.jsonl"); nc = [] if have_unrel else jl(f"{W}/nc.jsonl"); tv = [] if have_unrel else [{"q": q, "gold": g} for q, g in TRIVIA]; allp = pool + nc + tv
    out, _ = engine("mine", [req(i + 1, p["q"]) for i, p in enumerate(allp)], timeout=3600)
    right = lambda i: answer_ok(dec(out[i + 1]["toks"]), allp[i]["gold"])
    mis = [dict(p, got=dec(out[i + 1]["toks"]).strip().split("\n")[0]) for i, p in enumerate(pool) if not right(i)]
    unrel = [allp[i] for i in range(len(pool), len(allp)) if right(i)]; unrel = [u for u in unrel if u in tv] + [u for u in unrel if u not in tv]
    if not have_unrel: dump(f"{W}/unrel.jsonl", unrel[:100]); print("unrelated items the model answers right:", len(unrel[:100]))
    random.Random(2).shuffle(mis); n = len(mis)
    for j, m in enumerate(mis): m["split"] = "train" if j < n * 0.5 else ("dev" if j < n * 0.7 else "test")
    dump(f"{W}/mistakes.jsonl", mis); print(f"mine: {len(mis)} mistakes of {len(pool)} ({100 * len(mis) / len(pool):.1f} %), train/dev/test {[sum(m['split'] == s for m in mis) for s in ('train', 'dev', 'test')]}")

if STAGE == "capture" and not exists("t1-train.json"):
    tr, dv, te = t1_sets()
    if len(te) < 30: print(f"UNDERPOWERED: {len(tr)}/{len(dv)}/{len(te)} train/dev/test mistakes, fewer than 30 test mistakes: no verdict for this family (docs/prereg-implants.md); stopping before any GPU time is spent"); sys.exit(3)
    kmax = min(max(KS), len(tr)); pr, rt, wr = [], [], []
    for m in tr[:kmax]:
        gi, go, d = diff_info(m); pr.append(enc(m["q"]) + gi[:d]); rt.append(gi[d]); wr.append(go[d])
    items, cd, secs = capture_ids("t1", pr, rt, wr, LAYERS)
    path, bsecs = build_full(items, cd, LAYERS, kmax, "t1")
    json.dump({"items": items, "capdir": cd, "cap_secs": secs, "build_secs": bsecs, "file": path, "kmax": kmax}, open(f"{W}/t1-train.json", "w"))
    H, sets = read_sets(path); write_sets(f"{W}/ctl-t1.bin", H, control_sets(sets)); print(f"capture: {len(items)} train mistakes, engine {secs:.0f}s, builder {bsecs:.0f}s, file {path}")

if STAGE == "select" and not exists("select.json"):  # layer on DEV at k = 100: layers x scales, fit on the first 100 train mistakes (one job)
    tr, dv, te = t1_sets(); info = jload(f"{W}/t1-train.json"); ids = Ids(); rs = []; ntr = info["kmax"]; k100 = 100 if 100 <= ntr else ntr
    for L in LAYERS:
        for s in SCALES:
            for i in range(k100): rs.append(req(ids.new(kind="fit", L=L, s=s, i=i), tr[i]["q"], **imp_kw(L, k100, s)))
            for i, m in enumerate(dv[:100]): rs.append(req(ids.new(kind="dev", L=L, s=s, i=i), m["q"], **imp_kw(L, k100, s)))
    out, _ = engine("select", rs, {"BARO_IMPLANTS": info["file"]}, timeout=5400)
    fit, devacc = {}, {}
    for rid, m in ids.meta.items():
        if m["kind"] == "fit": fit.setdefault((m["L"], m["s"]), []).append(ok_of(out, rid, tr[m["i"]]["gold"]))
        if m["kind"] == "dev": devacc.setdefault((m["L"], m["s"]), []).append(ok_of(out, rid, dv[m["i"]]["gold"]))
    sel = {"layers": {}}
    for L in LAYERS:
        f100 = {s: float(np.mean(fit[(L, s)])) for s in SCALES}; s100 = pick_scale(f100)
        sel["layers"][str(L)] = {"scale100": s100, "train_fixed100": f100[s100], "dev_fixed": float(np.mean(devacc[(L, s100)])), "dev_n": len(devacc[(L, s100)]), "fit_by_scale": f100}
    sel["layer"] = sorted(LAYERS, key=lambda L: (-sel["layers"][str(L)]["dev_fixed"], -sel["layers"][str(L)]["train_fixed100"], L))[0]
    json.dump(sel, open(f"{W}/select.json", "w"), indent=1); print(json.dumps({k: (v["scale100"], round(v["train_fixed100"], 3), round(v["dev_fixed"], 3)) for k, v in sel["layers"].items()}), "-> layer", sel["layer"])

if STAGE == "selectk" and "per_k" not in (jload(f"{W}/select.json") if exists("select.json") else {}):  # scale per k on TRAIN at the chosen layer (one job)
    tr, dv, te = t1_sets(); info = jload(f"{W}/t1-train.json"); sel = jload(f"{W}/select.json"); L = sel["layer"]; ids = Ids(); rs = []; ntr = info["kmax"]
    ks = sorted(set([k for k in KS if k <= ntr] + [ntr]))
    for s in SCALES:
        for k in ks:
            for i in range(min(k, ntr)): rs.append(req(ids.new(kind="fit", s=s, k=k, i=i), tr[i]["q"], **imp_kw(L, k, s)))
    out, _ = engine("selectk", rs, {"BARO_IMPLANTS": info["file"]}, timeout=5400); fit = {}
    for rid, m in ids.meta.items(): fit.setdefault((m["k"], m["s"]), []).append(ok_of(out, rid, tr[m["i"]]["gold"]))
    sel["ks"] = ks; sel["per_k"] = {}
    for k in ks:
        f = {s: float(np.mean(fit[(k, s)])) for s in SCALES}; s = pick_scale(f); sel["per_k"][str(k)] = {"scale": s, "train_fixed": f[s], "fit_by_scale": f}
    json.dump(sel, open(f"{W}/select.json", "w"), indent=1); print(json.dumps({k: (v["scale"], round(v["train_fixed"], 3)) for k, v in sel["per_k"].items()}))

if STAGE in ("eval-real", "eval-ctl"):
    name = "real" if STAGE == "eval-real" else "ctl"
    if not exists(f"results-{name}.json"):
        tr, dv, te = t1_sets(); info = jload(f"{W}/t1-train.json"); sel = jload(f"{W}/select.json"); L = sel["layer"]; ids = Ids(); rs = []; unrel = unrel_items(); win = windows()
        cfgs = [(L, int(k), v["scale"]) for k, v in sel["per_k"].items()]
        for (L_, k, s) in cfgs:
            for i in range(min(k, info["kmax"])): rs.append(req(ids.new(kind="fit", k=k, i=i), tr[i]["q"], **imp_kw(L_, k, s)))
            for i, m in enumerate(te): rs.append(req(ids.new(kind="test", k=k, i=i), m["q"], **imp_kw(L_, k, s)))
            for i, (q, g) in enumerate(unrel): rs.append(req(ids.new(kind="unrel_on", k=k, i=i), q, **imp_kw(L_, k, s)))
        for i, (q, g) in enumerate(unrel): rs.append(req(ids.new(kind="unrel_off", i=i), q, implants=0))
        pr, last = ppl_reqs(ids, win, cfgs); rs += pr
        out, _ = engine(f"eval-{name}", rs, {"BARO_IMPLANTS": info["file"] if name == "real" else f"{W}/ctl-t1.bin"}, timeout=5400)
        by = lambda kind, **f: [(rid, m) for rid, m in ids.meta.items() if m["kind"] == kind and all(m.get(a) == b for a, b in f.items())]
        res = {}
        for (L_, k, s) in cfgs:
            fit = [ok_of(out, rid, tr[m["i"]]["gold"]) for rid, m in by("fit", k=k)]; tst = [ok_of(out, rid, te[m["i"]]["gold"]) for rid, m in by("test", k=k)]
            uon = [ok_of(out, rid, unrel[m["i"]][1]) for rid, m in by("unrel_on", k=k)]; uoff = [ok_of(out, rid, unrel[m["i"]][1]) for rid, m in by("unrel_off")]; sl = out[last[(L_, k, s)]]["slens"]
            res[str(k)] = {"k": min(k, info["kmax"]), "layer": L_, "scale": s, "train_fixed": float(np.mean(fit)), "n_test": len(tst), "test_fixed": float(np.mean(tst)), "test_fixed_n": int(sum(tst)),
                           "unrel_on": float(np.mean(uon)), "unrel_off": float(np.mean(uoff)), "dnll": sl["dnll"], "ppl_change": math.exp(sl["dnll"]) - 1, "kl": sl["kl"]}
        json.dump(res, open(f"{W}/results-{name}.json", "w"), indent=1); print(json.dumps(res, indent=1))

if STAGE == "verdict":
    sel = jload(f"{W}/select.json"); R, C = jload(f"{W}/results-real.json"), jload(f"{W}/results-ctl.json"); tr, dv, te = t1_sets(); lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))
    P("## T1 two-digit addition with a carry: generalisation (prereg docs/prereg-implants.md, G-A to G-D)\n")
    P(f"layer {sel['layer']} (chosen on dev, k=100), train/dev/test mistakes {len(tr)}/{len(dv)}/{len(te)}; the unmodified model is wrong on every one of them, so the gain is the test accuracy with implants.\n")
    P("| k | scale | train fixed (G-D) | test fixed, implants | test fixed, control | gain over control | sign-test p | unrelated acc drop | ppl change | control unrelated drop | control ppl |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    passk = []
    for k in sorted(R, key=int):
        r, c = R[k], C[k]; gain = r["test_fixed"]; gc = c["test_fixed"]; p = 0.5 ** r["test_fixed_n"]; du = r["unrel_off"] - r["unrel_on"]; duc = c["unrel_off"] - c["unrel_on"]
        A = gain >= 0.10 and p < 0.05; B = gain - gc >= 0.10 and abs(gc) <= 0.03; Cb = du <= 0.01 and r["ppl_change"] <= 0.01
        P(f"| {r['k']} | {r['scale']} | {r['train_fixed']:.3f} | {gain:.3f} ({r['test_fixed_n']}/{r['n_test']}) | {gc:.3f} | {gain - gc:+.3f} | {p:.2g} | {du:+.3f} | {r['ppl_change']:+.4f} | {duc:+.3f} | {c['ppl_change']:+.4f} |")
        if A and B and Cb: passk.append(str(r["k"]))
    kmax = max(R, key=int); D = R[kmax]["train_fixed"] >= 0.90
    P(f"\nG-D (builder works, train mistakes fixed at k = {R[kmax]['k']}): {R[kmax]['train_fixed']:.3f} -> {'PASS' if D else 'MISSED'}")
    P("G-A (test gain >= +10 pp, p < 0.05), G-B (gain minus control >= +10 pp, control within +-3 pp), G-C (unrelated drop <= 1 pp, ppl <= +1 %) are read per row above; a family passes only if all three hold at the same k.")
    P(f"T1 verdict: {('PASS at k = ' + ','.join(passk)) if passk and D else ('builder failed (G-D), G-A to G-C not interpretable' if not D else 'MISSED: no k satisfies G-A, G-B and G-C together')}")
    P(f"Underpowered rule: fewer than 30 test mistakes gets no G-A verdict; here n_test = {len(te)} -> {'UNDERPOWERED' if len(te) < 30 else 'ok'}.")
    open(f"{W}/verdict-t1.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

# ---- facts (verdict 2) ----
if STAGE == "facts-base" and not exists("facts-known.json"):
    fs = facts(); out, _ = engine("facts-base", [req(i + 1, f["exact"]) for i, f in enumerate(fs)])
    unknown = [i for i, f in enumerate(fs) if not answer_ok(dec(out[i + 1]["toks"]), f["t"])]
    json.dump({"unknown": unknown}, open(f"{W}/facts-known.json", "w")); print(f"facts: {len(unknown)} of {len(fs)} unknown to the model")

if STAGE == "facts-cap" and not exists("facts-train.json"):
    fs = facts(); unk = jload(f"{W}/facts-known.json")["unknown"]; sub = [fs[i] for i in unk][:1000]; L = jload(f"{W}/select.json")["layer"]
    pr, rt, wr, cum = [], [], [], [0]  # one request per (fact, answer token): keyed at the position that predicts that token; the model's own token is the wrong one (it does not know the fact)
    for f in sub:
        gi = enc(f["gold"])
        for j in range(len(gi)): pr.append(enc(f["exact"]) + gi[:j]); rt.append(gi[j]); wr.append(-1)
        cum.append(len(pr))
    items, cd, secs = capture_ids("facts", pr, rt, [None] * len(pr), [L])
    out_top = None
    path, bsecs = build_full(items, cd, [L], len(items), "facts")
    json.dump({"file": path, "cap_secs": secs, "build_secs": bsecs, "n": len(sub), "layer": L, "rows_cum": cum}, open(f"{W}/facts-train.json", "w"))
    H, sets = read_sets(path); write_sets(f"{W}/ctl-facts.bin", H, control_sets(sets)); print(f"facts capture: {len(sub)} facts, {len(items)} rows, engine {secs:.0f}s, builder {bsecs:.0f}s")

if STAGE == "facts-eval":
    name = os.environ.get("ARM", "real")
    if not exists(f"facts-results-{name}.json"):
        fs = facts(); unk = jload(f"{W}/facts-known.json")["unknown"]; info = jload(f"{W}/facts-train.json"); sub = [fs[i] for i in unk][:info["n"]]; L = info["layer"]
        ids = Ids(); rs = []; unrel = unrel_items(); win = windows(); ks = sorted(set([k for k in KS if k <= len(sub)] + [len(sub)])); k100 = 100 if 100 <= len(sub) else len(sub)
        for k in ks:
            for s in SCALES:
                for i in range(k): rs.append(req(ids.new(kind="exact", k=k, s=s, i=i), sub[i]["exact"], **imp_kw(L, info["rows_cum"][k], s)))
        for s in SCALES:
            for i in range(k100):
                for pi, ptxt in enumerate(sub[i]["para"]): rs.append(req(ids.new(kind="para", s=s, i=i, pi=pi), ptxt, **imp_kw(L, info["rows_cum"][k100], s)))
            for i, (q, g) in enumerate(unrel): rs.append(req(ids.new(kind="unrel_on", s=s, i=i), q, **imp_kw(L, info["rows_cum"][k100], s)))
        for i, (q, g) in enumerate(unrel): rs.append(req(ids.new(kind="unrel_off", i=i), q, implants=0))
        pr, last = ppl_reqs(ids, win, [(L, info["rows_cum"][k100], s) for s in SCALES]); rs += pr
        out, _ = engine(f"facts-eval-{name}", rs, {"BARO_IMPLANTS": info["file"] if name == "real" else f"{W}/ctl-facts.bin"}, timeout=5400)
        by = lambda kind, **f: [(rid, m) for rid, m in ids.meta.items() if m["kind"] == kind and all(m.get(a) == b for a, b in f.items())]
        res = {"k100": k100, "per_k": {}, "scales": {}}
        for k in ks:
            ex = {s: float(np.mean([ok_of(out, rid, sub[m["i"]]["t"]) for rid, m in by("exact", k=k, s=s)])) for s in SCALES}; s = pick_scale(ex); res["per_k"][str(k)] = {"scale": s, "exact": ex[s], "exact_by_scale": ex}
        for s in SCALES:
            res["scales"][str(s)] = {"para": float(np.mean([ok_of(out, rid, sub[m["i"]]["t"]) for rid, m in by("para", s=s)])), "unrel_on": float(np.mean([ok_of(out, rid, unrel[m["i"]][1]) for rid, m in by("unrel_on", s=s)])),
                                    "unrel_off": float(np.mean([ok_of(out, rid, unrel[m["i"]][1]) for rid, m in by("unrel_off")])), "ppl_change": math.exp(out[last[(L, info["rows_cum"][k100], s)]]["slens"]["dnll"]) - 1}
        json.dump(res, open(f"{W}/facts-results-{name}.json", "w"), indent=1); print(json.dumps(res, indent=1))

if STAGE == "facts-rank" and not exists("facts-rank.json"):  # docs/prereg-facts-rank.md: gold first-token NLL and top-20 rank at scale 0/8/32 on 20 facts
    fs = facts(); unk = jload(f"{W}/facts-known.json")["unknown"]; info = jload(f"{W}/facts-train.json"); sub = [fs[i] for i in unk][:20]; L = info["layer"]; kk = info["rows_cum"][100]
    RS = [0.0, 8.0, 32.0]; ids = Ids(); rs = []; gold = [enc(f["gold"])[0] for f in sub]
    for i, f in enumerate(sub): rs.append({"id": ids.new(kind="base", i=i), "prompt": enc(f["exact"]), "n": 1, "implants": 0, "slens_ctx": i, "slens_k": 0})
    for s_ in RS:
        for i, f in enumerate(sub):
            kw = {"implants": 0} if s_ == 0 else imp_kw(L, kk, s_)
            rs.append({"id": ids.new(kind="nll", i=i, s=s_), "prompt": enc(f["exact"]), "n": 1, **kw, "slens_ctx": i, "slens_k": 1, "slens_tgt": gold[i], "slens_flags": 3})
            rs.append({"id": ids.new(kind="top", i=i, s=s_), "prompt": enc(f["exact"]), "n": 1, "logits_topk": 20, **kw})
    out, _ = engine("facts-rank", rs, {"BARO_IMPLANTS": info["file"]}, timeout=900)
    rows = []
    for rid, m in ids.meta.items():
        if m["kind"] != "nll": continue
        o = out[rid]["slens"]; tk = out[ids.find(kind="top", i=m["i"], s=m["s"])]["topk"]; srt = sorted(tk, key=tk.get, reverse=True); g = gold[m["i"]]
        rows.append({"i": m["i"], "scale": m["s"], "nll": o["nll"], "dnll": o["dnll"], "rank": srt.index(g) + 1 if g in tk else None, "gap": tk[srt[0]] - tk[g] if g in tk else tk[srt[0]] - tk[srt[-1]], "gap_lower_bound": g not in tk})
    json.dump(rows, open(f"{W}/facts-rank.json", "w"), indent=1)
    base0 = [r for r in rows if r["scale"] == 0.0]; sane = max(abs(r["dnll"]) for r in base0)
    lines = ["## Facts rank falsifier (prereg docs/prereg-facts-rank.md, 20 facts, layer 16, k = 100)", "", "| scale | gold in top 20 (of 20) | median gold NLL | median gap (lower bound when absent) |", "|---|---|---|---|"]
    stats = {}
    for s_ in RS:
        rr = [r for r in rows if r["scale"] == s_]; n20 = sum(1 for r in rr if r["rank"] is not None); med = float(np.median([r["nll"] for r in rr])); stats[s_] = (n20, med)
        lines.append(f"| {s_:g} | {n20} | {med:.2f} | {float(np.median([r['gap'] for r in rr])):.2f} |")
    r1 = {s_: stats[s_][0] - stats[0.0][0] >= 3 for s_ in (8.0, 32.0)}; r2 = {s_: stats[0.0][1] - stats[s_][1] >= 2.0 for s_ in (8.0, 32.0)}
    lines += ["", f"sanity: max |dnll| at scale 0 = {sane:.2e} (must be 0)", f"R1 (top-20 count up by 3 or more): scale 8 {r1[8.0]}, scale 32 {r1[32.0]}", f"R2 (median NLL down by 2.0 nats or more): scale 8 {r2[8.0]}, scale 32 {r2[32.0]}",
              "RESULT: " + ("VOID, scale-0 dnll is not 0" if sane > 1e-6 else "VOID, implants inert: no fact changed its gold NLL between scale 0 and 32" if not any(abs(r["nll"] - z["nll"]) > 1e-6 for r in rows if r["scale"] == 32.0 for z in base0 if z["i"] == r["i"]) else "MOVES, next step to the boss" if any(r1.values()) or any(r2.values()) else "NULL, facts line closed")]
    open(f"{W}/facts-rank.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

if STAGE == "facts-rank2" and not exists("facts-rank2.json"):  # docs/prereg-facts-rank2.md: gold first token rank 1 / top 5 counts at scale 8..24
    fs = facts(); unk = jload(f"{W}/facts-known.json")["unknown"]; info = jload(f"{W}/facts-train.json"); sub = [fs[i] for i in unk][:20]; L = info["layer"]; kk = info["rows_cum"][100]
    RS = [0.0, 8.0, 12.0, 16.0, 20.0, 24.0]; ids = Ids(); rs = []; gold = [enc(f["gold"])[0] for f in sub]
    for s_ in RS:
        for i, f in enumerate(sub): rs.append({"id": ids.new(kind="top", i=i, s=s_), "prompt": enc(f["exact"]), "n": 1, "logits_topk": 20, **({"implants": 0} if s_ == 0 else imp_kw(L, kk, s_))})
    out, _ = engine("facts-rank2", rs, {"BARO_IMPLANTS": info["file"]}, timeout=900)
    res = {}; top1 = {}
    for rid, m in ids.meta.items():
        tk = out[rid]["topk"]; srt = sorted(tk, key=tk.get, reverse=True); g = gold[m["i"]]; top1[(m["s"], m["i"])] = srt[0]
        r = res.setdefault(m["s"], {"rank1": 0, "top5": 0, "top20": 0}); r["rank1"] += srt[0] == g; r["top5"] += g in srt[:5]; r["top20"] += g in tk
    inert = all(top1[(24.0, i)] == top1[(0.0, i)] for i in range(20)); best = max(r["rank1"] for r in res.values())
    lines = ["## Facts rank 2 (prereg docs/prereg-facts-rank2.md, 20 facts, layer 16, k = 100)", "", "| scale | gold rank 1 (of 20) | in top 5 | in top 20 |", "|---|---|---|---|"]
    lines += [f"| {s_:g} | {res[s_]['rank1']} | {res[s_]['top5']} | {res[s_]['top20']} |" for s_ in RS]
    lines += ["", "RK-A (rank 1 for 10 or more of 20 at some scale in 8..24): " + ("VOID, implants inert (scale-24 top-1 equals scale-0 top-1 for all 20)" if inert else f"{'PASS' if best >= 10 else 'MISSED'} (best {best}/20)"), "RK-B top-5 counts are in the table, no bar."]
    json.dump({"res": {str(k): v for k, v in res.items()}, "inert": inert}, open(f"{W}/facts-rank2.json", "w"), indent=1); open(f"{W}/facts-rank2.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

if STAGE == "facts-late" and not exists("facts-late.json"):  # docs/prereg-facts-late.md: same builder, last layers, k = 20, real vs control
    fs = facts(); unk = jload(f"{W}/facts-known.json")["unknown"]; sub = [fs[i] for i in unk][:20]; LL = [28, 30, 31]; SC = [0.5, 1.0, 2.0, 4.0, 8.0]
    gold = [enc(f["gold"])[0] for f in sub]; triv = [enc(q) for q, _ in TRIVIA]
    items, cd, _ = capture_ids("late", [enc(f["exact"]) for f in sub], gold, [None] * 20, LL)
    real, _ = build_full(items, cd, LL, 20, "late"); H, sets = read_sets(real); ctl = f"{W}/ctl-late.bin"; write_sets(ctl, H, control_sets(sets))
    res = {}
    for arm, path in (("real", real), ("ctl", ctl)):
        ids = Ids(); rs = []
        for p_i, p in enumerate([enc(f["exact"]) for f in sub] + triv):
            rs.append({"id": ids.new(p=p_i, L=0, s=0.0), "prompt": p, "n": 1, "logits_topk": 20, "implants": 0})
            for L in LL:
                for s_ in SC: rs.append({"id": ids.new(p=p_i, L=L, s=s_), "prompt": p, "n": 1, "logits_topk": 20, **imp_kw(L, 20, s_)})
        out, _ = engine(f"late-{arm}", rs, {"BARO_IMPLANTS": path}, timeout=900)
        top = {(m["p"], m["L"], m["s"]): max(out[rid]["topk"], key=out[rid]["topk"].get) for rid, m in ids.meta.items()}
        res[arm] = {f"{L}/{s_:g}": {"rank1": sum(top[(i, L, s_)] == gold[i] for i in range(20)), "moved": sum(top[(i, L, s_)] != top[(i, 0, 0.0)] for i in range(20)),
                                    "triv_kept": sum(top[(20 + j, L, s_)] == top[(20 + j, 0, 0.0)] for j in range(len(triv)))} for L in LL for s_ in SC}
        res[arm]["base_rank1"] = sum(top[(i, 0, 0.0)] == gold[i] for i in range(20))
    R, C = res["real"], res["ctl"]; cells = [c for c in R if "/" in c]; best = max(cells, key=lambda c: (R[c]["rank1"], R[c]["triv_kept"]))
    inert = all(R[c]["moved"] == 0 for c in cells)
    lines = ["## Facts late layer (prereg docs/prereg-facts-late.md, 20 facts, k = 20)", "", f"implants off: gold rank 1 for {R['base_rank1']}/20", "",
             "| layer/scale | real rank 1 | real moved | control rank 1 | trivia kept (of 10) real |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {R[c]['rank1']} | {R[c]['moved']} | {C[c]['rank1']} | {R[c]['triv_kept']} |" for c in cells]
    a = R[best]["rank1"] >= 16; b = C[best]["rank1"] <= 2
    lines += ["", "VOID, implants inert" if inert else f"LL-A (rank 1 >= 16/20): {'PASS' if a else 'MISSED'} (best {R[best]['rank1']}/20 at {best})",
              f"LL-B (control rank 1 <= 2 at {best}): {'PASS' if b else 'MISSED'} ({C[best]['rank1']}/20)", f"LL-C trivia kept at {best}: {R[best]['triv_kept']}/10"]
    json.dump(res, open(f"{W}/facts-late.json", "w"), indent=1); open(f"{W}/facts-late.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

if STAGE == "facts-verdict":
    R, C = jload(f"{W}/facts-results-real.json"), jload(f"{W}/facts-results-ctl.json"); info = jload(f"{W}/facts-train.json"); k100 = str(R["k100"]); s = str(R["per_k"][k100]["scale"]); r, c = R["scales"][s], C["scales"][s]
    ex, ex_c = R["per_k"][k100]["exact"], C["per_k"][k100]["exact_by_scale"][s]; secs_pf = (info["cap_secs"] + info["build_secs"]) / info["n"]
    lines = ["## Memorisation as cheap knowledge injection (prereg verdict 2)\n", "| k | scale | exact recall |", "|---|---|---|"] + [f"| {k} | {v['scale']} | {v['exact']:.3f} |" for k, v in sorted(R["per_k"].items(), key=lambda x: int(x[0]))]
    du = r["unrel_off"] - r["unrel_on"]; duc = c["unrel_off"] - c["unrel_on"]
    lines += ["", f"M-A capacity at k={k100} (scale {s}): exact {ex:.3f} (bar >= 0.90 -> {'PASS' if ex >= 0.9 else 'MISSED'}), paraphrase mean {r['para']:.3f} (bar >= 0.50 -> {'PASS' if r['para'] >= 0.5 else 'MISSED'}); k=1000 paraphrase NOT RUN (reported without a bar in the prereg, skipped for GPU time)",
              f"M-B interference at k={k100}: unrelated accuracy drop {du:+.3f} (bar <= 0.01), ppl change {r['ppl_change']:+.4f} (bar <= 0.01), control drop {duc:+.3f} / ppl {c['ppl_change']:+.4f} -> {'PASS' if du <= 0.01 and r['ppl_change'] <= 0.01 and abs(duc) <= abs(du) + 0.01 else 'MISSED'}",
              f"M-C cost: {secs_pf:.3f} s per fact (engine capture {info['cap_secs']:.1f} s + builder {info['build_secs']:.1f} s over {info['n']} facts) -> {'PASS' if secs_pf <= 5 else 'MISSED'} against 5 s; LoRA step time: UNMEASURED unless the report cites a train log", f"control exact recall at the same k and scale: {ex_c:.3f}"]
    open(f"{W}/verdict-facts.md", "w").write("\n".join(lines) + "\n" + "".join("\n" + open(f"{W}/{n}").read() for n in ("facts-rank.md", "facts-rank2.md") if exists(n))); print("\n".join(lines))
