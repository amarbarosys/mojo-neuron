"""G4 harness (docs/prereg-g4-tasks.md): prompt -> implant compiler (context distillation) for three skills, with random-implant, steering-vector and shortened-prompt baselines.

usage: g4-harness.py WORKDIR STAGE    STAGE: data gen cap dev test verdict     (same WORKDIR as implants-harness.py: it reads select.json for the layer)
Four GPU jobs: gen (the with-prompt outputs of the train tasks, base conditions on dev and test), cap (teacher-forced with/without runs, A1 capture of x and r at the layer), dev (scale per skill on dev),
test (real, control, vector on test + unrelated + perplexity windows). DRY=1 runs every stage on a stub engine. Helpers (engine runner, implants file I/O, unrelated set, windows) come from implants-harness.py.
"""
import importlib.util, json, os, random, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); W0, ST = os.path.abspath(sys.argv[1]), sys.argv[2]
sys.argv = [sys.argv[0], W0, "none"]
_spec = importlib.util.spec_from_file_location("ih", f"{HERE}/implants-harness.py"); ih = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ih)
W, DRY = ih.W, ih.DRY
MOJO = os.environ.get("MOJO", f"{os.path.expanduser('~')}/Projects/mojo/mojo-gguf/.venv/bin/mojo"); TAU = 0.9
ISC = [1.0, 3.0, 9.0]; VSC = [0.5, 1.0, 2.0]
EM = chr(0x2014)  # the character the style skill forbids (written as a code point so this source carries none)
e2 = lambda s: ih.TOK.encode(s, add_special_tokens=False).ids

NAMES = ["Maria", "John", "Aiko", "Omar", "Lena", "Pedro", "Ines", "Tariq", "Sofia", "Kenji", "Nadia", "Luca", "Freya", "Ravi", "Chloe", "Dmitri", "Amara", "Hugo", "Yuki", "Bruno", "Elsa", "Karim", "Mila", "Oscar", "Priya", "Sven", "Tess", "Viktor", "Wanda", "Zane", "Alba", "Boris", "Clara", "Dario", "Edith", "Felix", "Greta", "Hana", "Ivan", "Julia"]
CITIES = ["Porto", "Lyon", "Osaka", "Cairo", "Oslo", "Lima", "Turin", "Hanoi", "Perth", "Quito", "Riga", "Seoul", "Tunis", "Ulm", "Vigo", "Wells", "Xian", "Yalta", "Zurich", "Bonn", "Cork", "Delhi", "Essen", "Faro", "Graz", "Hull", "Izmir", "Jena", "Kobe", "Linz"]
TOPICS = ["a volcano", "the printing press", "photosynthesis", "a rainbow", "the moon", "a violin", "inflation", "a glacier", "the internet", "a bridge", "rust", "a tornado", "chess", "a vaccine", "the ocean", "a library", "gravity", "a coral reef", "the alphabet", "a thunderstorm",
          "a compass", "bread", "a satellite", "the immune system", "a desert", "electricity", "a map", "a telescope", "a forest fire", "the stock market", "a lighthouse", "an earthquake", "soap", "a clock", "the water cycle", "a microscope", "a canal", "a spreadsheet", "the sun", "a train",
          "glass", "a rainforest", "a magnet", "a river delta", "a battery", "the postal service", "a pyramid", "a spider web", "a loaf of cheese", "a harbor"]
PHR = ["Describe {t} in one sentence.", "Explain {t} briefly in one sentence.", "Give a one-sentence summary of {t}.", "In one sentence, what is {t}?"]
MOJO_T = ["returns its Int argument plus {k}", "returns its Int argument times {k}", "returns True if its Int argument is greater than {k}", "returns the Int argument minus {k}", "returns the larger of its Int argument and {k}",
          "returns the smaller of its Int argument and {k}", "returns True if its Int argument is divisible by {k}", "returns its Int argument modulo {k}", "returns the square of its Int argument plus {k}", "returns {k} if its Int argument is zero, else the argument"]

SK = {
    "json": {"prompt": 'Answer with a single JSON object with exactly the keys "name" and "age" and nothing else.', "cap": 40, "split": (50, 50, 100)},
    "style": {"prompt": f"Style rule: never use the em dash character ({EM}) and always end the answer with the word Done.", "cap": 80, "split": (50, 50, 100)},
    "mojo": {"prompt": "Mojo 1.2 rules: declare functions with def (not fn), use comptime (not alias), import from std. (for example from std.math import sqrt). Write only code.", "cap": 160, "split": (50, 20, 30)},
}
ORDER = ["json", "style", "mojo"]

def make_tasks(sk):
    rng = random.Random({"json": 11, "style": 12, "mojo": 13}[sk]); out = []
    if sk == "json":
        tpl = ["{n} is {a} years old and lives in {c}.", "Last week {n}, aged {a}, moved to {c}.", "In {c} we met {n} ({a})."]
        for i in range(200):
            n, a, c = rng.choice(NAMES), rng.randint(18, 90), rng.choice(CITIES)
            out.append({"task": f"Text: {tpl[i % 3].format(n=n, a=a, c=c)}\nExtract the person's name and age.\nAnswer:", "gold": {"name": n, "age": a}})
    elif sk == "style":
        for i in range(200): out.append({"task": f"Question: {PHR[i % 4].format(t=TOPICS[i % 50])}\nAnswer:", "gold": None})
    else:
        for i in range(100): out.append({"task": f"Task: write a Mojo function named f that {MOJO_T[i % 10].format(k=2 + (i // 10))}.\nCode:", "gold": None})
    rng.shuffle(out); a, b, c = SK[sk]["split"]
    return {"train": out[:a], "dev": out[a:a + b], "test": out[a + b:a + b + c]}

def cut(gen):
    t = gen
    for stop in ("\n\n", "\nQuestion:", "\nText:", "\nTask:"):
        j = t.find(stop, 1)
        if j > 0: t = t[:j]
    return t.strip()

def compile_ok(code, tag):
    if "```" in code:
        parts = code.split("```"); code = parts[1] if len(parts) > 1 else code
        code = code[4:] if code.startswith("mojo") else code
    if "def f" not in code: return False
    if "def main" not in code: code += "\n\ndef main():\n    pass\n"
    d = f"{W}/g4/compile"; os.makedirs(d, exist_ok=True); p = f"{d}/{tag}.mojo"; open(p, "w").write(code)
    if DRY: return True
    return subprocess.run([MOJO, "build", p, "-o", f"{d}/{tag}.bin"], capture_output=True, timeout=120).returncode == 0

def score(sk, task, text, tag):
    if sk == "json":
        try: j = json.loads(cut(text))
        except Exception: return 0
        return int(isinstance(j, dict) and set(j) == {"name", "age"} and j["name"] == task["gold"]["name"] and j["age"] == task["gold"]["age"] and isinstance(j["age"], int))
    if sk == "style":
        t = cut(text); return int(EM not in t and t.endswith("Done."))
    return int(compile_ok(text if "```" in text else text.split("\n\n\n")[0], tag))

def first_sentence(p): return p.split(". ")[0] + ("" if p.split(". ")[0].endswith(".") else ".")
def pre_ids(sk, short=False): return e2((first_sentence(SK[sk]["prompt"]) if short else SK[sk]["prompt"]) + "\n")
def gen_req(rid, sk, task, mode, **kw):
    pre = [] if mode == "none" else pre_ids(sk, mode == "short")
    return {"id": rid, "prompt": pre + e2(task), "n": SK[sk]["cap"], **kw}
def tasks(): return {sk: json.load(open(f"{W}/g4-tasks-{sk}.json")) for sk in ORDER}
def layer(): return json.load(open(f"{W}/select.json"))["layer"]
def txt(out, rid): return ih.dec(out[rid]["toks"])


if ST == "data":
    for sk in ORDER:
        t = make_tasks(sk); json.dump(t, open(f"{W}/g4-tasks-{sk}.json", "w")); print(sk, {k: len(v) for k, v in t.items()})

if ST == "gen" and not ih.exists("g4-gen.json"):
    T = tasks(); ids = ih.Ids(); rs = []
    for sk in ORDER:
        for i, t in enumerate(T[sk]["train"]): rs.append(gen_req(ids.new(kind="train", sk=sk, i=i), sk, t["task"], "with"))
        for sp in ("dev", "test"):
            for mode in ("with", "none", "short"):
                for i, t in enumerate(T[sk][sp]): rs.append(gen_req(ids.new(kind="base", sk=sk, sp=sp, mode=mode, i=i), sk, t["task"], mode))
    out, _ = ih.engine("g4-gen", rs, timeout=5400)
    res = {"train_O": {sk: [] for sk in ORDER}, "scores": {}, "texts": {}}
    for rid, m in ids.meta.items():
        t = txt(out, rid)
        if m["kind"] == "train": res["train_O"][m["sk"]].append(out[rid]["toks"][:8])
        else:
            tk = T[m["sk"]][m["sp"]][m["i"]]; key = f"{m['sk']}|{m['sp']}|{m['mode']}"
            res["scores"].setdefault(key, []).append(score(m["sk"], tk, t, f"gen-{rid}")); res["texts"].setdefault(key, []).append(t)
    pf = {}  # engine prefill seconds with and without the prompt on the first 10 test tasks
    for rid, m in ids.meta.items():
        if m["kind"] == "base" and m["sp"] == "test" and m["i"] < 10: pf.setdefault(f"{m['sk']}|{m['mode']}", []).append(out[rid].get("prefill", 0.0))
    res["prefill"] = {k: float(np.mean(v)) for k, v in pf.items()}
    json.dump(res, open(f"{W}/g4-gen.json", "w")); print({k: round(float(np.mean(v)), 3) for k, v in res["scores"].items()})

if ST == "cap" and not ih.exists("g4-file.json"):
    T = tasks(); G = json.load(open(f"{W}/g4-gen.json")); L = layer(); cd = f"{W}/g4/cap"; os.makedirs(cd, exist_ok=True); rs, meta = [], []
    for variant in ("with", "none"):
        for sk in ORDER:
            for i, t in enumerate(T[sk]["train"]):
                O = G["train_O"][sk][i]; ctx = (pre_ids(sk) if variant == "with" else []) + e2(t["task"]); rs.append({"id": len(rs) + 1, "prompt": ctx + O, "n": 1}); meta.append((variant, sk, i, len(ctx)))
    out, _ = ih.engine("g4-cap", rs, {"BARO_CAPTURE": cd, "BARO_CAPTURE_LAYERS": str(L), "BARO_CAPTURE_FIELDS": "xr"}, timeout=3000)
    H = 4096
    if not DRY:
        meta_j = json.load(open(f"{cd}/meta.json")); H = meta_j["h"]; fields = meta_j["fields"]; nf = len(fields)
        arr = np.fromfile(f"{cd}/L{L}.f16", dtype=np.float16).reshape(-1, nf, H); lens = [len(r["prompt"]) for r in rs]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        assert arr.shape[0] == sum(lens), (arr.shape, sum(lens))
    rows = {sk: [] for sk in ORDER}; half = len(rs) // 2
    for idx in range(half):
        variant, sk, i, ctxlen = meta[idx]; ctxlen0 = meta[half + idx][3]; O = G["train_O"][sk][i]
        for j in range(len(O) + 1):
            if DRY: rows[sk].append((np.random.randn(H).astype(np.float32), np.random.randn(H).astype(np.float32))); continue
            pw, pn = int(starts[idx]) + ctxlen - 1 + j, int(starts[half + idx]) + ctxlen0 - 1 + j
            key = arr[pn, fields.index("x")].astype(np.float32); push = arr[pw, fields.index("r")].astype(np.float32) - arr[pn, fields.index("r")].astype(np.float32); rows[sk].append((key, push))
    keys = np.stack([k for sk in ORDER for k, v in rows[sk]]); vals = np.stack([v for sk in ORDER for k, v in rows[sk]]); N = len(keys); ranges = {}; a = 0
    for sk in ORDER: ranges[sk] = [a, a + len(rows[sk])]; a += len(rows[sk])
    rng = np.random.default_rng(0); rk = rng.standard_normal(keys.shape).astype(np.float32); rk *= np.linalg.norm(keys, axis=1, keepdims=True) / np.linalg.norm(rk, axis=1, keepdims=True)
    rv = rng.standard_normal(vals.shape).astype(np.float32); rv *= np.linalg.norm(vals, axis=1, keepdims=True) / np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-12)
    ih.write_sets(f"{W}/g4-imp.bin", H, [{"layer": L, "k": 2 * N, "tau": TAU, "keys": np.concatenate([keys, rk]), "gates": np.full(2 * N, 1 / (1 - TAU), np.float32), "vals": np.concatenate([vals, rv])}])
    for sk in ORDER:
        v = np.mean([x for _, x in rows[sk]], axis=0).astype(np.float32)
        for s in VSC: (v * s).tofile(f"{W}/g4-vec-{sk}-{s}.f32")
    json.dump({"N": N, "ranges": ranges, "layer": L, "push_norm": {sk: float(np.mean([np.linalg.norm(x) for _, x in rows[sk]])) for sk in ORDER}}, open(f"{W}/g4-file.json", "w")); print("g4 file:", N, "real rows,", ranges)

def impl_req(rid, sk, task, F, mode, scale):
    lo, hi = F["ranges"][sk]; r = gen_req(rid, sk, task, "none")
    if mode == "impl": r.update({"implants": 1, "implants_layer": F["layer"], "implants_lo": lo, "implants_k": hi, "implants_scale": scale})
    elif mode == "ctrl": r.update({"implants": 1, "implants_layer": F["layer"], "implants_lo": F["N"] + lo, "implants_k": F["N"] + hi, "implants_scale": scale})
    elif mode == "vec": r.update({"implants": 0, "steer_file": f"{W}/g4-vec-{sk}-{scale}.f32", "steer_layer": F["layer"]})
    return r

if ST == "dev" and not ih.exists("g4-dev.json"):
    T = tasks(); F = json.load(open(f"{W}/g4-file.json")); ids = ih.Ids(); rs = []
    for sk in ORDER:
        for i, t in enumerate(T[sk]["dev"]):
            for s in ISC: rs.append(impl_req(ids.new(kind="impl", sk=sk, s=s, i=i), sk, t["task"], F, "impl", s))
            for s in VSC: rs.append(impl_req(ids.new(kind="vec", sk=sk, s=s, i=i), sk, t["task"], F, "vec", s))
    out, _ = ih.engine("g4-dev", rs, {"BARO_IMPLANTS": f"{W}/g4-imp.bin"}, timeout=5400)
    sc = {}
    for rid, m in ids.meta.items(): sc.setdefault((m["sk"], m["kind"], m["s"]), []).append(score(m["sk"], T[m["sk"]]["dev"][m["i"]], txt(out, rid), f"dev-{rid}"))
    best = {}
    for sk in ORDER:
        for kind, grid in (("impl", ISC), ("vec", VSC)):
            m = {s: float(np.mean(sc[(sk, kind, s)])) for s in grid}; b = max(m.values()); best[f"{sk}|{kind}"] = {"scale": min(s for s in grid if m[s] == b), "dev_score": b, "by_scale": {str(k): v for k, v in m.items()}}
    json.dump(best, open(f"{W}/g4-dev.json", "w"), indent=1); print(json.dumps({k: (v["scale"], round(v["dev_score"], 3)) for k, v in best.items()}))

if ST == "test" and not ih.exists("g4-test.json"):
    T = tasks(); F = json.load(open(f"{W}/g4-file.json")); D = json.load(open(f"{W}/g4-dev.json")); ids = ih.Ids(); rs = []; unrel = ih.unrel_items(); win = ih.windows()
    for sk in ORDER:
        for mode in ("impl", "ctrl", "vec"):
            s = D[f"{sk}|{'vec' if mode == 'vec' else 'impl'}"]["scale"]
            for i, t in enumerate(T[sk]["test"]): rs.append(impl_req(ids.new(kind=mode, sk=sk, i=i), sk, t["task"], F, mode, s))
        s = D[f"{sk}|impl"]["scale"]; lo, hi = F["ranges"][sk]
        for i, (q, g) in enumerate(unrel): rs.append(ih.req(ids.new(kind="u_on", sk=sk, i=i), q, implants=1, implants_layer=F["layer"], implants_lo=lo, implants_k=hi, implants_scale=s))
    for i, (q, g) in enumerate(unrel): rs.append(ih.req(ids.new(kind="u_off", i=i), q, implants=0))
    pr, last = [], {}
    for i, (c, t) in enumerate(win): pr.append({"id": ids.new(kind="pplbase"), "prompt": c, "n": 1, "implants": 0, "slens_ctx": i, "slens_k": 0})
    for sk in ORDER:
        s = D[f"{sk}|impl"]["scale"]; lo, hi = F["ranges"][sk]
        for i, (c, t) in enumerate(win):
            rid = ids.new(kind="ppl", sk=sk); pr.append({"id": rid, "prompt": c, "n": 1, "implants": 1, "implants_layer": F["layer"], "implants_lo": lo, "implants_k": hi, "implants_scale": s, "slens_ctx": i, "slens_k": 1, "slens_tgt": t, "slens_flags": (1 if i == 0 else 0) + (2 if i == len(win) - 1 else 0)})
            if i == len(win) - 1: last[sk] = rid
    out, _ = ih.engine("g4-test", rs + pr, {"BARO_IMPLANTS": f"{W}/g4-imp.bin"}, timeout=5400)
    res = {"scores": {}, "unrel": {}, "ppl": {}}
    for rid, m in ids.meta.items():
        if m["kind"] in ("impl", "ctrl", "vec"): res["scores"].setdefault(f"{m['sk']}|{m['kind']}", []).append(score(m["sk"], T[m["sk"]]["test"][m["i"]], txt(out, rid), f"test-{rid}"))
        if m["kind"] == "u_on": res["unrel"].setdefault(f"{m['sk']}|on", []).append(ih.ok_of(out, rid, unrel[m["i"]][1]))
        if m["kind"] == "u_off": res["unrel"].setdefault("off", []).append(ih.ok_of(out, rid, unrel[m["i"]][1]))
    for sk in ORDER: res["ppl"][sk] = float(np.exp(out[last[sk]]["slens"]["dnll"]) - 1)
    json.dump(res, open(f"{W}/g4-test.json", "w")); print({k: round(float(np.mean(v)), 3) for k, v in res["scores"].items()})

if ST == "verdict":
    G, D, R, F = (json.load(open(f"{W}/{n}.json")) for n in ("g4-gen", "g4-dev", "g4-test", "g4-file")); T = tasks(); m = lambda v: float(np.mean(v)); lines = ["## G4 prompt to implant compiler (prereg docs/prereg-implants.md + docs/prereg-g4-tasks.md)\n"]
    lines += ["| skill | n test | s_with | s_0 | s_short | s_impl | s_ctrl | s_vec | headroom D-A | retained D-B | control ratio D-C | unrelated drop | ppl change D-D | impl minus short | tokens saved | prefill with/without (s) | verdict |", "|" + "---|" * 17]
    for sk in ORDER:
        sw, s0, ss = (m(G["scores"][f"{sk}|test|{x}"]) for x in ("with", "none", "short")); si, sc, sv = (m(R["scores"][f"{sk}|{x}"]) for x in ("impl", "ctrl", "vec")); n = len(T[sk]["test"])
        head = sw - s0; ret = (si - s0) / head if head >= 0.2 else float("nan"); retc = (sc - s0) / head if head >= 0.2 else float("nan"); du = m(R["unrel"]["off"]) - m(R["unrel"][f"{sk}|on"]); pc = R["ppl"][sk]
        A = head >= 0.2; B = A and ret >= 0.6; C = A and retc <= 0.2; Dd = du <= 0.01 and pc <= 0.01; useful = si - ss >= 0.10
        verdict = "no headroom" if not A else ("PASS" if B and C and Dd and useful else "MISSED: " + ", ".join(x for x, ok in (("D-B", B), ("D-C", C), ("D-D", Dd), ("beats s_short", useful)) if not ok))
        if n < 30: verdict += " (UNDERPOWERED)"
        toks = len(pre_ids(sk)); pf = (G["prefill"].get(f"{sk}|with", 0), G["prefill"].get(f"{sk}|none", 0))
        lines.append(f"| {sk} | {n} | {sw:.3f} | {s0:.3f} | {ss:.3f} | {si:.3f} | {sc:.3f} | {sv:.3f} | {head:+.3f} {'ok' if A else 'LOW'} | {ret:.3f} | {retc:.3f} | {du:+.3f} | {pc:+.4f} | {si - ss:+.3f} | {toks} | {pf[0]:.3f} / {pf[1]:.3f} | {verdict} |")
    lines.append(f"\nlayer {F['layer']}, tau {TAU}, rows per skill {[F['ranges'][s][1] - F['ranges'][s][0] for s in ORDER]}; dev-chosen scales {{ {', '.join(k + ': ' + str(v['scale']) for k, v in D.items())} }}")
    open(f"{W}/verdict-g4.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
