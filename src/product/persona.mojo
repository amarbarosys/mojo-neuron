"""Persona store (F1): `nx persona list | save | apply`, one dir per persona under $PERSONA_DIR (default <main checkout>/.work/personas):
persona.json (name, model, gguf, recipe, rho, labels, lens per unit, sha of both vectors), steer.f32 (the BARO_STEER / per-request
`steer_file` vector) and steer.cvec.gguf (llama.cpp --control-vector), both built by the `cvec` tool ($NX_CVEC, default .work/cvec of
the main checkout).
save {name, recipe, gguf | model, rho?, labels?{"L:N": text}, overwrite?}; apply {name} -> files + the engine env and llama.cpp flag, after
the vectors' sha is checked against persona.json."""
from std.os import getenv
from std.os.path import exists
from std.sys import argv
from json import Json, parse
from mneuron.rlog import Rlog, sh
from mneuron.store import read_text, model_info


def pdir() -> String:
    var e = String(getenv("PERSONA_DIR"))
    return e if e.byte_length() > 0 else String(getenv("HOME")) + "/Projects/mojo/mojo-neuron/.work/personas"


def sha(path: String) raises -> String:
    return sh("sha256sum '" + path + "' | cut -c1-16")


def valid(name: String) -> Bool:
    if name.byte_length() == 0:
        return False
    for c in name.as_bytes():
        if not ((c >= 97 and c <= 122) or (c >= 48 and c <= 57) or c == 45 or c == 95):
            return False
    return True


def save(a: Json) raises -> Json:
    var name = a.text("name")
    if not valid(name):
        raise Error("persona name must be [a-z0-9_-]+, got '" + name + "'")
    var d = pdir() + "/" + name
    if exists(d + "/persona.json") and not (a.has("overwrite") and a.get("overwrite").int() == 1):
        raise Error("persona '" + name + "' exists (overwrite=1 replaces it)")
    var model = a.text("model")
    var gguf = a.text("gguf") if a.has("gguf") else model_info(model).text("gguf")
    var recipe = String(a.text("recipe").replace(" ", ""))
    var cvec = String(getenv("NX_CVEC"))
    cvec = cvec if cvec.byte_length() > 0 else String(getenv("HOME")) + "/Projects/mojo/mojo-neuron/.work/cvec"
    _ = sh("mkdir -p '" + d + "'")
    _ = sh("'" + cvec + "' '" + gguf + "' '" + recipe + "' '" + d + "/steer'" + (" '" + a.text("rho") + "'" if a.has("rho") else String("")))
    var lens = Json.array()
    if model.byte_length() > 0:
        for u in recipe.split(","):
            var p = String(u).split(":")
            try:
                var l = parse(sh("'" + String(argv()[0]) + "' lens '{\"model\":\"" + model + "\",\"layer\":" + String(p[0]) + ",\"neuron\":" + String(p[1]) + "}'"))
                var e = Json.object()
                e.set("unit", Json.of(String(p[0]) + ":" + String(p[1])))
                e.set("top", l.get("top"))
                lens.push(e)
            except:
                pass  # no lens.bin for that layer: the persona keeps the unit without a lens
    var j = Json.object()
    j.set("v", Json.of(1))
    j.set("name", Json.of(name))
    j.set("model", Json.of(model))
    j.set("gguf", Json.of(gguf))
    j.set("recipe", Json.of(recipe))
    j.set("rho", Json.of(a.text("rho")))
    j.set("labels", a.get("labels") if a.has("labels") else Json.object())
    j.set("lens", lens)
    j.set("sha_f32", Json.of(sha(d + "/steer.f32")))
    j.set("sha_cvec", Json.of(sha(d + "/steer.cvec.gguf")))
    with open(d + "/persona.json.tmp", "w") as f:
        f.write(j.dumps(False))
    _ = sh("mv '" + d + "/persona.json.tmp' '" + d + "/persona.json'")
    var rl = Rlog("persona", a.dumps(), gguf)
    rl.done("save " + name + " " + recipe, j.text("sha_f32"), List[Float64](), 0.0, 1.0)
    return j^


def apply(a: Json) raises -> Json:
    var name = a.text("name")
    var d = pdir() + "/" + name
    if not valid(name) or not exists(d + "/persona.json"):
        raise Error("no persona '" + name + "' in " + pdir())
    var j = parse(read_text(d + "/persona.json"))
    if sha(d + "/steer.f32") != j.text("sha_f32") or sha(d + "/steer.cvec.gguf") != j.text("sha_cvec"):
        Rlog("persona", a.dumps(), j.text("gguf")).event("anomaly", "apply " + name + " | vector files differ from persona.json", j.text("sha_f32"))
        raise Error("persona '" + name + "': vector files differ from persona.json (edited or truncated)")
    var o = Json.object()
    o.set("name", Json.of(name))
    o.set("recipe", j.get("recipe"))
    o.set("steer_file", Json.of(d + "/steer.f32"))
    o.set("cvec", Json.of(d + "/steer.cvec.gguf"))
    var field = Json.object()  # E1: baro-serve forwards {"file": PATH} as the engine's steer_file
    field.set("file", Json.of(d + "/steer.f32"))
    o.set("steer", field)
    o.set("env", Json.of("BARO_STEER=" + d + "/steer.f32"))
    o.set("llama", Json.of("--control-vector " + d + "/steer.cvec.gguf"))
    var rl = Rlog("persona", a.dumps(), j.text("gguf"))
    rl.done("apply " + name, j.text("sha_f32"), List[Float64](), 0.0, 1.0)
    return o^


def persona(a: Json) raises -> Json:
    var op = a.text("op", "list")
    if op == "save":
        return save(a)
    if op == "apply":
        return apply(a)
    if op != "list":
        raise Error("persona op must be list, save or apply")
    var out = Json.array()
    var names = sh("ls '" + pdir() + "' 2>/dev/null || true")
    for n in names.split("\n"):
        if n.byte_length() > 0 and exists(pdir() + "/" + String(n) + "/persona.json"):
            var j = parse(read_text(pdir() + "/" + String(n) + "/persona.json"))
            var e = Json.object()
            e.set("name", j.get("name"))
            e.set("model", j.get("model"))
            e.set("recipe", j.get("recipe"))
            out.push(e)
    return out^
