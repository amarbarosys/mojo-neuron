"""nx chat: a terminal chat with the model behind a running baro-serve (scripts/agents/baro-serve.sh up).
  nx chat [--url URL] [--steer L:N:gain,...] [--persona NAME] [--max N] [--script FILE] [--up GGUF]
Every turn is one POST /v1/chat/completions (history kept, `steer` sent as [{layer, neuron, scale}], E1: the engine multiplies FFN activation N of layer L by scale, 0 ablates, 1 leaves it) and ends
with an rlog line (tokens, tok/s). --script FILE: one user line per turn, no prompt. --up starts the server first and stops it at the end.
Transcript: .work/chat/<run>.jsonl (one user and one assistant line per turn)."""
from std.os import getenv, makedirs
from std.os.path import exists
from std.subprocess import run
from std.sys import argv
from std.time import perf_counter_ns
from json import Json, parse
from mneuron.rlog import Rlog
from product.persona import apply as persona_apply
from mneuron.stdin import read_line


def post(url: String, body: Json) raises -> Json:
    makedirs(".work/chat", exist_ok=True)
    with open(".work/chat/req.json", "w") as f:
        f.write(body.dumps(False))
    return parse(run("curl -s -m 900 -H 'content-type: application/json' --data-binary @.work/chat/req.json '" + url + "/v1/chat/completions'"))


def steer_json(recipe: String) raises -> Json:
    var a = Json.array()
    for u in recipe.split(","):
        var p = u.split(":")
        var o = Json.object()
        o.set("layer", Json.of(Int(String(p[0]))))
        o.set("neuron", Json.of(Int(String(p[1]))))
        o.set("scale", Json.of(Float64(String(p[2])[byte=1:]) if String(p[2]).startswith("+") else Float64(String(p[2]))))
        a.push(o)
    return a^


def chat() raises:
    var a = argv()
    var url = String("http://127.0.0.1:") + String(getenv("BARO_PORT", "18090"))
    var steer = String()
    var persona = String()
    var script = String()
    var up = String()
    var mx = 128
    var i = 2
    while i + 1 < len(a):
        var k = String(a[i])
        var v = String(a[i + 1])
        if k == "--url":
            url = v
        elif k == "--steer":
            steer = v
        elif k == "--persona":
            persona = v
        elif k == "--max":
            mx = Int(v)
        elif k == "--script":
            script = v
        elif k == "--up":
            up = v
        else:
            raise Error("unknown option " + k + ". usage: nx chat [--url U] [--steer L:N:gain,..] [--persona NAME] [--max N] [--script FILE] [--up GGUF]")
        i += 2
    var pfield = Json.object()
    if persona.byte_length() > 0:
        if steer.byte_length() > 0:
            raise Error("--persona and --steer together: pick one")
        var pa = Json.object()
        pa.set("name", Json.of(persona))
        pfield = persona_apply(pa).get("steer")
    var sh = String(getenv("NX_SERVE", "scripts/agents/baro-serve.sh"))
    if up.byte_length() > 0:
        print(run(sh + " up '" + up + "'").strip())
    var ml = String(run("curl -s -m 10 '" + url + "/v1/models'"))
    if ml.byte_length() == 0:
        raise Error("chat: no server answers on " + url + " (scripts/agents/baro-serve.sh up)")
    var model = parse(ml).get("data").items()[0].get("id").str()
    var run_id = String(run("date -u +%Y%m%dT%H%M%S").strip())
    var tr = ".work/chat/" + run_id + ".jsonl"
    var msgs = Json.array()
    var lines = List[String]()
    if script.byte_length() > 0:
        for l in open(script, "r").read().split("\n"):
            if String(l).byte_length() > 0:
                lines.append(String(l))
    var turn = 0
    while True:
        var user = String()
        if script.byte_length() > 0:
            if turn >= len(lines):
                break
            user = lines[turn]
        else:
            print("you> ", end="")
            var eof = False
            user = read_line(eof)
            if eof or user == "/quit" or user.byte_length() == 0:
                break
        turn += 1
        var m = Json.object()
        m.set("role", Json.of(String("user")))
        m.set("content", Json.of(user))
        msgs.push(m^)
        var req = Json.object()
        req.set("model", Json.of(model))
        req.set("messages", msgs.copy())
        req.set("max_tokens", Json.of(mx))
        req.set("temperature", Json.of(0.0))
        req.set("stream", Json.of(False))
        if steer.byte_length() > 0:
            req.set("steer", steer_json(steer))
        elif persona.byte_length() > 0:
            req.set("steer", pfield)
        var rl = Rlog("chat", url + " model " + model + " turn " + String(turn) + (" steer " + steer if steer.byte_length() > 0 else String()), String(getenv("NX_MODEL_FILE", "")))
        var t0 = perf_counter_ns()
        var r = post(url, req)
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        if not r.has("choices"):
            var full = r.dumps(False)
            var body = String(full[byte = 0 : min(full.byte_length(), 300)])
            rl.event("anomaly", "no choices in the reply: " + body, tr)
            raise Error("chat: the server answered without choices: " + body)
        var reply = r.get("choices").items()[0].get("message").text("content")
        var pt = r.get("usage").get("prompt_tokens").int()
        var ct = r.get("usage").get("completion_tokens").int()
        var am = Json.object()
        am.set("role", Json.of(String("assistant")))
        am.set("content", Json.of(reply))
        msgs.push(am^)
        print("model> " + reply)
        with open(tr, "a") as f:
            f.write(Json.of(user).dumps(False) + "\n" + Json.of(reply).dumps(False) + "\n")
        var tps = Float64(ct) / max(secs, 1e-9)
        rl.done("turn=" + String(turn) + " prompt_tokens=" + String(pt) + " completion_tokens=" + String(ct) + " wall_s=" + String(secs) + " tok_s=" + String(tps) + (" steer=" + steer if steer.byte_length() > 0 else String()), tr, [Float64(pt), Float64(ct), tps], 0.0, 1e6)
    if up.byte_length() > 0:
        print(run(sh + " down").strip())
