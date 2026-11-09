"""Image out (C4): one picture per chat turn from the strongest features. The prompt = the turn text + the words a feature set promotes
(lens tokens, cleaned to plain lowercase words) + a fixed style tail. Rendering is a job file for comfy-run (zimage) run inside ONE resident
ComfyUI (`with-comfy`) under gpu-wait (21 GB, priority 20), so the picture never overlaps an engine run and N turns share one server start."""
from std.os import getenv
from json import Json

comptime STYLE = "abstract concept art, soft light, detailed, no text"
comptime RUN = "/iTools/design/comfy-run/comfy_run.py"
comptime WITH = "/iTools/bin/with-comfy"  # absolute: the gpu-wait daemon spawns jobs with its own PATH


def word(t: String) -> String:
    """A lens token as one plain lowercase word, '' when it has fewer than 3 letters."""
    var out = List[UInt8]()
    for b in t.as_bytes():
        if b >= 65 and b <= 90:
            out.append(b + 32)
        elif b >= 97 and b <= 122:
            out.append(b)
    return String(unsafe_from_utf8=Span(out)) if len(out) >= 3 else String()


def turn_prompt(text: String, words: List[String]) -> String:
    var p = text
    var seen = List[String]()
    for w in words:
        var c = word(w)
        if c.byte_length() == 0:
            continue
        var dup = False
        for s in seen:
            dup = dup or s == c
        if not dup:
            seen.append(c)
            p += ", " + c
    return p + ", " + STYLE


def jobs_json(prompts: List[String]) -> String:
    var jobs = Json.array()
    for i in range(len(prompts)):
        var j = Json.object()
        j.set("id", Json.of("turn-" + String(i + 1)))
        j.set("kind", Json.of(String("zimage")))
        j.set("prompt", Json.of(prompts[i]))
        j.set("seed", Json.of(7 + i))
        jobs.push(j)
    return jobs.dumps(False)


def render_cmd(jobs: String, dest: String, runner: String) -> String:
    """The shell line: gpu-wait admits the 21 GB ComfyUI job, with-comfy serves it for this one command."""
    var home = String(getenv("HOME"))
    return runner + " " + home + WITH + " -- python3 " + home + RUN + " '" + jobs + "' --out '" + dest + "'"
