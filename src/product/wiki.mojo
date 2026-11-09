"""Wikipedia neuron map (F30). Forward `nx wiki {model, dir?, thr?, k?}`: for every Brain neuron note `L<l>-N<n>.md` under <dir> (default
$MNEURON_BRAIN or (private notes) /<model>) the label is the promoted lens tokens cleaned to words (alphabetic, >= 4 letters, not
a stop word); `brain-ask --net wiki "<label>"` gives the wiki neurons, those with score >= thr (default 4.5) that also share a word with the label (first 5 letters, docs/prereg-f30.md) are the
links: a `## Wikipedia` section in the note (`[[wiki/Title]]`, rewritten on every run) and a row in $MNEURON_DIR/<model>/wiki.jsonl (every unit,
linked or not). A label under 2 words is not asked. Reverse `nx wiki {query, model?, k?}`: the wiki neurons of the query, then the mapped
units linked to one of them, best first. The 40% of enwiki outside the net: ~/iTools/corpus/wiki-mine (not wired here)."""
from std.os import getenv
from std.sys import get_defined_int
from json import Json, parse
from mneuron.rlog import Rlog, sh
from mneuron.store import read_text, root, mdir, Steer, Lens, tokens

comptime MUT = get_defined_int["WIKI_MUT", 0]()  # 1: the label is the whole note body (the injection check must catch it)
comptime STOP = "that this with from have were been they them their there which would could should about into than then what when where while also very more most some such only over under after before other these those being does done made make like just"


def sq(s: String) -> String:
    return "'" + s.replace("'", "'\\''") + "'"


def words(text: String) -> List[String]:
    """Lower-case alphabetic words of >= 4 letters, no stop words, no repeats."""
    var out = List[String]()
    var cur = String()
    for c in (text + " ").codepoints():
        var b = Int(c)
        if (b >= 65 and b <= 90) or (b >= 97 and b <= 122):
            cur += chr(b + 32 if b <= 90 else b)
        else:
            if cur.byte_length() >= 4 and not (" " + cur + " " in " " + String(STOP) + " ") and cur not in out:
                out.append(cur)
            cur = String()
    return out^


struct Link(Copyable, Movable):
    var title: String
    var score: Float64

    def __init__(out self, title: String, score: Float64):
        self.title = title
        self.score = score


def ask(q: String, k: Int = 5) raises -> List[Link]:
    """Top wiki neurons of `brain-ask --net wiki`: lines `wiki/<Title>.md  (score s)`."""
    var out = List[Link]()
    for line in sh("brain-ask --net wiki " + sq(q) + " 2>/dev/null | grep -a '(score ' | head -" + String(k) + " || true").split("\n"):
        var i = line.find(".md  (score ")
        if line.startswith("wiki/") and i > 0:
            out.append(Link(String(line[byte=5:i]), Float64(String(line[byte=i + 12:line.byte_length() - 1]))))
    return out^


def shares(ws: List[String], title: String) -> Bool:
    """The title has a word whose first 5 letters equal those of a label word (Chlorophyll, chlorophyll)."""
    for t in words(title):
        for w in ws:
            if String(t[byte=:min(5, t.byte_length())]) == String(w[byte=:min(5, w.byte_length())]):
                return True
    return False


def promoted(note: String) -> String:
    var a = note.find("promotes the tokens")
    var b = note.find(" and suppresses")
    return String(note[byte=a + 19:b]) if a >= 0 and b > a else String()


def forward(a: Json) raises -> Json:
    var model = a.text("model")
    var thr = a.get("thr").num() if a.has("thr") else 4.5
    var k = a.get("k").int() if a.has("k") else 3
    var broot = String(getenv("MNEURON_BRAIN"))
    var dir = a.text("dir", (broot if broot.byte_length() > 0 else String(getenv("HOME")) + "/Brain/mojo-neuron/neurons") + "/" + model)
    var rl = Rlog("wiki", a.dumps(), "")
    var map = String()
    var linked = 0
    var vals = List[Float64]()
    var names = sh("ls " + sq(dir) + " | grep -E '^L[0-9]+-N[0-9]+\\.md$' || true")
    var n = 0
    for f in names.split("\n"):
        if f.byte_length() == 0:
            continue
        n += 1
        var path = dir + "/" + String(f)
        var note = read_text(path)
        var ws = words(note if MUT == 1 else promoted(note))
        var label = String(" ").join(ws)
        var links = List[Link]()
        var best = -1e9
        if len(ws) >= 2:
            for l in ask(label, k):
                best = max(best, l.score)
                if l.score >= thr and shares(ws, l.title):
                    links.append(l.copy())
            vals.append(Float64(len(links)))
        var cut = note.find("\n## Wikipedia\n")
        if cut >= 0:  # drop the previous section: up to the next "## " or the end
            var nxt = note.find("\n## ", cut + 1)
            note = String(note[byte=:cut]) + (String(note[byte=nxt:]) if nxt >= 0 else String())
        if len(links) > 0:
            linked += 1
            var sec = String("\n## Wikipedia\n\n")
            for l in links:
                sec += "- [[wiki/" + l.title + "]] score " + String(l.score) + "\n"
            var at = note.find("\n## Related")
            note = String(note[byte=:at]) + sec + String(note[byte=at:]) if at >= 0 else note + sec
        with open(path, "w") as w:
            w.write(note)
        var p = String(f[byte=1:f.find("-")]), String(f[byte=f.find("-N") + 2:f.find(".md")])
        var j = Json.object()
        j.set("layer", Json.of(Int(p[0])))
        j.set("neuron", Json.of(Int(p[1])))
        j.set("label", Json.of(label))
        j.set("best", Json.of(best))
        var ls = Json.array()
        for l in links:
            var e = Json.object()
            e.set("title", Json.of(l.title))
            e.set("score", Json.of(l.score))
            ls.push(e)
        j.set("links", ls)
        map += j.dumps() + "\n"
    var mdir = root() + "/" + model
    _ = sh("mkdir -p " + sq(mdir))
    with open(mdir + "/wiki.jsonl", "w") as w:
        w.write(map)
    var o = Json.object()
    o.set("units", Json.of(n))
    o.set("linked", Json.of(linked))
    o.set("map", Json.of(mdir + "/wiki.jsonl"))
    rl.done("units=" + String(n) + " linked=" + String(linked), "", vals, 0.0, 1e3)
    return o^


def model_of(path: String) -> String:
    """<root>/<model>/wiki.jsonl -> <model>."""
    var p = path.split("/")
    return String(p[len(p) - 2])


def reverse(a: Json) raises -> Json:
    var q = a.text("query")
    var k = a.get("k").int() if a.has("k") else 10
    var rl = Rlog("wiki", a.dumps(), "")
    var top = ask(q, 10)
    var qw = words(q)
    var hits = List[Json]()
    var sc = List[Float64]()
    for f in sh("ls " + sq(root()) + "/" + (a.text("model") if a.has("model") else String("*")) + "/wiki.jsonl 2>/dev/null || true").split("\n"):
        if f.byte_length() == 0:
            continue
        for line in read_text(String(f)).split("\n"):
            if line.byte_length() < 2:
                continue
            var j = parse(String(line))
            var s = -1.0
            var via = String()
            for w in qw:  # the query is a word of the unit's label
                if (" " + j.text("label") + " ") .find(" " + w + " ") >= 0:
                    s = 5.0
                    via = "label"
            for l in j.get("links").items():
                for t in top:
                    if t.title == l.text("title") and t.score + l.get("score").num() > s:
                        s = t.score + l.get("score").num()  # a linked wiki neuron is one of the query's wiki neurons
                        via = t.title
            if s < 0:
                continue
            var h = Json.object()
            h.set("model", Json.of(model_of(String(f))))
            h.set("layer", j.get("layer"))
            h.set("neuron", j.get("neuron"))
            h.set("label", j.get("label"))
            h.set("via", Json.of(via))
            h.set("score", Json.of(s))
            var i = len(sc)
            while i > 0 and sc[i - 1] < s:
                i -= 1
            sc.insert(i, s)
            hits.insert(i, h^)
    var out = Json.array()
    for i in range(min(k, len(hits))):
        out.push(hits[i].copy())
    var o = Json.object()
    o.set("query", Json.of(q))
    var wl = Json.array()
    for t in top:
        wl.push(Json.of(t.title))
    o.set("wiki", wl)
    o.set("units", out)
    rl.done("query=" + q + " units=" + String(len(hits)), "", List[Float64](), 0.0, 1.0)
    return o^


def toks(tk: List[String], ids: List[Int]) -> List[String]:
    var t = String()
    for i in ids:
        t += " " + tk[i]
    return words(t)


def steer_first(a: Json) raises -> Json:
    """F30b: units of <layer>.steer.bin are labelled by the tokens their steering raises (`up`), the lens tokens are kept as `lens_agree`
    (a shared 5-letter stem). A link stands when a title word shares a stem with the `up` words (`shares`); the null re-checks every link
    against the up words of another unit. {neuron, title}: only whether that title would stand for that unit (a planted link)."""
    var model = a.text("model")
    var layer = a.get("layer").int()
    var st = Steer(mdir(model) + "/" + String(layer) + ".steer.bin")
    var ln = Lens(model, layer)
    var tk = tokens(model)
    var sw = List[Int]()
    var ups = List[List[String]]()
    for n in range(st.ffn):
        if st.id(n, False, 0) >= 0:
            var ids = List[Int]()
            for i in range(st.k):
                if st.id(n, False, i) >= 0:
                    ids.append(st.id(n, False, i))
            sw.append(n)
            ups.append(toks(tk, ids))
    if a.has("title"):
        var at = -1
        for i in range(len(sw)):
            if sw[i] == a.get("neuron").int():
                at = i
        if at < 0:
            raise Error("neuron not swept")
        var o = Json.object()
        o.set("confirmed", Json.of(1 if shares(ups[at], a.text("title")) else 0))
        return o^
    var rl = Rlog("wiki", a.dumps(), "")
    var rows = String()
    var linked = 0
    var agree = 0
    var nullhit = 0
    for i in range(len(sw)):
        var ids = List[Int]()
        for j in range(ln.k):
            ids.append(ln.id(sw[i], False, j))
        var lw = toks(tk, ids)
        var hit = False
        for w in ups[i]:
            for v in lw:
                hit = hit or String(w[byte=:min(5, w.byte_length())]) == String(v[byte=:min(5, v.byte_length())])
        agree += Int(hit)
        var links = List[Link]()
        if len(ups[i]) >= 2:
            for l in ask(String(" ").join(ups[i]), 3):
                if l.score >= 4.5 and shares(ups[i], l.title):
                    links.append(l.copy())
        if len(links) > 0:
            linked += 1
            nullhit += Int(shares(ups[(i + len(sw) // 2) % len(sw)], links[0].title))
        var j = Json.object()
        j.set("neuron", Json.of(sw[i]))
        j.set("up", Json.of(String(" ").join(ups[i])))
        j.set("lens", Json.of(String(" ").join(lw)))
        j.set("lens_agree", Json.of(Int(hit)))
        var la = Json.array()
        for l in links:
            la.push(Json.of(l.title))
        j.set("links", la)
        rows += j.dumps() + "\n"
    with open(mdir(model) + "/" + String(layer) + ".wiki.jsonl", "w") as w:
        w.write(rows)
    var o = Json.object()
    o.set("units", Json.of(len(sw)))
    o.set("linked", Json.of(linked))
    o.set("lens_agree", Json.of(agree))
    o.set("null_hits", Json.of(nullhit))
    rl.done("steer-first units=" + String(len(sw)) + " linked=" + String(linked) + " null=" + String(nullhit), "", List[Float64](), 0.0, 1.0)
    return o^


def wiki(a: Json) raises -> Json:
    if a.has("steer"):
        return steer_first(a)
    return reverse(a) if a.has("query") else forward(a)
