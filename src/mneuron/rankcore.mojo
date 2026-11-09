"""Rank: word-cluster mass of a neuron's steered rollouts against random-direction controls.

mass(u) = share of u's distinctive word occurrences taken by its K most frequent words. A word is
distinctive when it is not an unsteered (`base`) word of the same prompt. Controls are units named
`ctrl*`. pct(u) = share of controls whose mass is below u's (ties count half). Latin words only.
"""
from std.collections import Dict
from json import parse

comptime K = 5
comptime MINLEN = 3
comptime STOP = " the and for are but not you all any can her was one our out has have had his how its may new now see two who did get him let say she too use with that this from they will would there their what about which when your said each been into than then them were more some time very just like also over only must these those such here make made after where while other could should because before being does doing done "


@fieldwise_init
struct Unit(Copyable):
    var name: String  # neuron id as text, or ctrl<i>
    var mass: Float64
    var pct: Float64
    var rows: Int
    var words: Int
    var top: String  # top K words with counts, "word:n word:n"


def words_of(text: String, stop: String) -> List[String]:
    """Lowercase ASCII-letter runs of length >= MINLEN that are not stop words."""
    var out = List[String]()
    var cur = List[UInt8]()
    var bs = text.as_bytes()
    for i in range(len(bs) + 1):
        var c = bs[i] if i < len(bs) else UInt8(32)
        if c >= 65 and c <= 90:
            c += 32
        if c >= 97 and c <= 122:
            cur.append(c)
        else:
            if len(cur) >= MINLEN:
                var w = String(from_utf8_lossy=Span(cur))
                if stop.find(" " + w + " ") < 0:
                    out.append(w)
            cur = List[UInt8]()
    return out^


def key_of(v: Int, i: Int) -> Int:
    return v * 1048576 + i


def rank_text(jsonl: String) raises -> List[Unit]:
    """One JSON per line: neuron (int or string), prompt, dose, text. Returns units sorted by mass, best first."""
    var stop = String(STOP)
    var base = Dict[String, Bool]()  # "prompt\x1fword" present in an unsteered rollout
    var names = List[String]()
    var at = Dict[String, Int]()
    var counts = List[Dict[String, Int]]()
    var rows = List[Int]()
    var total = List[Int]()
    var recs = List[Tuple[String, String, String]]()  # neuron, prompt, text (steered rows)
    for line in jsonl.split("\n"):
        if line.byte_length() == 0:
            continue
        var d = parse(String(line))
        var nj = d.get("neuron")
        var nm = nj.str() if nj.kind() == 4 else String(nj.int())
        var prompt = d.text("prompt")
        if nm == "base":
            for w in words_of(d.text("text"), stop):
                base[prompt + "\x1f" + w] = True
        elif d.get("dose").num() != 0.0:
            recs.append((nm, prompt, d.text("text")))
    for r in recs:
        var nm = r[0]
        if nm not in at:
            at[nm] = len(names)
            names.append(nm)
            counts.append(Dict[String, Int]())
            rows.append(0)
            total.append(0)
        var u = at[nm]
        rows[u] += 1
        for w in words_of(r[2], stop):
            if (r[1] + "\x1f" + w) in base:
                continue
            counts[u][w] = counts[u].get(w, 0) + 1
            total[u] += 1
    var out = List[Unit]()
    var ctrl = List[Float64]()
    for u in range(len(names)):
        var best = List[Int]()  # top K counts, descending
        var bw = List[String]()
        for e in counts[u].items():
            var p = len(best)
            best.append(e.value)
            bw.append(e.key)
            while p > 0 and (best[p - 1] < best[p] or (best[p - 1] == best[p] and bw[p - 1] > bw[p])):
                best.swap_elements(p - 1, p)
                bw.swap_elements(p - 1, p)
                p -= 1
            if len(best) > K:
                _ = best.pop()
                _ = bw.pop()
        var s = 0
        var top = String()
        for j in range(len(best)):
            s += best[j]
            top += (" " if j > 0 else "") + bw[j] + ":" + String(best[j])
        var m = Float64(s) / Float64(total[u]) if total[u] > 0 else 0.0
        out.append(Unit(names[u], m, 0.0, rows[u], total[u], top))
        if names[u].startswith("ctrl"):
            ctrl.append(m)
    for u in range(len(out)):
        if len(ctrl) == 0:
            out[u].pct = -1.0
            continue
        var below = 0.0
        for c in ctrl:
            if c < out[u].mass:
                below += 1.0
            elif c == out[u].mass:
                below += 0.5
        out[u].pct = below / Float64(len(ctrl))
    var keys = List[Int]()
    for u in range(len(out)):
        keys.append(key_of(Int(out[u].mass * 1000000.0), len(out) - 1 - u))
    sort(keys)
    var res = List[Unit]()
    for j in range(len(keys) - 1, -1, -1):
        res.append(out[len(out) - 1 - keys[j] % 1048576].copy())
    return res^


def rank_file(path: String) raises -> List[Unit]:
    with open(path, "r") as f:
        return rank_text(f.read())
