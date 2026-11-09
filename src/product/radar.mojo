"""Anomaly radar (F9): cluster <rlog dir>/anomalies.jsonl by tool, kind, unit (layer/neuron from args) and time (gap > gap_s starts a new
cluster); a cluster has >= 2 events, singletons are counted as isolated. `nx radar {file?, gap_s?, week?, report?, brain?}`: week is ISO
`YYYY-Www` (default: the week of the latest event); report = path of a markdown digest, brain = a Brain dir (note
`<today>-radar-<week>.md` under <brain>/mojo-neuron/). `-D RADAR_MUT=1` clusters by tool only (the gate must catch it)."""
from std.sys import get_defined_int
from json import Json, parse
from mneuron.rlog import Rlog, sh
from mneuron.store import read_text

comptime MUT = get_defined_int["RADAR_MUT", 0]()


def days(y: Int, m: Int, d: Int) -> Int:
    """Unix day of a civil date (Hinnant)."""
    var yy = y - (1 if m <= 2 else 0)
    var era = yy // 400
    var yoe = yy - era * 400
    var doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    return era * 146097 + yoe * 365 + yoe // 4 - yoe // 100 + doy - 719468


def num(s: Span[Byte, _], a: Int, b: Int) -> Int:
    var v = 0
    for i in range(a, b):
        v = v * 10 + Int(s[i]) - 48
    return v


def epoch(ts: String) raises -> Int:
    """Unix seconds of "YYYY-MM-DDTHH:MM:SS..."."""
    if ts.byte_length() < 19:
        raise Error("bad ts '" + ts + "'")
    var s = ts.as_bytes()
    return days(num(s, 0, 4), num(s, 5, 7), num(s, 8, 10)) * 86400 + num(s, 11, 13) * 3600 + num(s, 14, 16) * 60 + num(s, 17, 19)


def iso_week(t: Int) -> String:
    """ISO week "YYYY-Www" of unix seconds: the week holds its Thursday's year."""
    var d = t // 86400
    var thu = d - (d + 3) % 7 + 3
    var z = thu + 719468
    var era = z // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var y = yoe + era * 400 + (1 if mp >= 10 else 0)
    var w = (thu - days(y, 1, 1)) // 7 + 1
    return String(y) + "-W" + (String("0") if w < 10 else String("")) + String(w)


def unit_of(args: String) -> String:
    try:
        var a = parse(args)
        if a.has("layer"):
            return "L" + String(a.get("layer").int()) + ("N" + String(a.get("neuron").int()) if a.has("neuron") else String(""))
    except:
        pass
    return "-"


struct Ev(Copyable, Movable):
    var key: String
    var tool: String
    var kind: String
    var unit: String
    var t: Int
    var result: String

    def __init__(out self, key: String, tool: String, kind: String, unit: String, t: Int, result: String):
        self.key = key
        self.tool = tool
        self.kind = kind
        self.unit = unit
        self.t = t
        self.result = result


def radar(a: Json) raises -> Json:
    var rl = Rlog("radar", a.dumps(), "")
    var file = a.text("file", rl.dir + "/anomalies.jsonl")
    var gap = a.get("gap_s").int() if a.has("gap_s") else 21600
    var evs = List[Ev]()
    for line in read_text(file).split("\n"):
        if line.byte_length() < 2:
            continue
        var j = parse(String(line))
        var tool = j.text("tool")
        var kind = j.text("kind")
        var unit = unit_of(j.text("args"))
        var key = tool if MUT == 1 else tool + "|" + kind + "|" + unit
        evs.append(Ev(key, tool, kind, unit, epoch(j.text("ts")), j.text("result")))
    if len(evs) == 0:
        raise Error("no events in " + file)
    var last = evs[0].t
    for e in evs:
        last = max(last, e.t)
    var week = a.text("week", iso_week(last))
    var mine = List[Int]()  # indices of this week's events
    for i in range(len(evs)):
        if iso_week(evs[i].t) == week:
            mine.append(i)
    # sort by (key, t), insertion sort (weekly volumes are small)
    for i in range(1, len(mine)):
        var x = mine[i]
        var j = i - 1
        while j >= 0 and (evs[mine[j]].key > evs[x].key or (evs[mine[j]].key == evs[x].key and evs[mine[j]].t > evs[x].t)):
            mine[j + 1] = mine[j]
            j -= 1
        mine[j + 1] = x
    var out = Json.array()
    var isolated = 0
    var i = 0
    while i < len(mine):
        var j = i
        while j + 1 < len(mine) and evs[mine[j + 1]].key == evs[mine[i]].key and evs[mine[j + 1]].t - evs[mine[j]].t <= gap:
            j += 1
        if j > i:
            var c = Json.object()
            c.set("tool", Json.of(evs[mine[i]].tool))
            c.set("kind", Json.of(evs[mine[i]].kind))
            c.set("unit", Json.of(evs[mine[i]].unit))
            c.set("n", Json.of(j - i + 1))
            c.set("first", Json.of(sh("date -u -d @" + String(evs[mine[i]].t) + " +%FT%TZ")))
            c.set("last", Json.of(sh("date -u -d @" + String(evs[mine[j]].t) + " +%FT%TZ")))
            c.set("sample", Json.of(evs[mine[i]].result))
            out.push(c)
        else:
            isolated += 1
        i = j + 1
    var res = Json.object()
    res.set("week", Json.of(week))
    res.set("events", Json.of(len(mine)))
    res.set("isolated", Json.of(isolated))
    res.set("clusters", out)
    var md = String("# Anomaly radar ") + week + "\n\n" + String(len(mine)) + " events, " + String(out.len()) + " clusters, " + String(isolated) + " isolated (gap " + String(gap) + " s)\n\n| tool | kind | unit | n | first | last | sample |\n|---|---|---|---:|---|---|---|\n"
    for c in out.items():
        md += "| " + c.text("tool") + " | " + c.text("kind") + " | " + c.text("unit") + " | " + String(c.get("n").int()) + " | " + c.text("first") + " | " + c.text("last") + " | " + c.text("sample").replace("|", "\\|") + " |\n"
    if a.has("report"):
        with open(a.text("report"), "w") as f:
            f.write(md)
    if a.has("brain"):
        var d = a.text("brain") + "/mojo-neuron"
        _ = sh("mkdir -p '" + d + "'")
        with open(d + "/" + sh("date -u +%F") + "-radar-" + week + ".md", "w") as f:
            f.write(md + "\nSource: " + file + "\n")
    var vals = List[Float64]()
    vals.append(Float64(out.len()))
    rl.done("week=" + week + " events=" + String(len(mine)) + " clusters=" + String(out.len()), "", vals, 0.0, 1e9)
    return res^
