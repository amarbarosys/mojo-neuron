"""Research log (plan section D): every finished op appends one JSON line (mlog) to <rlog dir>/<date>.jsonl, anomalies also to
anomalies.jsonl. Dir: $NX_RLOG, default ~/Projects/mojo/mojo-neuron/.work/rlog; NX_RLOG=off disables.
Line = ts, level, msg="event", run, tool, args, model, msha (sha256 first 16, cached in sha.tsv by path+size+mtime), git, kind, result, receipt.
Use: `var rl = Rlog("lens", args, model)`, then at the end `rl.done(result, receipt, values, lo, hi)`: NaN/Inf or a value outside
[lo, hi] logs kind=anomaly with the reason, else normal. `rl.event(kind, ...)` for anomalies a caller detects itself (parity miss, missed bar).
`-D RLOG_MUT=1` makes `verdict` blind (the lane tests must catch it)."""
from std.math import isfinite
from std.os import getenv
from std.subprocess import run
from std.sys import get_defined_int
from mlog import Logger, Field, INFO

comptime MUT = get_defined_int["RLOG_MUT", 0]()


def sh(cmd: String) raises -> String:
    return String(String(run(cmd)).strip())


def verdict(vals: List[Float64], lo: Float64, hi: Float64) -> String:
    """"" when every value is finite and within [lo, hi], else the first reason."""
    if MUT == 1:
        return String()
    for i in range(len(vals)):
        if not isfinite(vals[i]):
            return "non-finite value at " + String(i)
        if vals[i] < lo or vals[i] > hi:
            return "value " + String(vals[i]) + " at " + String(i) + " outside [" + String(lo) + ", " + String(hi) + "]"
    return String()


struct Rlog(Movable):
    var tool: String
    var args: String
    var model: String
    var msha: String
    var git: String
    var run: String
    var dir: String
    var on: Bool

    def __init__(out self, tool: String, args: String = "", model: String = "") raises:
        self.tool = tool
        self.args = args
        self.model = model
        var d = String(getenv("NX_RLOG"))
        self.on = d != "off"
        self.dir = d if d.byte_length() > 0 else String(getenv("HOME")) + "/Projects/mojo/mojo-neuron/.work/rlog"
        self.msha = String()
        self.git = String()
        self.run = String()
        if self.on:
            _ = sh("mkdir -p '" + self.dir + "'")
            self.git = sh("git rev-parse --short HEAD 2>/dev/null")
            self.run = sh("date -u +%Y%m%dT%H%M%S%N | cut -c1-24")
            if model.byte_length() > 0:
                var key = model + " " + sh("stat -c '%s %Y' '" + model + "'")
                var f = self.dir + "/sha.tsv"
                self.msha = sh("l=$(grep -F -- '" + key + " ' '" + f + "' 2>/dev/null | tail -1); if [ -n \"$l\" ]; then echo \"${l##* }\"; else s=$(sha256sum '" + model + "' | cut -c1-16); printf '%s %s\\n' '" + key + "' \"$s\" >> '" + f + "'; echo \"$s\"; fi")

    def event(self, kind: String, result: String, receipt: String = "") raises:
        if not self.on:
            return
        var day = sh("date -u +%F")
        var paths = List[String]()
        paths.append(self.dir + "/" + day + ".jsonl")
        if kind == "anomaly":
            paths.append(self.dir + "/anomalies.jsonl")
        for p in paths:
            var lg = Logger(INFO, True, p)
            lg.begin(INFO, "event")
            lg.field(Field("run", self.run))
            lg.field(Field("tool", self.tool))
            lg.field(Field("args", self.args))
            lg.field(Field("model", self.model))
            lg.field(Field("msha", self.msha))
            lg.field(Field("git", self.git))
            lg.field(Field("kind", kind))
            lg.field(Field("result", result))
            lg.field(Field("receipt", receipt))
            lg.end()

    def done(self, result: String, receipt: String, vals: List[Float64], lo: Float64 = -1e300, hi: Float64 = 1e300) raises:
        var why = verdict(vals, lo, hi)
        if why.byte_length() > 0:
            self.event("anomaly", result + " | " + why, receipt)
        else:
            self.event("normal", result, receipt)
