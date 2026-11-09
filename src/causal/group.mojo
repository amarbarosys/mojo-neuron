"""Group MODEL.gguf OUTDIR layer= facts=FILE (capture=DIR | groups=FILE) [baro= spark= pack= runner= env= resp=] (plan A2c, docs/prereg-a2c-group.md).

facts: lines `context<TAB>target word<TAB>foil word`; target and foil = first token of " word". capture=DIR: run the contexts as n=1 requests with the A1 capture of field x at the layer
(input of `prerank`). groups=FILE: lines `label<TAB>n1,n2,...<TAB>gain`; every group is ablated at once (engine request field gain "L:N:k,...") on every context in ONE resident engine process,
with an identity-gain base per context (same layer-loop path). Writes OUTDIR/L<layer>.group.tsv: label, prompt, kl, dnll, dld per (group, prompt), where dld is the change of logit(target) - logit(foil) at the answer
position (ablated minus base). The first group listed that is repeated later is a repeat run: callers put the floor groups twice. Ends with rlog.
"""
from std.sys import argv
from std.os import getenv
from std.time import perf_counter_ns
from tokenizer import Tokenizer
from sweep import opt, read_text, num, at, run_spark
from mneuron.rlog import Rlog


def main() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    if len(a) < 3:
        raise Error("usage: group MODEL.gguf OUTDIR layer= facts= (capture=DIR | groups=FILE)")
    var odir = a[2] if a[2].startswith("/") else getenv("PWD") + "/" + a[2]
    var layer = Int(opt(a, "layer", "0"))
    var ls = String(layer)
    var tok = Tokenizer(a[1])
    var ctx = List[String]()
    var tgt = List[Int]()
    var foil = List[Int]()
    for l in read_text(opt(a, "facts", "")).split("\n"):
        if l.byte_length() == 0:
            continue
        var p = l.split("\t")
        if len(p) != 3:
            raise Error("facts line needs context, target, foil: " + String(l))
        var ids = tok.encode(String(p[0]))
        var s = String()
        for t in range(len(ids)):
            s += ("," if t > 0 else "") + String(ids[t])
        ctx.append(s)
        tgt.append(tok.encode(String(" ") + String(p[1]), add_special=False)[0])
        foil.append(tok.encode(String(" ") + String(p[2]), add_special=False)[0])
    var np = len(ctx)
    var capdir = opt(a, "capture", "")
    var reqs = List[String]()
    if capdir != "":
        for pi in range(np):
            reqs.append("{\"id\":" + String(pi + 1) + ",\"prompt\":[" + ctx[pi] + "],\"n\":1}")
        var lines = run_spark(reqs, a, odir, "BARO_CAPTURE='" + capdir + "' BARO_CAPTURE_LAYERS=" + ls + " BARO_CAPTURE_FIELDS=x")
        var nd = 0
        for l in lines:
            if l.find("\"done\"") >= 0:
                nd += 1
        if nd != np:
            raise Error("capture run: " + String(nd) + " done lines for " + String(np) + " requests")
        print("group: captured", np, "contexts into", capdir)
        return
    var labels = List[String]()
    var gains = List[String]()
    for l in read_text(opt(a, "groups", "")).split("\n"):
        if l.byte_length() == 0:
            continue
        var p = l.split("\t")
        if len(p) != 3:
            raise Error("groups line needs label, neurons, gain: " + String(l))
        labels.append(String(p[0]))
        var gs = String()
        var first = True
        for n in p[1].split(","):
            gs += ("" if first else ",") + ls + ":" + String(n) + ":" + String(p[2])
            first = False
        gains.append(gs)
    var ng = len(labels)
    for pi in range(np):
        reqs.append("{\"id\":" + String(pi + 1) + ",\"prompt\":[" + ctx[pi] + "],\"n\":1,\"slens_ctx\":" + String(pi) + ",\"slens_k\":0,\"gain\":\"" + ls + ":0:1\"}")
    for gi in range(ng):
        for pi in range(np):
            reqs.append("{\"id\":" + String(np + gi * np + pi + 1) + ",\"prompt\":[" + ctx[pi] + "],\"n\":1,\"slens_ctx\":" + String(pi) + ",\"slens_k\":1,\"slens_tgt\":" + String(tgt[pi])
                + ",\"slens_foil\":" + String(foil[pi]) + ",\"slens_flags\":3,\"gain\":\"" + gains[gi] + "\"}")
    print("group: mode = zero/gain per group, clean prompts only, metric = logit(target) - logit(foil) change (plus KL, dNLL);", ng, "groups,", np, "facts,", len(reqs), "requests")
    var t0 = perf_counter_ns()
    var rfile = opt(a, "resp", "")
    var lines = List[String]()
    if rfile != "":
        for l in read_text(rfile).split("\n"):
            lines.append(String(l))
    else:
        lines = run_spark(reqs, a, odir)
    var kl = List[Float64](length=ng * np, fill=0)
    var dn = List[Float64](length=ng * np, fill=0)
    var dl = List[Float64](length=ng * np, fill=0)
    var seen = List[Bool](length=ng * np, fill=False)
    var ndone = 0
    for l in lines:
        if not l.startswith("{\"id\":"):
            continue
        var i = 0
        _ = at(l, "\"id\":", i)
        var id = Int(num(l.as_bytes(), i)) - 1
        if l.find("\"error\"") >= 0:
            raise Error("engine request " + String(id + 1) + ": " + String(l))
        if l.find("\"done\"") >= 0:
            ndone += 1
        if l.find("\"slens\"") >= 0:
            var u = id - np
            if u < 0 or u >= ng * np or seen[u]:
                raise Error("unexpected slens line: " + String(l))
            var j = 0
            var b = l.as_bytes()
            if not at(l, "\"kl\":", j):
                raise Error("malformed slens line: " + String(l))
            kl[u] = num(b, j)
            if not at(l, "\"dnll\":", j):
                raise Error("slens line without dnll: " + String(l))
            dn[u] = num(b, j)
            if not at(l, "\"dld\":", j):
                raise Error("slens line without dld (engine older than the slens_foil commit): " + String(l))
            dl[u] = num(b, j)
            seen[u] = True
    if ndone != len(reqs):
        raise Error("response incomplete: " + String(ndone) + " done lines for " + String(len(reqs)) + " requests")
    var tsv = String("label\tprompt\tkl\tdnll\tdld\n")
    var vals = List[Float64]()
    for gi in range(ng):
        for pi in range(np):
            var u = gi * np + pi
            if not seen[u]:
                raise Error("group " + labels[gi] + " prompt " + String(pi) + " has no slens line")
            tsv += labels[gi] + "\t" + String(pi) + "\t" + String(kl[u]) + "\t" + String(dn[u]) + "\t" + String(dl[u]) + "\n"
            vals.append(kl[u])
            vals.append(dl[u])
    var path = odir + "/L" + ls + ".group.tsv"
    with open(path, "w") as f:
        f.write(tsv)
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    print("group: wrote", path, "wall", wall, "s")
    var rl = Rlog("group", "layer=" + ls + " groups=" + String(ng) + " facts=" + String(np) + " mode=zero/gain metric=logit-diff,KL,dNLL corrupt=none", a[1])
    rl.done("groups=" + String(ng) + " facts=" + String(np), path, vals, -200.0, 200.0)
