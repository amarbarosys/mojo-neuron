"""Scan MODEL.gguf OUTDIR layer= neurons=LIST|all prompts=FILE [k=gain, default 0] [brain=DIR] [resp=FILE]: ablation scan (plan A2).

Zeroes (gain k) each listed FFN neuron of one layer in ONE resident engine process (engine.mojo, request field gain) and scores the damage
on the eval prompts: every prompt is a context (all tokens but the last) plus a target (its last token); damage = mean KL(base || ablated) of the
next-token distribution and the mean NLL change of the target. Base requests run the same layer-loop path (gain 1 on neuron 0), so a
neuron scored 0 is a no-op to the bit. Output OUTDIR/L<layer>.scan.tsv, ranked by KL (neuron, kl, dnll), a rlog line, optional Brain note.
Noise floor: any change that survives bf16 rounding of the FFN activation re-rolls the rounding of everything downstream and scores KL 7e-6 to 1.2e-5 whatever its size
(k=0.999 rounds back to the same value and scores exactly 0; k=0.99 already sits on the floor). ctl=0.99 runs a second pass at that gain and reports excess = kl - kl_ctl;
the table is ranked by excess when ctl is set: below about 2x the floor a neuron's score is the engine's numeric noise, not its effect.
keys: ctl=GAIN (noise-floor pass) baro=DIR spark=ENGINE pack=DIR runner="gpu-wait run --timeout 3000 --" env="BARO_..=.." resp=FILE (parse a saved response, no GPU)
"""
from std.sys import argv
from std.os import getenv
from std.time import perf_counter_ns
from ggufx.model import Gguf
from tokenizer import Tokenizer
from lens import tensor
from sweep import opt, ints, read_text, num, at, run_spark
from mneuron.rlog import Rlog


def main() raises:
    var a = List[String]()
    for x in argv():
        a.append(String(x))
    if len(a) < 3:
        raise Error("usage: scan MODEL.gguf OUTDIR layer= neurons= prompts= [k=0] [brain=DIR] [resp=FILE]")
    var model = a[1]
    var odir = a[2] if a[2].startswith("/") else getenv("PWD") + "/" + a[2]
    var layer = Int(opt(a, "layer", "0"))
    var gk = opt(a, "k", "0")
    var g = Gguf(model)
    var di = tensor(g, "blk." + String(layer) + ".ffn_down.weight")
    if di < 0:
        raise Error("no blk." + String(layer) + ".ffn_down.weight")
    var neurons = ints(opt(a, "neurons", "0"), g.tensors[di].dims[0])
    var tok = Tokenizer(model)
    var ids = List[List[Int]]()
    for l in read_text(opt(a, "prompts", "")).split("\n"):
        if l.byte_length() > 0:
            ids.append(tok.encode(String(l).replace("\\n", "\n")))
    var np = len(ids)
    if np < 2:
        raise Error("prompts= needs at least 2 lines")
    var ls = String(layer)
    var reqs = List[String]()
    var ctx = List[String]()
    var tgt = List[Int]()
    for p in ids:
        if len(p) < 2:
            raise Error("a prompt has fewer than 2 tokens")
        var s = String()
        for t in range(len(p) - 1):
            s += ("," if t > 0 else "") + String(p[t])
        ctx.append(s)
        tgt.append(p[len(p) - 1])
    for pi in range(np):
        reqs.append("{\"id\":" + String(len(reqs) + 1) + ",\"prompt\":[" + ctx[pi] + "],\"n\":1,\"slens_ctx\":" + String(pi) + ",\"slens_k\":0,\"gain\":\"" + ls + ":0:1\"}")
    for n in neurons:
        for pi in range(np):
            reqs.append("{\"id\":" + String(len(reqs) + 1) + ",\"prompt\":[" + ctx[pi] + "],\"n\":1,\"slens_ctx\":" + String(pi) + ",\"slens_k\":1,\"slens_tgt\":" + String(tgt[pi])
                + ",\"slens_flags\":" + String((1 if pi == 0 else 0) + (2 if pi == np - 1 else 0)) + ",\"gain\":\"" + ls + ":" + String(n) + ":" + gk + "\"}")
    var ctl = opt(a, "ctl", "")
    if ctl != "":
        for n in neurons:
            for pi in range(np):
                reqs.append("{\"id\":" + String(len(reqs) + 1) + ",\"prompt\":[" + ctx[pi] + "],\"n\":1,\"slens_ctx\":" + String(pi) + ",\"slens_k\":1,\"slens_tgt\":" + String(tgt[pi])
                    + ",\"slens_flags\":" + String((1 if pi == 0 else 0) + (2 if pi == np - 1 else 0)) + ",\"gain\":\"" + ls + ":" + String(n) + ":" + ctl + "\"}")
    var mode = (String("zero") if gk == "0" else "gain " + gk) + ", clean prompts only (no corrupted run), metric = KL(base || ablated) + dNLL of the target token"
    print("scan: mode =", mode)
    print("scan:", len(reqs), "requests,", len(neurons), "neurons,", np, "prompts, layer", layer, "gain", gk)
    var t0 = perf_counter_ns()
    var rfile = opt(a, "resp", "")
    var lines = List[String]()
    if rfile != "":
        for l in read_text(rfile).split("\n"):
            lines.append(String(l))
    else:
        lines = run_spark(reqs, a, odir)
    var kl = List[Float64](length=len(neurons), fill=-1)
    var dn = List[Float64](length=len(neurons), fill=0)
    var seen = List[Bool](length=len(neurons), fill=False)
    var klc = List[Float64](length=len(neurons), fill=0)
    var seenc = List[Bool](length=len(neurons), fill=False)
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
            var u = (id - np) // np
            var is_ctl = ctl != "" and u >= len(neurons)
            if is_ctl:
                u -= len(neurons)
            if id < np or u >= len(neurons) or (id - np) % np != np - 1 or (seenc[u] if is_ctl else seen[u]):
                raise Error("unexpected slens line: " + String(l))
            var j = 0
            var b = l.as_bytes()
            if not at(l, "\"kl\":", j):
                raise Error("malformed slens line: " + String(l))
            var klv = num(b, j)
            if not at(l, "\"dnll\":", j):
                raise Error("slens line without dnll (engine older than 69aec4f+): " + String(l))
            if is_ctl:
                klc[u] = klv
                seenc[u] = True
            else:
                kl[u] = klv
                dn[u] = num(b, j)
                seen[u] = True
    var nd2 = len(reqs)
    if ndone != nd2:
        raise Error("response incomplete: " + String(ndone) + " done lines for " + String(len(reqs)) + " requests")
    for u in range(len(neurons)):
        if not seen[u] or (ctl != "" and not seenc[u]):
            raise Error("neuron " + String(neurons[u]) + " has no slens line")
    var order = List[Int]()
    for u in range(len(neurons)):
        order.append(u)
    var sc = List[Float64]()  # ranking score: KL, or the excess over the noise-floor pass
    for u in range(len(neurons)):
        sc.append(kl[u] - klc[u] if ctl != "" else kl[u])
    for x in range(1, len(order)):  # insertion sort, score descending
        var v = order[x]
        var y = x
        while y > 0 and sc[order[y - 1]] < sc[v]:
            order[y] = order[y - 1]
            y -= 1
        order[y] = v
    var tsv = String("neuron\tkl\tdnll\n") if ctl == "" else String("neuron\tkl\tdnll\tkl_ctl\texcess\n")
    for u in order:
        tsv += String(neurons[u]) + "\t" + String(kl[u]) + "\t" + String(dn[u]) + ("\n" if ctl == "" else "\t" + String(klc[u]) + "\t" + String(kl[u] - klc[u]) + "\n")
    var path = odir + "/L" + ls + ".scan.tsv"
    with open(path, "w") as f:
        f.write(tsv)
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    print("scan: wrote", path, "wall", wall, "s,", wall / Float64(len(reqs)) * 1000, "ms/request")
    var mp = model.split("/")
    var stem = String(String(mp[len(mp) - 1]).replace(".gguf", "").lower())
    var bd = opt(a, "brain", "")
    if bd != "":
        var md = String("---\ntype: scan\nmodel: ") + stem + "\nlayer: " + ls + "\ngain: " + gk + "\nprompts: " + String(np) + "\n---\n\n# Ablation scan " + stem + " L" + ls + "\n\nTop 20 of " + String(len(neurons)) + " neurons by KL (" + mode + "; " + String(np) + " prompts). Receipt `" + path + "`.\n\n| neuron | KL | dNLL |\n|---|---|---|\n"
        for r in range(min(20, len(order))):
            md += "| " + String(neurons[order[r]]) + " | " + String(kl[order[r]]) + " | " + String(dn[order[r]]) + " |\n"
        with open(bd + "/" + stem + "-L" + ls + "-scan.md", "w") as f:
            f.write(md)
    var vals = List[Float64]()
    var neg = -1
    for u in range(len(neurons)):
        vals.append(kl[u])
        vals.append(dn[u])  # dNLL may be negative: only finiteness and a +-50 bound apply to it
        if kl[u] < -1e-6:
            neg = neurons[u]
    var rl = Rlog("scan", "layer=" + ls + " neurons=" + String(len(neurons)) + " prompts=" + String(np) + " k=" + gk + " mode=" + (String("zero") if gk == "0" else "gain") + " metric=KL+dNLL(target) corrupt=none", model)
    var res = "top neuron " + String(neurons[order[0]]) + " kl=" + String(kl[order[0]]) + " dnll=" + String(dn[order[0]])
    if neg >= 0:
        rl.event("anomaly", res + " | negative KL at neuron " + String(neg), path)
    else:
        rl.done(res, path, vals, -50.0, 50.0)
