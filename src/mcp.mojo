"""MCP server on stdio (newline-delimited JSON-RPC 2.0): initialize, ping, tools/list, tools/call. Each tool is one call of
mneuron/calls.mojo, so CLI, app and MCP answer from the same code. Errors of a call come back as isError results, not protocol errors."""
from mneuron.stdin import read_line
from json import Json, parse
from mneuron.calls import call

comptime S_MODEL = '"model":{"type":"string","description":"model id as listed by models"}'
comptime S_LAYER = '"layer":{"type":"integer","description":"0-based layer"}'
comptime S_NEURON = '"neuron":{"type":"integer"}'


def tool(name: String, desc: String, props: String, req: String) raises -> Json:
    return parse('{"name":"' + name + '","description":"' + desc + '","inputSchema":{"type":"object","properties":{' + props + '},"required":[' + req + ']}}')


comptime SUPPORTED = ["2024-11-05", "2025-03-26", "2025-06-18"]  # tools/list and tools/call only: identical in all three


def tools() raises -> Json:
    var a = Json.array()
    a.push(tool("models", "List explored models with arch, layer count, FFN width and the layers that have a lens.", "", ""))
    a.push(tool("lens", "Lens of one neuron: tokens it promotes and suppresses. kind r reads the write direction through the fitted ridge map (R-style lens) when rlens.bin exists.", S_MODEL + "," + S_LAYER + "," + S_NEURON + ',"kind":{"type":"string","enum":["logit","r"]}', '"model","layer","neuron"'))
    a.push(tool("search_lens", "Neurons whose lens top or bottom tokens contain a token.", S_MODEL + "," + S_LAYER + ',"token":{"type":"string"},"k":{"type":"integer"}', '"model","layer","token"'))
    a.push(tool("rank", "Neurons of a layer ranked by word-cluster mass of their steered rollouts against random controls (needs the engine sweep).", S_MODEL + "," + S_LAYER + ',"k":{"type":"integer"}', '"model","layer"'))
    a.push(tool("connections", "Weight-cosine connections upstream and downstream of a neuron.", S_MODEL + "," + S_LAYER + "," + S_NEURON + ',"span":{"type":"integer"},"k":{"type":"integer"}', '"model","layer","neuron"'))
    a.push(tool("fires", "Run text through the model on CPU and return the neurons of a layer that fire most, or one neuron's activation per token.", S_MODEL + "," + S_LAYER + ',"text":{"type":"string"},"neuron":{"type":"integer"},"k":{"type":"integer"}', '"model","layer","text"'))
    a.push(tool("dossier", "Everything known about a neuron (lens, connections, rank, rollouts, optional fires); also written to the store.", S_MODEL + "," + S_LAYER + "," + S_NEURON + ',"text":{"type":"string"}', '"model","layer","neuron"'))
    a.push(tool("recipe", "Validate and normalise a steering recipe L:N:dose,L:N:dose.", S_MODEL + ',"recipe":{"type":"string"}', '"model","recipe"'))
    a.push(tool("export_brain", "Write a neuron as a note of the Brain network and index it, so brain-ask finds it.", S_MODEL + "," + S_LAYER + "," + S_NEURON, '"model","layer","neuron"'))
    return a^


def reply(id: Json, result: Json):
    var o = Json.object()
    o.set("jsonrpc", Json.of(String("2.0")))
    o.set("id", id)
    o.set("result", result)
    print(o.dumps())


def fail(id: Json, code: Int, msg: String):
    var e = Json.object()
    e.set("code", Json.of(code))
    e.set("message", Json.of(msg))
    var o = Json.object()
    o.set("jsonrpc", Json.of(String("2.0")))
    o.set("id", id)
    o.set("error", e)
    print(o.dumps())


def handle(line: String) raises:
    var m = parse(line)
    var method = m.text("method")
    var id = m.get("id")
    if not m.has("id"):  # a notification: no answer
        return
    if method == "initialize":
        var r = Json.object()
        var want = m.get("params").text("protocolVersion", "")
        var ver = String("2025-06-18")  # newest we speak; a client asking for anything else is told this one and may disconnect
        var sup = materialize[SUPPORTED]()
        for i in range(len(sup)):
            if want == sup[i]:
                ver = want
        r.set("protocolVersion", Json.of(ver))
        var caps = Json.object()
        caps.set("tools", Json.object())
        r.set("capabilities", caps)
        var info = Json.object()
        info.set("name", Json.of(String("mojo-neuron")))
        info.set("version", Json.of(String("0.1.0")))
        r.set("serverInfo", info)
        reply(id, r)
    elif method == "ping":
        reply(id, Json.object())
    elif method == "tools/list":
        var r = Json.object()
        r.set("tools", tools())
        reply(id, r)
    elif method == "tools/call":
        var p = m.get("params")
        var r = Json.object()
        var content = Json.array()
        var item = Json.object()
        item.set("type", Json.of(String("text")))
        try:
            if p.text("name") == "live":
                raise Error("live is a GPU job and is not offered over MCP")
            var args = p.get("arguments")
            if p.text("name") == "export_brain" and not args.has("sync"):
                args.set("sync", Json.of(1))  # the tool says it indexes the note
            item.set("text", Json.of(call(p.text("name"), args).dumps()))
            r.set("isError", Json.of(False))
        except e:
            item.set("text", Json.of(String(e)))
            r.set("isError", Json.of(True))
        content.push(item)
        r.set("content", content)
        reply(id, r)
    else:
        fail(id, -32601, "method not found: " + method)


def main() raises:
    var eof = False
    while not eof:
        var line = read_line(eof)
        if line.byte_length() == 0:
            continue
        try:
            handle(line)
        except e:
            fail(Json(), -32700, "parse error: " + String(e))
