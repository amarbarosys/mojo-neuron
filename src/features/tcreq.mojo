"""tcreq MODEL.gguf CORPUS OUT.jsonl TOKENS SEG: engine request lines (serve/PROTOCOL.md) for a capture run: one llama-tokenize pass,
windows of SEG tokens, `{"id":i,"prompt":[...],"n":1,"spec":false}` per window, TOKENS tokens in total. Feed to the engine with
BARO_SERVE=1 BARO_CAPTURE=DIR on stdin (scripts/capture-qwythos.sh)."""
from std.sys import argv, exit
from mneuron.ggio import llama_ids, write_file
from mneuron.store import read_text


def main() raises:
    var a = argv()
    if len(a) < 6:
        print("usage: tcreq MODEL.gguf CORPUS OUT.jsonl TOKENS SEG")
        exit(1)
    var ids = llama_ids(String(a[1]), read_text(String(a[2])), String(a[3]) + ".tmp")
    var seg = Int(String(a[5]))
    var nw = min(Int(String(a[4])) // seg, len(ids) // seg)
    var s = String()
    for w in range(nw):
        s += '{"id":' + String(w + 1) + ',"prompt":['
        for j in range(seg):
            s += (String(ids[w * seg + j]) + ("," if j + 1 < seg else ""))
        s += '],"n":1,"spec":false}\n'
    var b = List[UInt8]()
    for c in s.as_bytes():
        b.append(c)
    write_file(String(a[3]), b)
    print("windows", nw, "tokens", nw * seg, "corpus tokens", len(ids))
