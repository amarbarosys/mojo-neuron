"""Nx CALL [JSON]: run one call (src/mneuron/calls.mojo) and print its JSON. Arguments as one JSON object, or `-` to read stdin."""
from std.sys import argv, exit
from json import Json, parse
from mneuron.calls import call
from chat import chat


def main():
    try:
        var a = argv()
        if len(a) < 2:
            raise Error("usage: nx CALL [JSON]")
        if String(a[1]) == "chat":
            chat()
            return
        var sub = len(a) > 2 and (String(a[1]) == "persona" or String(a[1]) == "replay") and not String(a[2]).startswith("{")  # lane product: `nx persona save JSON`
        var args = parse(String(a[2 + Int(sub)])) if len(a) > 2 + Int(sub) else Json.object()
        if sub:
            args.set("op", Json.of(String(a[2])))
        print(call(String(a[1]), args).dumps(False))
    except e:
        print("FAIL nx:", e)
        exit(1)
