"""rlogcli TOOL KIND RESULT [RECEIPT] [MODEL]: one research-log line from a shell script (src/mneuron/rlog.mojo)."""
from std.sys import argv
from mneuron.rlog import Rlog


def main() raises:
    var a = argv()
    var rl = Rlog(String(a[1]), "", String(a[5]) if len(a) > 5 else "")
    rl.event(String(a[2]), String(a[3]), String(a[4]) if len(a) > 4 else "")
