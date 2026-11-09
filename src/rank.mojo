"""Rank SWEEP.jsonl [N]: units ranked by word-cluster mass against `ctrl*` controls. Prints neuron, mass, pct, rows, words, top."""
from std.sys import argv, exit
from mneuron.rankcore import rank_file


def main():
    try:
        var a = argv()
        if len(a) < 2:
            raise Error("usage: rank SWEEP.jsonl [N]")
        var n = Int(String(a[2])) if len(a) > 2 else 1 << 30
        var r = rank_file(String(a[1]))
        for i in range(min(n, len(r))):
            print(r[i].name + "\t" + String(r[i].mass) + "\t" + String(r[i].pct) + "\t" + String(r[i].rows) + "\t" + String(r[i].words) + "\t" + r[i].top)
    except e:
        print("FAIL rank:", e)
        exit(1)
