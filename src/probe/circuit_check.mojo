"""Gate helper for tests/circuit.sh: circuit_check PIN VOCAB MEASURED -> prints PASS or FAIL for the verdict, or `REFUSED: reason`."""
from std.sys import argv
from probe.circuit import load_pin, passes


def main() raises:
    var a = argv()
    try:
        var p = load_pin(String(a[1]), Int(String(a[2])))
        print("PASS" if passes(p, Float64(String(a[3]))) else "FAIL")
    except e:
        print("REFUSED:", e)
