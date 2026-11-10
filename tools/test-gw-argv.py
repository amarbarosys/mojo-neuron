"""Test: the gpu-wait argv the harness builds for an engine job (no launch). VRAM_GB unset -> --shared --vram 10; VRAM_GB=7 -> --vram 7. Run: python3 -I tools/test-gw-argv.py"""
import importlib.util, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
def load():
    sys.argv = ["h", tempfile.mkdtemp(), "none"]; os.environ["DRY"] = "1"
    sp = importlib.util.spec_from_file_location("ih", f"{HERE}/implants-harness.py"); ih = importlib.util.module_from_spec(sp); sp.loader.exec_module(ih); return ih
ih = load(); bad = []
def check(name, argv, want):
    if argv != want: bad.append(f"{name}: got {argv}, want {want}")
os.environ.pop("VRAM_GB", None)
check("default", ih.gw_argv("/d", 99), ["gpu-wait", "run", "--shared", "--vram", "10", "--timeout", "99", "--", "/d/run.sh"])
os.environ["VRAM_GB"] = "7"
check("VRAM_GB=7", ih.gw_argv("/d", 99), ["gpu-wait", "run", "--shared", "--vram", "7", "--timeout", "99", "--", "/d/run.sh"])
if bad: print("FAIL test-gw-argv: " + "; ".join(bad)); sys.exit(1)
print("PASS test-gw-argv: default --shared --vram 10, VRAM_GB=7 -> --vram 7")
