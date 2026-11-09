"""Gate for src/mneuron/rlog.mojo (NX_RLOG set by tests/rlog.sh): one normal op, then a forced NaN."""
from mneuron.rlog import Rlog


def main() raises:
    var rl = Rlog("rlog_check", "{\"layer\":12}", "")
    var ok = List[Float64]()
    ok.append(0.5)
    rl.done("fvu=0.5", "", ok, 0.0, 1.0)
    var bad = List[Float64]()
    bad.append(0.5)
    bad.append(Float64.MAX * 2)  # inf
    rl.done("fvu=nan-forced", "", bad, 0.0, 1.0)
    print("ran")
