"""cvec-check X.cvec.gguf X.f32 N_LAYER HIDDEN: the gguf holds exactly the non-zero layers >= 1 of the f32 file."""
import sys
import numpy as np
from gguf import GGUFReader

path, raw, nl, nh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
v = np.fromfile(raw, dtype=np.float32).reshape(nl, nh)
r = GGUFReader(path)
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
assert arch == "controlvector", arch
assert int(r.fields["controlvector.layer_count"].parts[-1][0]) == nl
seen = set()
for t in r.tensors:
    L = int(t.name.split(".")[1])
    assert t.name == f"direction.{L}" and L >= 1 and t.data.shape == (nh,) and t.tensor_type.name == "F32", t.name
    assert np.array_equal(t.data, v[L]), f"layer {L} differs"
    seen.add(L)
want = {L for L in range(1, nl) if v[L].any()}
assert seen == want, (sorted(seen), sorted(want))
print("PASS cvec-check:", sorted(seen))
