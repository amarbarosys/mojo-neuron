"""rho HIDDEN OUT.f32 DUMP...: median residual L2 norm per layer from BARO_DUMP_LAYERS files (f32[n_layer * hidden] each)."""
import sys
import numpy as np

h, out = int(sys.argv[1]), sys.argv[2]
d = np.stack([np.fromfile(p, dtype=np.float32).reshape(-1, h) for p in sys.argv[3:]])
np.median(np.linalg.norm(d, axis=2), axis=0).astype(np.float32).tofile(out)
print("rho", out, d.shape[1], "layers from", d.shape[0], "dumps")
