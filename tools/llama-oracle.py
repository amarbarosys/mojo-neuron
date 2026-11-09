"""llama-oracle GGUF IDS.txt OUT.f32 [VEC.f32]: next-token logits at the last prompt position from llama.cpp (CPU).

VEC.f32 = f32[n_layer * hidden] (the BARO_STEER file); row L is applied as llama.cpp control vector
`direction.L` (layers >= 1). Oracle only: gates compare mojo-baro against this file."""
import sys
import ctypes
import numpy as np
import llama_cpp as lc

gguf, ids_path, out = sys.argv[1:4]
ids = [int(x) for x in open(ids_path).read().split()]
llm = lc.Llama(model_path=gguf, n_ctx=256, n_gpu_layers=0, logits_all=True, verbose=False, n_threads=8)
nl, nh = lc.llama_model_n_layer(llm._model.model), llm.n_embd()
if len(sys.argv) > 4:
    v = np.fromfile(sys.argv[4], dtype=np.float32)
    assert v.size == nl * nh, f"VEC must be f32[{nl} * {nh}], got {v.size}"
    buf = np.ascontiguousarray(v[nh:])
    r = lc.llama_set_adapter_cvec(llm.ctx, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buf.size, nh, 1, nl - 1)
    assert r == 0, f"llama_set_adapter_cvec failed: {r}"
llm.eval(ids)
np.asarray(llm.scores[len(ids) - 1], dtype=np.float32).tofile(out)
print("ok", out)
