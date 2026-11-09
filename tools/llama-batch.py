"""llama-batch GGUF TASKS.json: llama.cpp (CPU) next-token logits at the last prompt position for many tasks, model loaded once.

TASKS.json = [{"ids": "ids.txt", "out": "out.f32", "vec": "optional f32[n_layer * hidden] control vector file"}, ...]
Oracle only, same semantics as tools/llama-oracle.py."""
import sys, json, ctypes
import numpy as np
import llama_cpp as lc

gguf, tasks = sys.argv[1], json.load(open(sys.argv[2]))
llm = lc.Llama(model_path=gguf, n_ctx=256, n_gpu_layers=0, logits_all=True, verbose=False, n_threads=8)
nl, nh = lc.llama_model_n_layer(llm._model.model), llm.n_embd()
for t in tasks:
    ids = [int(x) for x in open(t["ids"]).read().split()]
    llm.reset()
    if t.get("vec"):
        v = np.fromfile(t["vec"], dtype=np.float32)
        assert v.size == nl * nh, f"vec must be f32[{nl} * {nh}], got {v.size}"
        buf = np.ascontiguousarray(v[nh:])
        assert lc.llama_set_adapter_cvec(llm.ctx, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buf.size, nh, 1, nl - 1) == 0
    else:
        lc.llama_set_adapter_cvec(llm.ctx, None, 0, nh, 0, 0)
    llm.eval(ids)
    np.asarray(llm.scores[len(ids) - 1], dtype=np.float32).tofile(t["out"])
print("ok", len(tasks))
