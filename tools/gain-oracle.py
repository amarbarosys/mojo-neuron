"""gain-oracle HF_DIR IDS.txt OUT.f32 [L:N:k,...]: next-token logits at the last prompt position of the HF fp32 model,
neuron gains applied to the input of mlp.down_proj (every position). CPU only."""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

d, ids_p, out = sys.argv[1:4]
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
ids = torch.tensor([[int(t) for t in open(ids_p).read().split()]])
if len(sys.argv) > 4:
    for u in sys.argv[4].split(","):
        L, n, k = u.split(":")
        def hook(mod, args, n=int(n), k=float(k)):
            x = args[0].clone()
            x[..., n] *= k
            return (x,)
        m.model.layers[int(L)].mlp.down_proj.register_forward_pre_hook(hook)
with torch.inference_mode():
    m(ids).logits[0, -1].numpy().astype("f4").tofile(out)
print("ok", out)
