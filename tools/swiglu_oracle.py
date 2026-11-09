"""Oracle: llama.cpp's own ffn_swiglu activations. Usage: swiglu_oracle.py MODEL PROMPT OUTFILE [LLAMA_BIN_DIR]
Writes `ids a,b,c` then per layer and token: `L<layer> t<token> v1..v6` (first and last 3 of the T x ffn tensor) and `L<layer> sum S`."""
import os, re, subprocess, sys
model, prompt, outp = sys.argv[1:4]
bd = sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser('~/Models/llama.cpp/build/bin')
env = dict(os.environ, LD_LIBRARY_PATH=bd)
ids = eval(subprocess.run([f'{bd}/llama-tokenize', '-m', model, '-p', prompt, '--ids'], env=env, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
txt = subprocess.run([f'{bd}/llama-eval-callback', '-m', model, '-p', prompt, '-n', '1', '-ngl', '0'], env=env, capture_output=True, text=True, check=True)
lines = (txt.stdout + txt.stderr).splitlines()
out = ['ids ' + ','.join(map(str, ids))]
i = 0
while i < len(lines):
    m = re.search(r'ffn_swiglu-(\d+) = ', lines[i])
    if m:
        L = m.group(1); t = 0; i += 1
        while 'sum =' not in lines[i]:
            v = re.findall(r'-?\d+\.\d+|nan|inf', lines[i]) if '...' in lines[i] else []
            if len(v) == 6:
                out.append(f'L{L} t{t} ' + ' '.join(v)); t += 1
            i += 1
        out.append(f'L{L} sum ' + lines[i].split('=')[1].strip())
        rows = [r for r in out if r.startswith(f'L{L} t')]
        if len(rows) != len(ids):  # llama.cpp keeps only the output position in the last layer: skip that layer
            out = [r for r in out if not r.startswith(f'L{L} ')]
    i += 1
open(outp, 'w').write('\n'.join(out) + '\n')
