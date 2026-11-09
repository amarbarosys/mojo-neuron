"""Oracle for the data-scaling extrapolation: fit FVU(N) = f + b * N**-c to held-out FVU at several training sizes and report the floor f and the N where it reaches a target. usage: scaling_fit.py TARGET N1:F1 N2:F2 N3:F3"""
import sys, math
import numpy as np
from scipy.optimize import brentq
t = float(sys.argv[1]); pts = [tuple(map(float, a.split(":"))) for a in sys.argv[2:]]
(n1, f1), (n2, f2), (n3, f3) = pts
# three points, three unknowns: solve for c from the ratio of differences, then b and f
def ratio(c): return ((n1**-c) - (n2**-c)) / ((n2**-c) - (n3**-c)) - (f1 - f2) / (f2 - f3)
try:
    c = brentq(ratio, 1e-3, 6.0)
except ValueError:
    raise SystemExit('no power-law fit: the improvement is not slowing down between the points, so there is no floor to extrapolate to')
b = (f1 - f2) / ((n1**-c) - (n2**-c)); f = f1 - b * n1**-c
print(f"c={c:.3f} b={b:.3g} floor f={f:.3f}")
print(f"target {t}: " + (f"N = {(b/(t-f))**(1/c):.3g} tokens" if t > f else f"never reached (floor {f:.3f} >= target)"))
