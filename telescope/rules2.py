import pickle, numpy as np, warnings
from rules import tests, real, plan_of
from shiftA import Shift
warnings.filterwarnings('ignore')
base = {tid: t.sm for tid, t in tests.items()}
for dA, dB in [(0, 0), (-0.5, 0), (-1, 0), (-1.5, 0), (-1, -0.5), (-2, 0)]:
    row = []
    for f, r in real.items():
        tot = 0
        for tid, t in tests.items():
            t.sm = Shift(base[tid], dA, dB); t.trunc = True
            tot += t.simulate(plan_of(t, f), 1000, 5, return_all=True)[0].mean()
            t.trunc = False; t.sm = base[tid]
        row.append((r, round(tot)))
    err = np.sqrt(np.mean([(a - b) ** 2 for a, b in row]))
    print(dA, dB, row, 'rmse', round(err), flush=True)
