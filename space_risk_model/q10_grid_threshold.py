# Q10 variant: threshold t on grid 0.00..1.00 step 0.01 (P > t is dangerous).
import sys, itertools, numpy as np, pandas as pd
step = float(sys.argv[1]); only10 = len(sys.argv) > 2 and sys.argv[2] == "10"
part = int(sys.argv[3]) if len(sys.argv) > 3 else 0; nparts = int(sys.argv[4]) if len(sys.argv) > 4 else 1
df = pd.read_csv("space_risk_model.csv")
y = df.failure_in_12h.to_numpy(); g = df.orbit_type.to_numpy()
P = {i: df[f"model_{i}_pred"].to_numpy() for i in range(1, 11)}
T = np.round(np.arange(0, 101) / 100, 2)
masks = [(g == k) for k in range(3)]
N = len(y); npos = y.sum()
units = round(1 / step)
W = [(i / units, j / units, (units - i - j) / units) for i in range(1, units) for j in range(1, units - i)]
tr = [c for c in itertools.combinations(range(1, 11), 3) if (not only10 or 10 in c)][part::nparts]
best = None
for c in tr:
    for w in W:
        p = w[0] * P[c[0]] + w[1] * P[c[1]] + w[2] * P[c[2]]
        FN = np.zeros(len(T), int); FP = np.zeros(len(T), int); ok = np.ones(len(T), bool)
        for m in masks:
            pp = np.sort(p[m & (y == 1)]); pn = np.sort(p[m & (y == 0)])
            fn = np.searchsorted(pp, T, "right"); tp = len(pp) - fn
            fp = len(pn) - np.searchsorted(pn, T, "right")
            ok &= (fn <= 0.15 * len(pp)) & ((tp + fp) <= 0.30 * m.sum())
            FN += fn; FP += fp
        ok &= (FN <= 0.10 * npos) & ((npos - FN + FP) <= 0.25 * N)
        if not ok.any(): continue
        L = np.where(ok, 20_000_000 * FN + 100_000 * FP, np.iinfo(np.int64).max)
        k = int(np.argmin(L))  # first = minimal t
        cand = (int(L[k]), T[k], c, w)
        if best is None or cand[:3] < best[:3]: best = cand
print("BEST", best)
