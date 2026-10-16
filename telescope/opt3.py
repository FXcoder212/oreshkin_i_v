"""Constructive greedy (prefix simulation with rest penalty) + coordinate ascent in history-aware sim."""
import pickle, sys, time, numpy as np, warnings
from sim import FIRST, LAST
from opt2 import ascent
warnings.filterwarnings('ignore')
NG = 300


def greedy(t, lam, seed=5):
    plan = {st: -1 for st in range(FIRST, LAST + 1)}
    for st in range(FIRST, LAST + 1):
        best = None
        for a in [-1] + list(range(len(t.opts[st]))):
            plan[st] = a
            sc, rs = t.simulate(plan, NG, seed, return_all=True, last=st)
            v = sc.mean() - lam * rs.mean() - 200 * np.mean(rs > 45)
            if best is None or v > best[0]:
                best = (v, a)
        plan[st] = best[1]
    return plan


if __name__ == '__main__':
    tid = int(sys.argv[1])
    t = pickle.load(open('tests2.pkl', 'rb'))[tid]
    cands = []
    for lam in [float(x) for x in sys.argv[2:]]:
        p = greedy(t, lam)
        sc, rs = t.simulate(p, 2000, 99, return_all=True)
        v = np.mean(sc * (rs <= 45))
        print(tid, 'lam', lam, 'obj', round(v, 2), 'P>45', np.mean(rs > 45), 'rests med', np.median(rs), time.strftime('%X'), flush=True)
        cands.append((v, p))
    v, p = max(cands, key=lambda x: x[0])
    pickle.dump(p, open(f'greedy3_{tid}.pkl', 'wb'))
    p, _ = ascent(t, p, 1, 6, sys.stdout)
    pickle.dump(p, open(f'plan3_{tid}.pkl', 'wb'))
    sc, rs = t.simulate(p, 3000, 777, return_all=True)
    print(tid, 'FINAL holdout', np.mean(sc * (rs <= 45)), 'P>45', np.mean(rs > 45), flush=True)
