"""Soft objective E[score] - k*E[excess rests] in the history-aware sim, start from max-p plan."""
import pickle, sys, time, numpy as np, warnings
from sim import FIRST, LAST
from sim2 import SFH
warnings.filterwarnings('ignore')
N = 400
tid = int(sys.argv[1])
K = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
t = pickle.load(open('tests2.pkl', 'rb'))[tid]


def obj(plan, n=N, seed=1):
    sc, rs = t.simulate(plan, n, seed, return_all=True)
    return sc.mean() - K * np.maximum(rs - 45, 0).mean()


plan = {}
z = np.zeros(1)
for st in range(FIRST, LAST + 1):
    best = None
    for i, o in enumerate(t.opts[st]):
        row = t.static[(st, i)]
        dyn = {'temperature': z + 28, 'dT_amb': z + 28 - row.ambient_temperature, 'obs_l1': z + 1, 'obs_l2': z + 1,
               'obs_l3': z + 1, 'cls_l1': z + 2, 'cls_l2': z + 2, 'cls_l3': z + 2, 'dT_l1': z, 'dT_l2': z,
               'obs_last5': z + 5, 'streak_prev': z + 5, 'mag_l1': z + 3}
        p = t.sm.predict(t._mat(row, SFH, dyn, 1))[0]
        if best is None or p > best[0]:
            best = (p, i)
    plan[st] = best[1] if best else -1
best = obj(plan)
print('start', best, flush=True)
for sw in range(6):
    imp = False
    for st in range(FIRST, LAST + 1):
        cur = plan[st]
        for a in [-1] + list(range(len(t.opts[st]))):
            if a == cur:
                continue
            plan[st] = a
            v = obj(plan)
            if v > best + 1e-9:
                best, cur, imp = v, a, True
        plan[st] = cur
    sc, rs = t.simulate(plan, 2000, 99, return_all=True)
    print(f'sweep {sw} obj {best:.2f} holdout raw {sc.mean():.1f} capped {np.mean(sc * (rs <= 45)):.1f} '
          f'rests med {np.median(rs)} P<=45 {np.mean(rs <= 45):.3f} {time.strftime("%X")}', flush=True)
    pickle.dump(plan, open(f'plan4_{tid}.pkl', 'wb'))
    if not imp:
        break
