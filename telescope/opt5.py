"""Maximise raw expected score (rest limit applies to REST rows in the file only) in the history-aware sim."""
import pickle, sys, time, numpy as np, warnings
from sim import FIRST, LAST
warnings.filterwarnings('ignore')
N = 400
tid = int(sys.argv[1])
t = pickle.load(open('tests2.pkl', 'rb'))[tid]


def obj(plan, n=N, seed=1):
    if sum(a < 0 for a in plan.values()) > 45:
        return -1e9
    return t.simulate(plan, n, seed, return_all=True)[0].mean()


def greedy(seed=5):
    plan = {st: -1 for st in range(FIRST, LAST + 1)}
    for st in range(FIRST, LAST + 1):
        best = None
        for a in [-1] + list(range(len(t.opts[st]))):
            plan[st] = a
            v = t.simulate(plan, 300, seed, return_all=True, last=min(st + 3, LAST))[0].mean()
            if best is None or v > best[0] + 1e-9:
                best = (v, a)
        plan[st] = best[1]
    return plan


from sim2 import SFH
z = np.zeros(1)


def pcool(st, i):
    row = t.static[(st, i)]
    dyn = {'temperature': z + 24, 'dT_amb': z + 24 - row.ambient_temperature, 'obs_l1': z, 'obs_l2': z, 'obs_l3': z,
           'cls_l1': z + 3, 'cls_l2': z + 3, 'cls_l3': z + 3, 'dT_l1': z, 'dT_l2': z, 'obs_last5': z,
           'streak_prev': z, 'mag_l1': z + np.nan}
    return t.sm.predict(t._mat(row, SFH, dyn, 1))[0]


def a_first(fill, thr):
    plan, nrest = {}, 0
    for st in range(FIRST, LAST + 1):
        o = t.opts[st]
        ps = [pcool(st, i) for i in range(len(o))]
        A = [(ps[i], i) for i, x in enumerate(o) if x[1][0] == 'A']
        if A and max(A)[0] > thr:
            plan[st] = max(A)[1]
        elif fill == 'REST' and nrest < 45:
            plan[st] = -1
            nrest += 1
        else:
            ev = [(ps[i] * x[2], i) for i, x in enumerate(o)]
            plan[st] = max(ev)[1] if ev else -1
    return plan


starts = {f'A_{f}_{th}': a_first(f, th) for f in ('REST', 'EV') for th in (0.2, 0.3, 0.4)}
if len(sys.argv) > 2:
    starts['greedy'] = greedy()
for f in [f'plan2_{tid}.pkl', f'plan4_{tid}.pkl', f'plan5_{tid}.pkl']:
    try:
        starts[f] = pickle.load(open(f, 'rb'))
    except FileNotFoundError:
        pass
for k, p in starts.items():
    print(k, round(obj(p, 1000, 99), 2), flush=True)
k = max(starts, key=lambda k: obj(starts[k], 1000, 99))
plan = dict(starts[k])
best = obj(plan)
print('start from', k, best, flush=True)
for sw in range(8):
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
    pickle.dump(plan, open(f'plan5_{tid}.pkl', 'wb'))
    print(f'sweep {sw} obj {best:.2f} holdout {obj(plan, 2000, 99):.2f} {time.strftime("%X")}', flush=True)
    if not imp:
        break
