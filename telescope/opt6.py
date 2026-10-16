"""Thorough raw-score optimisation: fresh CRN seed per sweep, single + adjacent-pair moves,
best plan chosen by holdout on the main and an independently trained simulator."""
import pickle, sys, time, itertools, numpy as np, warnings
from sim import FIRST, LAST
warnings.filterwarnings('ignore')
tid = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
t = pickle.load(open('tests2.pkl', 'rb'))[tid]
tb = pickle.load(open('tests2b.pkl', 'rb'))[tid]


def nrest(plan):
    return sum(a < 0 for a in plan.values())


def ev(sim, plan, n, seed):
    if nrest(plan) > 45:
        return -1e9
    return sim.simulate(plan, n, seed, return_all=True)[0].mean()


def holdout(plan):
    a, b = ev(t, plan, 3000, 10_001), ev(tb, plan, 3000, 10_002)
    return (a + b) / 2, a, b


starts = []
for f in [f'plan5_{tid}.pkl', f'plan6_{tid}.pkl', f'plan4_{tid}.pkl', f'plan2_{tid}.pkl']:
    try:
        starts.append(pickle.load(open(f, 'rb')))
    except FileNotFoundError:
        pass
plan = max(starts, key=lambda p: holdout(p)[0])
best_h = holdout(plan)
best_plan = dict(plan)
print('start holdout', [round(x, 2) for x in best_h], flush=True)

opts = {st: [-1] + list(range(len(t.opts[st]))) for st in range(FIRST, LAST + 1)}
for sw in range(40):
    seed = 1000 + sw
    cur = ev(t, plan, N, seed)
    imp = 0
    # single moves
    for st in range(FIRST, LAST + 1):
        keep = plan[st]
        for a in opts[st]:
            if a == keep:
                continue
            plan[st] = a
            v = ev(t, plan, N, seed)
            if v > cur + 1e-9:
                cur, keep, imp = v, a, imp + 1
        plan[st] = keep
    # adjacent pair moves (every other sweep, they are costly)
    if sw % 2 == 1:
        for st in range(FIRST, LAST):
            k1, k2 = plan[st], plan[st + 1]
            for a, b in itertools.product(opts[st], opts[st + 1]):
                if a == k1 or b == k2:
                    continue
                plan[st], plan[st + 1] = a, b
                v = ev(t, plan, N, seed)
                if v > cur + 1e-9:
                    cur, k1, k2, imp = v, a, b, imp + 1
            plan[st], plan[st + 1] = k1, k2
    h = holdout(plan)
    if h[0] > best_h[0]:
        best_h, best_plan = h, dict(plan)
        pickle.dump(best_plan, open(f'plan6_{tid}.pkl', 'wb'))
    print(f'sweep {sw} moves {imp} obj {cur:.2f} holdout {h[0]:.2f} ({h[1]:.2f}/{h[2]:.2f}) '
          f'best {best_h[0]:.2f} rests {nrest(plan)} {time.strftime("%X")}', flush=True)
    if imp == 0 and sw % 2 == 1:
        break
pickle.dump(best_plan, open(f'plan6_{tid}.pkl', 'wb'))
print('DONE best', [round(x, 2) for x in best_h], flush=True)
