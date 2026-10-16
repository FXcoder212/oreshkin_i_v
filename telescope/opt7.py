"""Final optimisation under the leaderboard-calibrated model:
scoring stops once realised rests exceed 45, class-A success logit shifted by -1 and -2."""
import pickle, sys, time, numpy as np, pandas as pd, warnings
from sim import FIRST, LAST
from shiftA import Shift
warnings.filterwarnings('ignore')
tid = int(sys.argv[1])
N = 500
t = pickle.load(open('tests2.pkl', 'rb'))[tid]
tb = pickle.load(open('tests2b.pkl', 'rb'))[tid]
base, baseb = t.sm, tb.sm
SC = [-1.0, -2.0]


def ev(plan, n, seed, sim=t, b=None):
    if sum(a < 0 for a in plan.values()) > 45:
        return -1e9
    b = b or base
    sim.trunc = True
    vals = []
    for i, d in enumerate(SC):
        sim.sm = Shift(b, d)
        vals.append(sim.simulate(plan, n, seed + i, return_all=True)[0].mean())
    sim.sm = b
    return float(np.mean(vals))


def holdout(plan):
    a, c = ev(plan, 2000, 50_001), ev(plan, 2000, 50_011, tb, baseb)
    return (a + c) / 2, a, c


def plan_of(f):
    g = pd.read_csv(f)
    g = g[g.test_id == tid].set_index('step')
    return {st: -1 if g.loc[st, 'action'] == 'REST' else [o[0] for o in t.opts[st]].index(g.loc[st, 'slot_id'])
            for st in range(FIRST, LAST + 1)}


cands = {}
for f in ['submission_telescope.csv', 'submission_telescope_robust.csv', 'submission_telescope_v3.csv',
          'submission_telescope_v4.csv']:
    cands[f] = plan_of(f)
for f in [f'plan2_{tid}.pkl', f'plan4_{tid}.pkl', f'plan5_{tid}.pkl', f'plan6_{tid}.pkl', f'robust_{tid}.pkl']:
    try:
        cands[f] = pickle.load(open(f, 'rb'))
    except FileNotFoundError:
        pass
hs = {k: holdout(p) for k, p in cands.items()}
for k, h in hs.items():
    print('cand', k, [round(x, 1) for x in h], flush=True)
k = max(hs, key=lambda k: hs[k][0])
plan, best_h = dict(cands[k]), hs[k]
best_plan = dict(plan)
pickle.dump(best_plan, open(f'plan7_{tid}.pkl', 'wb'))
print('start', k, flush=True)
opts = {st: [-1] + list(range(len(t.opts[st]))) for st in range(FIRST, LAST + 1)}
for sw in range(30):
    seed = 2000 + 10 * sw
    cur = ev(plan, N, seed)
    imp = 0
    for st in range(FIRST, LAST + 1):
        keep = plan[st]
        for a in opts[st]:
            if a == keep:
                continue
            plan[st] = a
            v = ev(plan, N, seed)
            if v > cur + 1e-9:
                cur, keep, imp = v, a, imp + 1
        plan[st] = keep
    h = holdout(plan)
    if h[0] > best_h[0]:
        best_h, best_plan = h, dict(plan)
        pickle.dump(best_plan, open(f'plan7_{tid}.pkl', 'wb'))
    print(f'sweep {sw} moves {imp} obj {cur:.2f} holdout {h[0]:.2f} ({h[1]:.2f}/{h[2]:.2f}) best {best_h[0]:.2f} '
          f'{time.strftime("%X")}', flush=True)
    if imp <= 1:
        break
print('DONE', [round(x, 2) for x in best_h], flush=True)
