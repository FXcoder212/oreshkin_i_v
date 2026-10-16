import pickle, sys, numpy as np
from sim import FIRST, LAST, Test

tests = pickle.load(open('tests.pkl', 'rb'))


def greedy_plan(t, mode):
    plan = {}
    for st in range(FIRST, LAST + 1):
        o = t.opts[st]
        if not o:
            plan[st] = -1; continue
        if mode == 'C':  # safest object
            plan[st] = int(np.argmax([x[3][80] for x in o]))  # p at T=20
        else:
            plan[st] = int(np.argmax([x[3][100] * x[2] for x in o]))  # EV at T=25
    return plan


def coord_ascent(t, plan, n=2000, seed=1, sweeps=6, verbose=True):
    plan = dict(plan)
    best = t.simulate(plan, n, seed)
    for sw in range(sweeps):
        improved = False
        for st in range(FIRST, LAST + 1):
            cur = plan[st]
            for a in [-1] + list(range(len(t.opts[st]))):
                if a == cur:
                    continue
                plan[st] = a
                v = t.simulate(plan, n, seed)
                if v > best + 1e-9:
                    best, cur, improved = v, a, True
            plan[st] = cur
        if verbose:
            print(f'  sweep {sw} obj {best:.2f}', flush=True)
        if not improved:
            break
    return plan, best


if __name__ == '__main__':
    out = {}
    for tid, t in tests.items():
        res = []
        for mode in ['C', 'EV']:
            p0 = greedy_plan(t, mode)
            print(tid, mode, 'start', round(t.simulate(p0, 4000, 99), 2))
            p, v = coord_ascent(t, p0)
            sc, rs = t.simulate(p, 4000, 99, return_all=True)
            print(tid, mode, 'final holdout', round(t.simulate(p, 4000, 99), 2), 'P(rest>45)', np.mean(rs > 45), 'mean rests', rs.mean())
            res.append((t.simulate(p, 4000, 99), p))
        out[tid] = max(res, key=lambda x: x[0])[1]
    pickle.dump(out, open('plans.pkl', 'wb'))
