"""Robust re-optimisation: objective = mean expected score over a set of pessimistic model perturbations."""
import pickle, sys, copy, numpy as np
from sim import FIRST, LAST
from opt import greedy_plan

SCEN = [(0, 0), (-0.5, 0), (0, 0.5), (-0.5, 0.5), (-1, 0), (-0.3, 1.0)]


def perturb(t, dl=0.0, dT=0.0):
    t2 = copy.copy(t)
    t2.opts = {}
    for st, lst in t.opts.items():
        new = []
        for sid, typ, v, p, mus, muf in lst:
            pc = np.clip(p, 1e-4, 1 - 1e-4)
            lg = np.log(pc / (1 - pc)) + dl
            new.append((sid, typ, v, 1 / (1 + np.exp(-lg)), mus + dT, muf + dT))
        t2.opts[st] = new
    t2.rest_mu = {k: v + dT for k, v in t.rest_mu.items()}
    return t2


class Robust:
    def __init__(self, t):
        self.t = t
        self.ts = [perturb(t, *s) for s in SCEN]

    def simulate(self, plan, n, seed):
        return float(np.mean([x.simulate(plan, n, seed + i) for i, x in enumerate(self.ts)]))

    def report(self, plan):
        out = []
        for x in self.ts:
            sc, rs = x.simulate(plan, 4000, 777, return_all=True)
            out.append((round(np.mean(sc * (rs <= 45))), round(np.mean(rs > 45), 2)))
        return out


def ascent(R, plan, n, seed, sweeps):
    plan = dict(plan)
    best = R.simulate(plan, n, seed)
    for sw in range(sweeps):
        imp = False
        for st in range(FIRST, LAST + 1):
            cur = plan[st]
            for a in [-1] + list(range(len(R.t.opts[st]))):
                if a == cur:
                    continue
                plan[st] = a
                v = R.simulate(plan, n, seed)
                if v > best + 1e-9:
                    best, cur, imp = v, a, True
            plan[st] = cur
        print(f'  sweep {sw} obj {best:.2f}', flush=True)
        if not imp:
            break
    return plan, best


if __name__ == '__main__':
    tid = int(sys.argv[1])
    tests = pickle.load(open('tests.pkl', 'rb'))
    old = pickle.load(open('plans.pkl', 'rb'))[tid]
    R = Robust(tests[tid])
    print('old plan', R.report(old), R.simulate(old, 3000, 999))
    best = None
    for name, p0 in [('C', greedy_plan(tests[tid], 'C')), ('old', old)]:
        p, _ = ascent(R, p0, 800, 1, 8)
        v = R.simulate(p, 3000, 999)
        print(tid, name, 'holdout', round(v, 2), R.report(p), flush=True)
        if best is None or v > best[0]:
            best = (v, p)
    pickle.dump(best[1], open(f'robust_{tid}.pkl', 'wb'))
