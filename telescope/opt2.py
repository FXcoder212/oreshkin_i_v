import pickle, sys, time, numpy as np, pandas as pd
from sim import FIRST, LAST

N = 400


def plan_from_csv(t, f):
    s = pd.read_csv(f)
    g = s[s.test_id == t.tid].set_index('step')
    return {st: -1 if g.loc[st, 'action'] == 'REST' else [o[0] for o in t.opts[st]].index(g.loc[st, 'slot_id'])
            for st in range(FIRST, LAST + 1)}


def safe_plan(t):
    # highest-p object at a cool reference state, crude: prefer C, then B
    plan = {}
    for st in range(FIRST, LAST + 1):
        o = t.opts[st]
        cs = [i for i, x in enumerate(o) if x[1][0] == 'C']
        plan[st] = cs[0] if cs else (-1 if st % 2 else ([i for i, x in enumerate(o) if x[1][0] == 'B'] + [-1])[0])
    return plan


def ascent(t, plan, seed, sweeps, log):
    plan = dict(plan)
    best = t.simulate(plan, N, seed)
    for sw in range(sweeps):
        imp = False
        for st in range(FIRST, LAST + 1):
            cur = plan[st]
            for a in [-1] + list(range(len(t.opts[st]))):
                if a == cur:
                    continue
                plan[st] = a
                v = t.simulate(plan, N, seed)
                if v > best + 1e-9:
                    best, cur, imp = v, a, True
            plan[st] = cur
        print(f'  sweep {sw} obj {best:.2f} {time.strftime("%X")}', file=log, flush=True)
        pickle.dump(plan, open(f'plan2_{t.tid}.pkl', 'wb'))
        if not imp:
            break
    return plan, best


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    tid = int(sys.argv[1])
    t = pickle.load(open('tests2.pkl', 'rb'))[tid]
    log = sys.stdout
    t0 = time.time(); t.simulate(safe_plan(t), N, 0); print('eval time', time.time() - t0, flush=True)
    starts = {'robust': plan_from_csv(t, 'submission_telescope_robust.csv'), 'safe': safe_plan(t)}
    res = {}
    for k, p0 in starts.items():
        print(k, 'start', t.simulate(p0, 2000, 99), flush=True)
    # start from the better one
    k = max(starts, key=lambda k: t.simulate(starts[k], 1000, 98))
    p, _ = ascent(t, starts[k], 1, 6, log)
    sc, rs = t.simulate(p, 3000, 777, return_all=True)
    print(tid, 'from', k, 'holdout', np.mean(sc * (rs <= 45)), 'P>45', np.mean(rs > 45), flush=True)
