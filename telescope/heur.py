"""Evaluate a family of structured heuristic plans under the truncation rule (dA=0 and dA=-1.5)."""
import pickle, sys, itertools, numpy as np, warnings
from sim import FIRST, LAST
from sim2 import SFH
from shiftA import Shift
warnings.filterwarnings('ignore')
tests = pickle.load(open('tests2.pkl', 'rb'))
z = np.zeros(1)


def pst(t, st, i, T, rested):
    row = t.static[(st, i)]
    o = 0.0 if rested else 1.0
    dyn = {'temperature': z + T, 'dT_amb': z + T - row.ambient_temperature, 'obs_l1': z + o, 'obs_l2': z + o,
           'obs_l3': z + o, 'cls_l1': z + (3 if rested else 2), 'cls_l2': z + 2, 'cls_l3': z + 2, 'dT_l1': z,
           'dT_l2': z, 'obs_last5': z + 5 * o, 'streak_prev': z + 5 * o, 'mag_l1': z + (np.nan if rested else 3)}
    return t.sm.predict(t._mat(row, SFH, dyn, 1))[0]


def heuristic(t, thrA, thrB, rest_before_A, budget, T):
    plan, nrest = {}, 0
    for st in range(FIRST, LAST + 1):
        o = t.opts[st]
        pr = [pst(t, st, i, T, True) for i in range(len(o))]
        pw = [pst(t, st, i, T, False) for i in range(len(o))]
        A = [(pr[i], i) for i, x in enumerate(o) if x[1][0] == 'A']
        if A and max(A)[0] > thrA:
            if rest_before_A and nrest < budget and st > FIRST and plan[st - 1] >= 0:
                plan[st - 1] = -1
                nrest += 1
            plan[st] = max(A)[1]
            continue
        B = [(pw[i], i) for i, x in enumerate(o) if x[1][0] == 'B']
        if B and max(B)[0] > thrB:
            plan[st] = max(B)[1]
            continue
        C = [(pw[i], i) for i, x in enumerate(o) if x[1][0] == 'C']
        if C:
            plan[st] = max(C)[1]
        else:
            plan[st] = max(range(len(o)), key=lambda i: pw[i] * o[i][2]) if o else -1
    return plan


if __name__ == '__main__':
    tid = int(sys.argv[1])
    t = tests[tid]
    base = t.sm
    res = []
    for thrA, thrB, rb, bud, T in itertools.product([0.3, 0.45, 0.6, 0.75], [0.5, 0.7, 0.85, 1.1], [0, 1],
                                                    [10, 25], [24, 28]):
        if rb == 0 and bud == 25:
            continue
        t.sm = base
        p = heuristic(t, thrA, thrB, rb, bud, T)
        out = []
        for dA in (0.0, -1.5):
            t.sm = Shift(base, dA)
            t.trunc = True
            out.append(t.simulate(p, 400, 7, return_all=True)[0].mean())
        t.sm = base
        t.trunc = False
        raw = t.simulate(p, 400, 7, return_all=True)[0].mean()
        res.append((out[1], out[0], raw, thrA, thrB, rb, bud, T))
        pickle.dump(p, open(f'heur_{tid}_{thrA}_{thrB}_{rb}_{bud}_{T}.pkl', 'wb'))
    res.sort(reverse=True)
    for r in res[:8]:
        print(tid, 'trunc_dA-1.5 %.1f trunc_dA0 %.1f raw %.1f | thrA %s thrB %s restA %s bud %s T %s' % r, flush=True)
