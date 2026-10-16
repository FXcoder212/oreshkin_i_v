"""P(total >= 450) for candidate plans under trunc rule, dA in {0, -1.5}; per-test sims convolved by independent sampling."""
import pickle, glob, numpy as np, warnings
from rules import tests, plan_of
from shiftA import Shift
warnings.filterwarnings('ignore')
N = 1500
base = {tid: t.sm for tid, t in tests.items()}


def dist(tid, plan, dA, rule):
    t = tests[tid]
    t.sm = Shift(base[tid], dA)
    t.trunc = rule == 'trunc'
    sc, rs = t.simulate(plan, N, 77, return_all=True)
    t.sm, t.trunc = base[tid], False
    return sc


cands = {tid: {} for tid in (1, 2, 3)}
for tid in (1, 2, 3):
    for f in ['submission_telescope.csv', 'submission_telescope_v3.csv', 'submission_telescope_v4.csv',
              'submission_telescope_final.csv']:
        cands[tid][f] = plan_of(tests[tid], f)
    for f in glob.glob(f'plan[56]_{tid}.pkl') + glob.glob(f'heur_{tid}_*.pkl'):
        cands[tid][f] = pickle.load(open(f, 'rb'))
rng = np.random.default_rng(0)
for rule in ['trunc', 'raw']:
    for dA in (0.0, -1.5):
        best = {}
        for tid in (1, 2, 3):
            res = []
            for k, p in cands[tid].items():
                sc = dist(tid, p, dA, rule)
                res.append((np.mean(sc >= 150), sc.mean(), np.percentile(sc, 90), k, sc))
            res.sort(key=lambda r: -r[1])
            best[tid] = res
            print(rule, dA, tid, 'top by mean:', [(round(r[1]), round(r[2]), r[3][:28]) for r in res[:3]], flush=True)
        # best combination for P(sum>=450): try top-4 by mean and top-4 by p90 per test
        pools = {tid: sorted(best[tid], key=lambda r: -r[1])[:4] + sorted(best[tid], key=lambda r: -r[2])[:4] for tid in best}
        bestc = None
        for a in pools[1]:
            for b in pools[2]:
                for c in pools[3]:
                    s = rng.permutation(a[4]) + rng.permutation(b[4]) + rng.permutation(c[4])
                    pr = np.mean(s >= 450)
                    if bestc is None or pr > bestc[0]:
                        bestc = (pr, s.mean(), a[3], b[3], c[3])
        print(rule, dA, 'BEST P(>=450) %.3f mean %.0f' % bestc[:2], bestc[2:], flush=True)
