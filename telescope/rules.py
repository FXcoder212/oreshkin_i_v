"""Compare scoring-rule hypotheses on all submitted files."""
import pickle, numpy as np, pandas as pd, warnings
from sim import FIRST, LAST
warnings.filterwarnings('ignore')
tests = pickle.load(open('tests2.pkl', 'rb'))
real = {'submission_telescope.csv': 272, 'submission_telescope_robust.csv': 211,
        'submission_telescope_v3.csv': 264, 'submission_telescope_v4.csv': 117}


def plan_of(t, f):
    g = pd.read_csv(f)
    g = g[g.test_id == t.tid].set_index('step')
    return {st: -1 if g.loc[st, 'action'] == 'REST' else [o[0] for o in t.opts[st]].index(g.loc[st, 'slot_id'])
            for st in range(FIRST, LAST + 1)}


if __name__ == "__main__":
  for f, r in real.items():
      out = {'raw': [], 'zero': [], 'trunc': []}
      tot = {k: 0 for k in out}
      for tid, t in tests.items():
          p = plan_of(t, f)
          t.trunc = False
          sc, rs = t.simulate(p, 2000, 5, return_all=True)
          t.trunc = True
          sct, _ = t.simulate(p, 2000, 5, return_all=True)
          t.trunc = False
          for k, v in [('raw', sc), ('zero', sc * (rs <= 45)), ('trunc', sct)]:
              out[k].append(round(v.mean(), 1)); tot[k] += v.mean()
      print(f, 'real', r, {k: (round(tot[k]), out[k]) for k in out})
