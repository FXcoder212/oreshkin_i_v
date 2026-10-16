import pickle, warnings, numpy as np, pandas as pd
from sim import FIRST, LAST
warnings.filterwarnings('ignore')
tests = pickle.load(open('tests2.pkl', 'rb'))
import os
files = {i: f'plan5_{i}.pkl' if os.path.exists(f'plan5_{i}.pkl') else f'plan4_{i}.pkl' for i in (1, 2, 3)}
rows = []
for tid, t in tests.items():
    p = pickle.load(open(files[tid], 'rb'))
    sc, rs = t.simulate(p, 2000, 4242, return_all=True)
    print(tid, 'raw', round(sc.mean(), 1), 'capped', round(np.mean(sc * (rs <= 45)), 1), 'rests med', np.median(rs))
    for st in range(FIRST, LAST + 1):
        a = p[st]
        rows.append((tid, st, 'REST', '') if a < 0 else (tid, st, t.opts[st][a][1], t.opts[st][a][0]))
sub = pd.DataFrame(rows, columns=['test_id', 'step', 'action', 'slot_id'])
slots = pd.read_csv('data/slots.csv')
avail = {(r.test_id, r.step_id): set(r.available_object_ids.split('|')) for r in slots.itertuples()}
assert len(sub) == 300 and sub.groupby('test_id').size().eq(100).all()
for r in sub.itertuples():
    if r.action != 'REST':
        assert r.slot_id in avail[(r.test_id, r.step)] and r.slot_id.split('_')[0] == r.action
assert sub.groupby('test_id').action.apply(lambda s: (s == 'REST').sum()).le(45).all()
sub.to_csv('submission_telescope_v4.csv', index=False)
print(sub.groupby('test_id').action.apply(lambda s: s.str[0].value_counts().to_dict()))
