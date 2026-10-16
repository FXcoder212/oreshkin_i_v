import pickle, numpy as np, pandas as pd
from sim import FIRST, LAST

tests = pickle.load(open('tests.pkl', 'rb'))
rows = []
for tid, t in tests.items():
    p = pickle.load(open(f'robust_{tid}.pkl', 'rb'))
    sc, rs = t.simulate(p, 8000, 321, return_all=True)
    print(tid, 'nominal expected', round(np.mean(sc * (rs <= 45)), 1), 'P(rests>45)', round(np.mean(rs > 45), 3))
    for st in range(FIRST, LAST + 1):
        a = p[st]
        rows.append((tid, st, 'REST', '') if a < 0 else (tid, st, *t.opts[st][a][:2][::-1]))
sub = pd.DataFrame(rows, columns=['test_id', 'step', 'action', 'slot_id'])
slots = pd.read_csv('data/slots.csv')
avail = {(r.test_id, r.step_id): set(r.available_object_ids.split('|')) for r in slots.itertuples()}
assert len(sub) == 300 and sub.groupby('test_id').size().eq(100).all()
for r in sub.itertuples():
    if r.action != 'REST':
        assert r.slot_id in avail[(r.test_id, r.step)] and r.slot_id.split('_')[0] == r.action
assert sub.groupby('test_id').action.apply(lambda s: (s == 'REST').sum()).le(45).all()
sub.to_csv('submission_telescope_robust.csv', index=False)
print(sub.groupby('test_id').action.apply(lambda s: s.str[0].value_counts().to_dict()))
