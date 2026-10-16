import pickle, numpy as np, pandas as pd
from sim import FIRST, LAST
from opt import tests, coord_ascent

plans = pickle.load(open('plans.pkl', 'rb'))
rows = []
for tid, t in tests.items():
    p, _ = coord_ascent(t, plans[tid], n=4000, seed=5, sweeps=3, verbose=False)
    sc, rs = t.simulate(p, 8000, 123, return_all=True)
    print(tid, 'expected', round(np.mean(sc * (rs <= 45)), 2), 'P(rests>45)', round(np.mean(rs > 45), 3))
    plans[tid] = p
    for st in range(FIRST, LAST + 1):
        a = p[st]
        if a < 0:
            rows.append((tid, st, 'REST', ''))
        else:
            sid, typ = t.opts[st][a][:2]
            rows.append((tid, st, typ, sid))
pickle.dump(plans, open('plans.pkl', 'wb'))
sub = pd.DataFrame(rows, columns=['test_id', 'step', 'action', 'slot_id'])
# format checks
slots = pd.read_csv('data/slots.csv')
avail = {(r.test_id, r.step_id): set(r.available_object_ids.split('|')) for r in slots.itertuples()}
assert len(sub) == 300 and sub.groupby('test_id').size().eq(100).all()
for r in sub.itertuples():
    if r.action != 'REST':
        assert r.slot_id in avail[(r.test_id, r.step)] and r.slot_id.split('_')[0] == r.action
assert sub.groupby('test_id').action.apply(lambda s: (s == 'REST').sum()).le(45).all()
sub.to_csv('submission_telescope.csv', index=False)
print('saved', sub.action.str[0].value_counts().to_dict())
