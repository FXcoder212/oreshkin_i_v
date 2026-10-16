"""Replay train series through the history-aware simulator (out-of-fold models) to check bias."""
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import GroupKFold
from feat import obj_feats, VAL
from models import load_train
from hist import add_hist
from sim2 import Test2, fit_models, oof_resid
warnings.filterwarnings('ignore')


class TrainSeries(Test2):
    def __init__(self, sid, raw, sm, tm, resid):
        self.tid, self.resid = sid, resid
        self.sm, self.tm = sm.booster_, tm.booster_
        from feat import add_step_feats
        tl = add_step_feats(raw.assign(series_id=sid)).set_index('step_id')
        ctx = obj_feats(tl.loc[1:10].reset_index())
        self.ctx = ctx
        forced = 0
        for r in ctx.itertuples():
            forced = max(forced - 1, 0) if r.action.startswith('REST') else (3 if r.result == 0 else 0)
        self.forced0 = forced
        self.static, self.opts, self.plan = {}, {}, {}
        self.static[(10, 'ctx')] = obj_feats(tl.loc[[10]].reset_index()).iloc[0]
        for st in range(11, 81):
            base = tl.loc[[st]].reset_index()
            act = base.action.iloc[0]
            r0 = base.copy(); r0['action'] = 'REST'
            self.static[(st, -1)] = obj_feats(r0).iloc[0]
            if act.startswith('REST'):
                self.opts[st] = []; self.plan[st] = -1
            else:
                r = obj_feats(base).iloc[0]
                self.static[(st, 0)] = r
                self.opts[st] = [('x', act, VAL[act[0]])]; self.plan[st] = 0


raw = pd.read_csv('data/train.csv')
d = add_hist(load_train())
resid = oof_resid(d)
rows = []
for tr, va in GroupKFold(5).split(d, groups=d.series_id):
    sm, tm = fit_models(d.iloc[tr])
    for sid in d.iloc[va].series_id.unique()[:10]:
        s = TrainSeries(sid, raw[raw.series_id == sid].drop(columns='series_id'), sm, tm, resid)
        sc, rs = s.simulate(s.plan, 400, 0, return_all=True, last=80)
        r = raw[(raw.series_id == sid) & (raw.step_id > 10)]
        real_sc = sum(VAL[a[0]] for a, res in zip(r.action, r.result) if not a.startswith('REST') and res == 1)
        real_rs = r.action.str.startswith('REST').sum()
        rows.append((sid, sc.mean(), real_sc, rs.mean(), real_rs))
df = pd.DataFrame(rows, columns=['sid', 'sim_score', 'real_score', 'sim_rests', 'real_rests'])
print(df.round(1).to_string())
print(df.mean().round(2))
