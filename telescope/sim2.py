"""History-aware simulator: models use lagged actions/temperatures, evaluated per trajectory."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from feat import obj_feats, VAL, TYPES
from models import load_train, SF, TF
from hist import add_hist, HF
from sim import build_timeline, FIRST, LAST, MAXREST

SFH, TFH = SF + HF, TF + HF


def fit_models(d, seed=0):
    o = d[d.cls != 'R']
    sm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.09, num_leaves=8, min_child_samples=30, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1).fit(o[SFH], o.result)
    t = d.dropna(subset=['next'])
    tm = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.1, num_leaves=15, min_child_samples=20, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1).fit(t[TFH], t.next - t.temperature)
    return sm, tm


def oof_resid(d):
    t = d.dropna(subset=['next']).reset_index(drop=True)
    q = np.zeros(len(t))
    for tr, va in GroupKFold(5).split(t, groups=t.series_id):
        q[va] = fit_models(t.iloc[tr])[1].predict(t.iloc[va][TFH])
    return (t.next - t.temperature - q).values


class Test2:
    def __init__(self, tid, sm, tm, resid):
        self.tid, self.resid = tid, resid
        self.sm, self.tm = sm.booster_, tm.booster_
        tl = build_timeline(tid)
        objs = pd.read_csv('data/objects.csv')
        objs = objs[objs.test_id == tid]
        ctx = obj_feats(tl.loc[1:10].reset_index())
        self.ctx = ctx
        forced = 0
        for r in ctx.itertuples():
            if r.action.startswith('REST'):
                forced = max(forced - 1, 0)
            elif r.result == 0:
                forced = 3
            else:
                forced = 0
        self.forced0 = forced
        self.static = {}   # (st, a) -> feature row dict (a=-1 rest)
        self.opts = {}
        for st in range(10, LAST + 1):
            base = tl.loc[[st]].reset_index()
            if st == 10:
                self.static[(10, 'ctx')] = obj_feats(base).iloc[0]
                continue
            base['action'] = 'REST'
            self.static[(st, -1)] = obj_feats(base).iloc[0]
            lst = []
            for i, ob in enumerate(objs[objs.step_id == st].itertuples()):
                r = base.copy()
                r['action'] = ob.slot_id.split('_')[0]
                for c in ['object_magnitude', 'object_color', 'object_altitude', 'moon_separation']:
                    r[c] = getattr(ob, c)
                r = obj_feats(r).iloc[0]
                self.static[(st, i)] = r
                lst.append((ob.slot_id, r.action, VAL[r.action[0]]))
            self.opts[st] = lst

    def _mat(self, row, cols, dyn, n):
        X = np.empty((n, len(cols)))
        for j, c in enumerate(cols):
            X[:, j] = dyn[c] if c in dyn else row[c]
        return X

    def simulate(self, plan, n=500, seed=0, return_all=False, last=LAST):
        rng = np.random.default_rng(seed)
        c = self.ctx
        # history state from context (steps 8,9,10)
        T = np.full(n, c.temperature.iloc[-1])
        obsh = list((c.cls != 'R').astype(float).values)
        clsh = list(c.cls_i.values.astype(float))
        magh = list(c.object_magnitude.values)
        Th = list(c.temperature.values)

        def hist_dyn(Tcur):
            st = np.zeros(n)
            # streak of previous obs (vectorised over histories stored as arrays)
            return st

        H = {  # arrays per sim, most recent last
            'obs': [np.full(n, v) for v in obsh[-5:]],
            'cls': [np.full(n, v) for v in clsh[-3:]],
            'mag': [np.full(n, v) for v in magh[-1:]],
            'T': [np.full(n, v) for v in Th[-3:]],
        }
        streak = np.zeros(n)
        for v in obsh[:-1]:
            pass

        def streak_from(obs_list):
            s = np.zeros(n); alive = np.ones(n, bool)
            for o in reversed(obs_list):
                alive &= o > 0
                s += alive
            return s

        def dyn_hist():
            # features describing the past relative to current step; H['T'][-1] is current T
            o, cl, Tt = H['obs'], H['cls'], H['T']
            return {'obs_l1': o[-1], 'obs_l2': o[-2], 'obs_l3': o[-3], 'cls_l1': cl[-1], 'cls_l2': cl[-2],
                    'cls_l3': cl[-3], 'dT_l1': Tt[-1] - Tt[-2], 'dT_l2': Tt[-2] - Tt[-3],
                    'obs_last5': sum(o[-5:]), 'streak_prev': streak_from(o[-5:]), 'mag_l1': H['mag'][-1]}

        # step 10 transition: history for step 10 is steps 7..9
        H10 = {'obs': [np.full(n, v) for v in obsh[-6:-1]], 'cls': [np.full(n, v) for v in clsh[-4:-1]],
               'mag': [np.full(n, magh[-2])], 'T': [np.full(n, v) for v in Th[-4:-1]] + []}
        Hsave = H
        H = {'obs': H10['obs'], 'cls': H10['cls'], 'mag': H10['mag'], 'T': H10['T'][1:] + [np.full(n, Th[-1])]}
        H['T'] = [np.full(n, v) for v in Th[-3:]]
        row10 = self.static[(10, 'ctx')]
        dyn = dyn_hist(); dyn.update(temperature=T, dT_amb=T - row10.ambient_temperature, result=np.full(n, row10.result))
        T = T + self.tm.predict(self._mat(row10, TFH, dyn, n), num_threads=1) + rng.choice(self.resid, n)
        H = Hsave
        H['T'] = H['T'][1:] + [T.copy()]

        forced = np.full(n, self.forced0)
        rests = np.zeros(n, int)
        score = np.zeros(n)
        for st in range(FIRST, last + 1):
            a = plan[st]
            isf = forced > 0
            dyn = dyn_hist()
            dyn.update(temperature=T, dT_amb=T - self.static[(st, -1)].ambient_temperature)
            obs_now = np.zeros(n); cls_now = np.full(n, 3.0); mag_now = np.full(n, np.nan)
            if a < 0:
                rests += 1
                res = np.ones(n)
                dT = self.tm.predict(self._mat(self.static[(st, -1)], TFH, {**dyn, 'result': res}, n), num_threads=1)
                forced = np.maximum(forced - 1, 0)
            else:
                row = self.static[(st, a)]
                act = ~isf
                p = self.sm.predict(self._mat(row, SFH, dyn, n), num_threads=1)
                ok = (rng.random(n) < p) & act
                fail = act & ~ok
                score += ok * VAL[row.action[0]]
                rests += isf
                res = np.where(fail, 0.0, 1.0)
                dTo = self.tm.predict(self._mat(row, TFH, {**dyn, 'result': res}, n), num_threads=1)
                dTr = self.tm.predict(self._mat(self.static[(st, -1)], TFH, {**dyn, 'result': np.ones(n)}, n), num_threads=1)
                dT = np.where(act, dTo, dTr)
                obs_now = act.astype(float); cls_now = np.where(act, row.cls_i, 3.0)
                mag_now = np.where(act, row.object_magnitude, np.nan)
                forced = np.where(isf, forced - 1, np.where(fail, 3, 0))
            T = T + dT + rng.choice(self.resid, n)
            H['obs'] = H['obs'][1:] + [obs_now]
            H['cls'] = H['cls'][1:] + [cls_now]
            H['mag'] = [mag_now]
            H['T'] = H['T'][1:] + [T.copy()]
        if return_all:
            return score, rests
        return float(np.mean(score * (rests <= MAXREST)))
