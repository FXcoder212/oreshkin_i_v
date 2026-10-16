"""Precomputed per-step tables on a temperature grid + vectorised Monte Carlo of an open-loop plan."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from feat import add_step_feats, obj_feats, VAL
from models import load_train, fit_success, fit_temp, SF, TF

TG = np.arange(0.0, 70.01, 0.25)
FIRST, LAST, MAXREST = 11, 110, 45


def build_timeline(tid):
    ctx = pd.read_csv('data/context.csv')
    slots = pd.read_csv('data/slots.csv')
    c = ctx[ctx.test_id == tid].copy()
    s = slots[slots.test_id == tid].copy()
    s['action'] = 'REST'
    tl = pd.concat([c, s], ignore_index=True).sort_values('step_id')
    tl['series_id'] = tid
    return add_step_feats(tl).set_index('step_id')


class Test:
    def __init__(self, tid, sm, tm, resid):
        self.tid = tid
        tl = build_timeline(tid)
        objs = pd.read_csv('data/objects.csv')
        objs = objs[objs.test_id == tid]
        self.tl = tl
        ctx = tl.loc[1:10]
        # forced rests carried into step 11+
        forced = 0
        for st, r in ctx.iterrows():
            if r.action.startswith('REST'):
                forced = max(forced - 1, 0)
            elif r.result == 0:
                forced = 3
            else:
                forced = 0
        self.forced0 = forced
        self.T10 = ctx.loc[10, 'temperature']
        # step-10 transition (known action/result)
        row = obj_feats(ctx.loc[[10]].reset_index())
        self.mu10 = self._tgrid(tm, row)
        self.resid = resid
        # options per step: list of (slot_id, type, value, p(T) grid, mu_succ grid, mu_fail grid)
        self.opts = {}
        self.rest_mu = {}
        for st in range(FIRST, LAST + 1):
            g = tl.loc[st]
            base = tl.loc[[st]].reset_index()
            base['action'] = 'REST'
            self.rest_mu[st] = self._tgrid(tm, obj_feats(base.assign(result=1)))
            lst = []
            for _, ob in objs[objs.step_id == st].iterrows():
                r = base.copy()
                r['action'] = ob.slot_id.split('_')[0]
                for c in ['object_magnitude', 'object_color', 'object_altitude', 'moon_separation']:
                    r[c] = ob[c]
                r = obj_feats(r)
                p = self._pgrid(sm, r)
                mus = self._tgrid(tm, r.assign(result=1))
                muf = self._tgrid(tm, r.assign(result=0))
                lst.append((ob.slot_id, r.action.iloc[0], VAL[r.action.iloc[0][0]], p, mus, muf))
            self.opts[st] = lst

    @staticmethod
    def _grid_rows(r):
        R = pd.concat([r] * len(TG), ignore_index=True)
        R['temperature'] = TG
        R['dT_amb'] = TG - R.ambient_temperature
        return R

    def _pgrid(self, sm, r):
        R = self._grid_rows(r)
        if isinstance(sm, list):
            return np.mean([m.predict_proba(R[SF])[:, 1] for m in sm], axis=0)
        return sm.predict_proba(R[SF])[:, 1]

    def _tgrid(self, tm, r):
        R = self._grid_rows(r)
        if isinstance(tm, list):
            return np.mean([m.predict(R[TF]) for m in tm], axis=0)
        return tm.predict(R[TF])

    def simulate(self, plan, n=2000, seed=0, return_all=False):
        """plan: dict step -> option index (-1 = REST). Returns mean objective (score zeroed if rests>45)."""
        rng = np.random.default_rng(seed)
        res = self.resid
        T = self.T10 + np.interp(self.T10, TG, self.mu10) + rng.choice(res, n)
        forced = np.full(n, self.forced0)
        rests = np.zeros(n, int)
        score = np.zeros(n)
        U = rng.random((LAST + 1, n))
        E = rng.choice(res, (LAST + 1, n))
        for st in range(FIRST, LAST + 1):
            a = plan[st]
            isf = forced > 0
            dT = np.empty(n)
            if a < 0:
                rests += 1
                dT[:] = np.interp(T, TG, self.rest_mu[st])
                forced = np.maximum(forced - 1, 0)
            else:
                sid, typ, v, p, mus, muf = self.opts[st][a]
                pr = np.interp(T, TG, p)
                ok = (U[st] < pr) & ~isf
                fail = ~ok & ~isf
                score += ok * v
                rests += isf
                dT = np.where(isf, np.interp(T, TG, self.rest_mu[st]),
                              np.where(ok, np.interp(T, TG, mus), np.interp(T, TG, muf)))
                forced = np.where(isf, forced - 1, np.where(fail, 3, 0))
            T = T + dT + E[st]
        valid = rests <= MAXREST
        if return_all:
            return score, rests
        return float(np.mean(score * valid))


def oof_resid():
    d = load_train()
    t = d.dropna(subset=['next']).reset_index(drop=True)
    q = np.zeros(len(t))
    for tr, va in GroupKFold(5).split(t, groups=t.series_id):
        q[va] = fit_temp(t.iloc[tr]).predict(t.iloc[va][TF])
    return (t.next - t.temperature - q).values
