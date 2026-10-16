import numpy as np, pandas as pd
from models import load_train, TF, SF


def add_hist(d):
    d = d.sort_values(['series_id', 'step_id']).copy()
    g = d.groupby('series_id')
    d['obs'] = (d.cls != 'R').astype(int)
    d['fail'] = ((d.cls != 'R') & (d.result == 0)).astype(int)
    for k in (1, 2, 3):
        d[f'obs_l{k}'] = g.obs.shift(k).fillna(0)
        d[f'cls_l{k}'] = g.cls_i.shift(k).fillna(3)
        d[f'T_l{k}'] = g.temperature.shift(k)
        d[f'mag_l{k}'] = g.object_magnitude.shift(k)
    d['dT_l1'] = d.temperature - d.T_l1
    d['dT_l2'] = d.T_l1 - d.T_l2
    d['obs_last5'] = g.obs.transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
    st = []
    for _, s in g.obs:
        c = 0
        for v in s.shift(1).fillna(0).values:
            c = c + 1 if v else 0
            st.append(c)
    d['streak_prev'] = st
    return d

HF = ['obs_l1', 'obs_l2', 'obs_l3', 'cls_l1', 'cls_l2', 'cls_l3', 'dT_l1', 'dT_l2', 'obs_last5', 'streak_prev', 'mag_l1']
