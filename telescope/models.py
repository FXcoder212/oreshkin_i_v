import numpy as np, pandas as pd, lightgbm as lgb
from feat import add_step_feats, obj_feats

SF = ['temperature', 'ambient_temperature', 'dT_amb', 'type_i', 'cls_i', 'object_magnitude', 'object_color',
      'object_altitude', 'moon_separation', 'sun_altitude', 'sun_prev', 'sun_next', 'moon_altitude',
      'moon_illumination', 'moon_term', 'airmass']
TF = ['temperature', 'ambient_temperature', 'dT_amb', 'type_i', 'cls_i', 'result', 'object_magnitude',
      'sun_altitude', 'sun_next', 'dsun_next', 'sun_prev', 'moon_altitude', 'moon_next', 'moon_illumination']


def load_train():
    d = obj_feats(add_step_feats(pd.read_csv('data/train.csv')))
    g = d.groupby('series_id')
    d['next'] = g.temperature.shift(-1)
    return d


def fit_success(d, seed=0):
    o = d[d.cls != 'R']
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                              verbose=-1).fit(o[SF], o.result)


def fit_temp(d, seed=0):
    t = d.dropna(subset=['next'])
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                          verbose=-1).fit(t[TF], t.next - t.temperature)
    return m
