import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

SENS = ['t_battery', 'v_battery', 'v_bus_aux', 'i_solar', 'flux_sensor', 't_engine', 't_radiator',
        'p_fuel', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'signal_strength']


def _wsum(v, lo, hi):
    n = len(v); cs = np.r_[0, np.cumsum(v)]; i = np.arange(n)
    return cs[np.clip(i + hi, 0, n)] - cs[np.clip(i + lo, 0, n)]


def _wmean(x, lo, hi):
    m = ~np.isnan(x)
    s = _wsum(np.where(m, x, 0), lo, hi); c = _wsum(m.astype(float), lo, hi)
    s2 = _wsum(np.where(m, x * x, 0), lo, hi)
    mu = s / np.maximum(c, 1); var = np.maximum(s2 / np.maximum(c, 1) - mu ** 2, 0)
    mu[c == 0] = np.nan
    return mu, var, c


def diff_residuals(df):
    """Первые разности каждого датчика минус их линейный прогноз по разностям остальных
    датчиков (лаги -1, 0, +1). Регрессия обучается без меток — на всех строках df."""
    X = df.groupby('session_id')[SENS].transform(lambda v: v.interpolate(limit_direction='both'))
    g = df.session_id.values
    D = X.groupby(g).diff().fillna(0); D = D / D.std()
    lags = [D.groupby(g).shift(k).fillna(0).add_suffix(f'_{k}') for k in (-1, 1)]
    R = {}
    for s in SENS:
        oth = [o for o in SENS if o != s]
        Z = pd.concat([D[oth]] + [l[[f'{o}_{k}' for o in oth]] for l, k in zip(lags, (-1, 1))], axis=1)
        e = D[s] - Ridge(1.0).fit(Z, D[s]).predict(Z)
        R[s] = np.where(df[s].isna() | df[s].groupby(g).shift(1).isna(), np.nan, e)
    return pd.DataFrame(R, index=df.index)


def _channel(F, agg, name, dz, lvl):
    """dz — нормированные первые разности, lvl — нормированный уровень."""
    adz = np.abs(dz); sa = pd.Series(np.nan_to_num(adz))
    for w in (10, 25):
        agg.setdefault(f'{name}jmpf{w}', []).append(sa[::-1].rolling(w, min_periods=1).max()[::-1].shift(-1).fillna(0).values)
        agg.setdefault(f'{name}jmpb{w}', []).append(sa.rolling(w, min_periods=1).max().values)
    agg.setdefault(f'{name}vol11', []).append(np.nan_to_num(pd.Series(adz).rolling(11, center=True, min_periods=1).mean().values))
    e2 = dz ** 2
    for h in (6, 10, 16):
        m, _, _ = _wmean(e2, -h, h + 1)
        for C in (60, 150):
            mc, _, _ = _wmean(e2, -h - C, h + C + 1)
            agg.setdefault(f'{name}dvr{h}_{C}', []).append(np.nan_to_num(np.log(m + 1e-3) - np.log(mc + 1e-3)))
    for h in (4, 8, 12, 20):
        mi, vi, ci = _wmean(lvl, -h, h + 1)
        ml, vl, cl = _wmean(lvl, -h - 30, -h); mr, vr, cr = _wmean(lvl, h + 1, h + 31)
        mc = np.where(np.isnan(ml), mr, np.where(np.isnan(mr), ml, (ml * cl + mr * cr) / np.maximum(cl + cr, 1)))
        vc = (vl * cl + vr * cr) / np.maximum(cl + cr, 1) + 1e-9
        agg.setdefault(f'{name}bump{h}', []).append(np.abs(np.nan_to_num((mi - mc) / np.sqrt(vc) * np.sqrt(np.maximum(ci, 1)))))
        agg.setdefault(f'{name}vr{h}', []).append(np.nan_to_num(np.sqrt(vi / vc), nan=1))
        if h == 8:
            agg.setdefault(f'{name}lr8', []).append(np.abs(np.nan_to_num((mr - ml) / np.sqrt(vc))))
    for h in (6, 10, 14):
        mi, vi, ci = _wmean(lvl, -h, h + 1)
        ml, vl, cl = _wmean(lvl, -h - 80, -h); mr, vr, cr = _wmean(lvl, h + 1, h + 81)
        vc = (vl * cl + vr * cr) / np.maximum(cl + cr, 1) + 1e-9
        agg.setdefault(f'{name}lvr{h}', []).append(np.nan_to_num(np.log(vi + 1e-9) - np.log(vc)))
    med = pd.Series(lvl).rolling(151, center=True, min_periods=1).median().values
    sm = pd.Series(lvl).rolling(9, center=True, min_periods=1).mean().values
    agg.setdefault(f'{name}dev151', []).append(np.abs(np.nan_to_num(sm - med)))


def session_feats(d, res):
    n = len(d); pos = np.arange(n); F = {}
    F['mode'] = d['mode'].map({'cruise': 0, 'maneuver': 1, 'eclipse': 2}).values
    ch = np.r_[0, np.where(d['mode'].values[1:] != d['mode'].values[:-1])[0] + 1, n]
    k = np.searchsorted(ch, pos, side='right')
    F['dist_prev_mode'] = pos - ch[k - 1]
    F['dist_next_mode'] = ch[np.minimum(k, len(ch) - 1)] - pos
    F['rel_pos'] = pos / n
    agg = {}
    for s in SENS:
        raw = d[s].values.astype(float)
        dx = np.diff(raw, prepend=np.nan)
        ns = 1.4826 * np.nanmedian(np.abs(dx - np.nanmedian(dx))) + 1e-12
        _channel(F, agg, '', dx / ns, raw / ns)
        e = res[s].values
        es = 1.4826 * np.nanmedian(np.abs(e - np.nanmedian(e))) + 1e-12
        e = e / es
        _channel(F, agg, 'r_', e, np.nancumsum(np.nan_to_num(e)))   # уровень = накопленный остаток
    for k_, v in agg.items():
        A = np.vstack(v)
        F[f'agg_{k_}_mean'] = A.mean(0); F[f'agg_{k_}_max'] = A.max(0)
        F[f'agg_{k_}_q80'] = np.quantile(A, 0.8, axis=0)
        F[f'agg_{k_}_top3'] = np.sort(A, 0)[-3:].mean(0)
        if 'dvr' in k_ or 'lvr' in k_:
            F[f'agg_{k_}_n05'] = (A > 0.5).sum(0); F[f'agg_{k_}_n10'] = (A > 1.0).sum(0)
    out = pd.DataFrame(F)
    out['timestamp_id'] = d['timestamp_id'].values; out['session_id'] = d['session_id'].values
    return out


def build(df, res):
    return pd.concat([session_feats(g.reset_index(drop=True), res.loc[g.index].reset_index(drop=True))
                      for _, g in df.groupby('session_id', sort=False)], ignore_index=True)
