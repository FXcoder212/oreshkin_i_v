import numpy as np, pandas as pd

VS = ['t_battery', 'v_battery', 'v_bus_aux', 'i_solar', 'flux_sensor', 't_engine', 't_radiator',
      'p_fuel', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'signal_strength']


def rev_feats(df, lens=(6, 10, 14, 20, 28, 40)):
    """«Скачок в конце отменяет накопленный дрейф»: для конца окна e и длины L
    drift = mean(x[e-2:e]) - mean(x[e-L-3:e-L]), jump = mean(x[e:e+2]) - mean(x[e-2:e]);
    rev = -sign(drift)*jump / локальный шум; revmin = min(|drift|/(σ√L), rev).
    Признак строки — максимум по окнам, покрывающим строку. Агрегаты по датчикам."""
    g = df.session_id.values; parts = []
    for s in pd.unique(g):
        m = np.where(g == s)[0]; n = len(m); R = []; RM = []
        for c in VS:
            x = df[c].iloc[m].interpolate(limit_direction='both').values.astype(float)
            d = np.abs(np.diff(x, prepend=x[0]))
            sd = 1.4826 * pd.Series(d).rolling(121, center=True, min_periods=20).median().values + 1e-12
            cs = np.r_[0, np.cumsum(x)]
            mean = lambda a, b: (cs[np.clip(b, 0, n)] - cs[np.clip(a, 0, n)]) / np.maximum(np.clip(b, 0, n) - np.clip(a, 0, n), 1)
            e = np.arange(n + 1)
            last = mean(e - 2, e); post = mean(e, e + 2)
            jump = post - last
            best_r = np.zeros(n); best_m = np.zeros(n)
            for L in lens:
                pre = mean(e - L - 3, e - L)
                drift = last - pre
                sde = sd[np.clip(e, 0, n - 1)]
                rv = -np.sign(drift) * jump / sde
                rm = np.where(rv > 0, np.minimum(np.abs(drift) / (sde * np.sqrt(L)), rv), 0.0)
                bad = (e - L - 3 < 0) | (e + 2 > n)
                rv[bad] = 0; rm[bad] = 0
                # строка t покрыта окнами с e в (t, t+L]
                for arr, best in ((rv, best_r), (rm, best_m)):
                    a = np.clip(arr[1:], 0, None)
                    cov = pd.Series(a[::-1]).rolling(L, min_periods=1).max().values[::-1]
                    np.maximum(best, cov, out=best)
            R.append(best_r); RM.append(best_m)
        P = {}
        for nm, M in (('rev', np.vstack(R)), ('revm', np.vstack(RM))):
            M = np.clip(M, 0, 50)
            P[f'{nm}_max'] = M.max(0); P[f'{nm}_top2'] = np.sort(M, 0)[-2:].mean(0)
            P[f'{nm}_top3'] = np.sort(M, 0)[-3:].mean(0); P[f'{nm}_mean'] = M.mean(0)
            P[f'{nm}_n3'] = (M > 3).sum(0)
        parts.append(pd.DataFrame(P, index=m))
    out = pd.concat(parts).sort_index()
    return pd.concat([out, out.groupby(g).rank(pct=True).add_suffix('_rk')], axis=1)
