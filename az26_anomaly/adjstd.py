import numpy as np, pandas as pd
from feats2 import _wmean

AS = ['t_battery', 'v_battery', 'v_bus_aux', 'i_solar', 'flux_sensor', 't_engine', 't_radiator',
      'p_fuel', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'signal_strength']


def adj_feats(df, hs=(5, 8, 12, 16, 22)):
    """Дисперсия уровня в центрированном окне против соседних окон той же длины слева и справа
    (лучший признак «дрейфового» типа эпизодов), плюс несимметричные варианты (окно, кончающееся/
    начинающееся в t). Агрегаты по датчикам."""
    g = df.session_id.values; parts = []
    for s in pd.unique(g):
        m = np.where(g == s)[0]; agg = {}
        for c in AS:
            x = df[c].iloc[m].values.astype(float)
            for h in hs:
                L = 2 * h + 1
                mi, vi, ci = _wmean(x, -h, h + 1)
                _, vl, cl = _wmean(x, -h - L, -h); _, vr, cr = _wmean(x, h + 1, h + 1 + L)
                vadj = np.where((cl > 3) & (cr > 3), 0.5 * (vl + vr), np.where(cl > 3, vl, vr))
                agg.setdefault(f'adj{h}', []).append(np.nan_to_num(np.log(vi + 1e-12) - np.log(vadj + 1e-12)))
                # окно, заканчивающееся в t (эпизод справа от t закончился?) и начинающееся в t
                _, vb, cb = _wmean(x, -L + 1, 1); _, vbb, _ = _wmean(x, -2 * L + 1, -L + 1)
                _, vf, cf = _wmean(x, 0, L); _, vff, _ = _wmean(x, L, 2 * L)
                agg.setdefault(f'adjb{h}', []).append(np.nan_to_num(np.log(vb + 1e-12) - np.log(vbb + 1e-12)))
                agg.setdefault(f'adjf{h}', []).append(np.nan_to_num(np.log(vf + 1e-12) - np.log(vff + 1e-12)))
        P = {}
        for k, v in agg.items():
            A = np.clip(np.vstack(v), -8, 8)
            P[f'{k}_max'] = A.max(0); P[f'{k}_top3'] = np.sort(A, 0)[-3:].mean(0); P[f'{k}_mean'] = A.mean(0)
            P[f'{k}_n1'] = (A > 1).sum(0)
        parts.append(pd.DataFrame(P, index=m))
    out = pd.concat(parts).sort_index()
    return pd.concat([out, out.groupby(g).rank(pct=True).add_suffix('_rk')], axis=1)
