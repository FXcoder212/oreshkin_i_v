import numpy as np, pandas as pd
from feats2 import SENS, diff_residuals


def seq_inputs(df, res):
    """Словарь session_id -> матрица (каналы, время) для свёрточной сети."""
    out = {}
    for sid, d in df.groupby('session_id', sort=False):
        r = res.loc[d.index]
        ch = []
        for s in SENS:
            raw = d[s].values.astype(float)
            dx = np.diff(raw, prepend=np.nan)
            ns = 1.4826 * np.nanmedian(np.abs(dx - np.nanmedian(dx))) + 1e-12
            xi = pd.Series(raw).interpolate(limit_direction='both').values
            lvl = (xi - pd.Series(xi).rolling(101, center=True, min_periods=1).median().values) / ns
            e = r[s].values; es = 1.4826 * np.nanmedian(np.abs(e - np.nanmedian(e))) + 1e-12
            ch += [np.nan_to_num(dx / ns), np.nan_to_num(e / es), lvl / 5, np.isnan(raw).astype(float)]
        for m in ('cruise', 'maneuver', 'eclipse'):
            ch.append((d['mode'].values == m).astype(float))
        out[sid] = np.clip(np.vstack(ch), -10, 10).astype(np.float32)
    return out
