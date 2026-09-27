import numpy as np, pandas as pd
from hmm import norm_diffs


def bracket_feats(df):
    """Признаки одновременных скачков нескольких датчиков (маркеры границ эпизода)."""
    Z = np.nan_to_num(norm_diffs(df))
    g = df.session_id.values
    F = {}
    for thr in (3, 4, 6):
        F[f'cnt{thr}'] = (Z > thr).sum(1).astype(float)
    F['sumlog'] = np.log1p(Z).sum(1)
    out = pd.DataFrame(F)
    res = {}
    for s in pd.unique(g):
        m = np.where(g == s)[0]; n = len(m); pos = np.arange(n)
        R = {}
        for thr, k in ((3, 2), (4, 2), (4, 1), (6, 1)):
            ev = out[f'cnt{thr}'].values[m] >= k
            idx = np.where(ev)[0]
            j = np.searchsorted(idx, pos, side='right')
            prev = np.where(j > 0, idx[np.maximum(j - 1, 0)], -10 ** 4)       # событие на строке <= t (начало)
            nxt = np.where(j < len(idx), idx[np.minimum(j, len(idx) - 1)], 10 ** 4 + n)  # событие > t (конец)
            dp = np.minimum(pos - prev, 200); dn = np.minimum(nxt - pos, 200); span = np.minimum(nxt - prev, 400)
            key = f'{thr}_{k}'
            R[f'dprev_{key}'] = dp; R[f'dnext_{key}'] = dn; R[f'span_{key}'] = span
            R[f'brk_{key}'] = ((span >= 6) & (span <= 60)).astype(float)
        for c in ('cnt3', 'cnt4', 'sumlog'):
            v = pd.Series(out[c].values[m])
            for w in (15, 31):
                R[f'{c}_m{w}'] = v.rolling(w, center=True, min_periods=1).mean().values
        res[s] = pd.DataFrame(R, index=m)
    return pd.concat([out, pd.concat(res.values()).sort_index()], axis=1)
