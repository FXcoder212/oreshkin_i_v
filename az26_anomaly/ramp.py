import numpy as np, pandas as pd

RS = ['t_battery', 'v_battery', 'v_bus_aux', 'i_solar', 'flux_sensor', 't_engine', 't_radiator',
      'p_fuel', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'signal_strength']
LENS = (8, 12, 16, 20, 26, 34, 44)


def _cs(v):
    return np.r_[0.0, np.cumsum(v)]


def ramp_scan(x, lens=LENS, m=10, shape='ramp'):
    """Для каждого конца e (первая строка после окна) и длины L: МНК-амплитуда добавки формы w
    в окне [e-L, e) относительно линии между средними до и после окна. Возвращает dict L -> z[e]
    (робастно нормированная по сеансу)."""
    n = len(x); c = _cs(x); i = np.arange(x.size); ci = _cs(x * i)
    out = {}
    for L in lens:
        e = np.arange(n + 1)
        s = e - L
        ok = (s - m >= 0) & (e + m <= n)
        z = np.full(n + 1, np.nan)
        ee, ss = e[ok], s[ok]
        pre = (c[ss] - c[ss - m]) / m; post = (c[ee + m] - c[ee]) / m
        # сумма x и сумма x*t по окну
        Sx = c[ee] - c[ss]; Sxt = ci[ee] - ci[ss]
        k = np.arange(1, L + 1, dtype=float)               # позиции 1..L внутри окна
        if shape == 'ramp':
            w = k / L
        else:
            w = np.ones(L)
        # база b_t = pre + (post-pre)*(k-0.5)/L ; d_t = x_t - b_t
        bw = (k - 0.5) / L
        Sw = w.sum(); Sww = (w * w).sum(); Swb = (w * bw).sum()
        # sum w*x: w линейна по k для ramp -> через Sx и Sxt (t = ss + k - 1)
        if shape == 'ramp':
            Swx = (Sxt - (ss - 1) * Sx) / L
        else:
            Swx = Sx
        Swd = Swx - pre * Sw - (post - pre) * Swb
        z[ok] = Swd / np.sqrt(Sww)
        zz = z[:n + 1]
        med = np.nanmedian(zz); sc = 1.4826 * np.nanmedian(np.abs(zz - med)) + 1e-12
        out[L] = (zz - med) / sc
    return out


def row_cover_max(zdict, n):
    """Строковый признак: максимум |z| по окнам [e-L, e), покрывающим строку t."""
    best = np.zeros(n); bestL = np.zeros(n)
    for L, z in zdict.items():
        a = np.nan_to_num(np.abs(z[1:]), nan=0.0)        # индекс e-1 -> e=1..n
        # строка t покрыта при e в (t, t+L]  ->  a[e-1] при e-1 в [t, t+L-1]
        r = pd.Series(a[::-1]).rolling(L, min_periods=1).max().values[::-1]
        upd = r > best; best = np.where(upd, r, best); bestL = np.where(upd, L, bestL)
    return best, bestL


def ramp_feats(df):
    g = df.session_id.values; F = {}
    parts = []
    for s in pd.unique(g):
        m = np.where(g == s)[0]; n = len(m); A = []; Bx = []
        for c in RS:
            x = df[c].iloc[m].interpolate(limit_direction='both').values.astype(float)
            dx = np.diff(x); x = x / (1.4826 * np.median(np.abs(dx - np.median(dx))) + 1e-12)
            zr, _ = row_cover_max(ramp_scan(x, shape='ramp'), n)
            zb, _ = row_cover_max(ramp_scan(x, shape='box'), n)
            A.append(zr); Bx.append(zb)
        A = np.vstack(A); Bx = np.vstack(Bx)
        P = {}
        for nm, M in (('ramp', A), ('box', Bx)):
            P[f'{nm}_max'] = M.max(0); P[f'{nm}_top3'] = np.sort(M, 0)[-3:].mean(0)
            P[f'{nm}_mean'] = M.mean(0); P[f'{nm}_n3'] = (M > 3).sum(0); P[f'{nm}_n4'] = (M > 4).sum(0)
            for j, c in enumerate(RS):
                P[f'{nm}_{c}'] = M[j]
        parts.append(pd.DataFrame(P, index=m))
    out = pd.concat(parts).sort_index()
    rk = out[[c for c in out.columns if c.endswith(('_max', '_top3', '_mean', '_n3', '_n4'))]].groupby(g).rank(pct=True).add_suffix('_rk')
    return pd.concat([out, rk], axis=1)
