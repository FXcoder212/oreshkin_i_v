import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

PS = ['t_battery', 'v_battery', 'v_bus_aux', 'i_solar', 'flux_sensor', 't_engine', 't_radiator', 'p_fuel', 'signal_strength']


def phys_residuals(df):
    """Остаток уровня датчика относительно линейной модели от экспоненциально сглаженных (причинно)
    значений остальных датчиков и режима. Средние по сеансу вычитаются. Без меток."""
    g = df.session_id.values
    X = df.groupby('session_id')[PS].transform(lambda v: v.interpolate(limit_direction='both'))
    M = pd.get_dummies(df['mode']).astype(float)
    drv = {}
    for c in PS + list(M.columns):
        base = pd.Series(X[c].values if c in PS else M[c].values)
        for tau in (3, 10, 30, 100):
            drv[f'{c}_e{tau}'] = base.groupby(g).transform(lambda s: s.ewm(span=tau, adjust=False).mean()).values
    D = pd.DataFrame(drv)
    D = D - D.groupby(g).transform('mean')
    R = {}
    for t in PS:
        cols = [c for c in D.columns if not c.startswith(t + '_')]
        tt = X[t] - X[t].groupby(g).transform('mean')
        e = tt.values - Ridge(1.0).fit(D[cols], tt).predict(D[cols])
        R[t] = np.where(df[t].isna(), np.nan, e / np.std(e))
    return pd.DataFrame(R, index=df.index)
