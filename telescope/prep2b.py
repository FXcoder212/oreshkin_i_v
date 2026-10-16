"""Independent validation simulator: models fit on a bootstrap of train series with another seed."""
import pickle, numpy as np, warnings
from models import load_train
from hist import add_hist
from sim2 import fit_models, oof_resid, Test2
import pandas as pd
warnings.filterwarnings('ignore')
d = add_hist(load_train())
rng = np.random.default_rng(7)
ids = rng.choice(d.series_id.unique(), d.series_id.nunique(), replace=True)
parts = []
for k, s in enumerate(ids):
    x = d[d.series_id == s].copy()
    x['series_id'] = 10000 + k
    parts.append(x)
b = pd.concat(parts, ignore_index=True)
sm, tm = fit_models(b, seed=7)
resid = oof_resid(d)
pickle.dump({t: Test2(t, sm, tm, resid) for t in (1, 2, 3)}, open('tests2b.pkl', 'wb'))
