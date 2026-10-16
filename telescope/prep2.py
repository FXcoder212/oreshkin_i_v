import pickle, numpy as np
from models import load_train
from hist import add_hist
from sim2 import fit_models, oof_resid, Test2
d = add_hist(load_train())
sm, tm = fit_models(d)
resid = oof_resid(d)
tests = {t: Test2(t, sm, tm, resid) for t in (1, 2, 3)}
pickle.dump(tests, open('tests2.pkl', 'wb'))
