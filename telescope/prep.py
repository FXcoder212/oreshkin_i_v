import pickle
from models import load_train, fit_success, fit_temp
from sim import Test, oof_resid
d = load_train()
sm = [fit_success(d, s) for s in range(5)]
tm = [fit_temp(d, s) for s in range(5)]
resid = oof_resid()
tests = {t: Test(t, sm, tm, resid) for t in (1, 2, 3)}
pickle.dump(tests, open('tests.pkl', 'wb'))
for t, o in tests.items():
    print(t, 'forced0', o.forced0, 'T10', round(o.T10, 2))
