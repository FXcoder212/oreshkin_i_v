"""Answers for the space_risk_model analytics task (questions 1-10).

Usage: python solve.py [path/to/space_risk_model.csv] [--q10-step 0.1]
Writes submission.csv next to this script and prints diagnostics.
"""
import argparse
import csv
import itertools
import os
import time

import numpy as np
import pandas as pd

MODELS = [f"model_{i}_pred" for i in range(1, 11)]


def fmt(x, nd):
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- Q1
def q1(df):
    m = df.groupby(["orbit_type", "satellite_id"])["solar_panel_temp_mean"].mean().reset_index()
    ids = []
    for ot in (0, 1, 2):
        g = m[m.orbit_type == ot]
        ids.append(g.loc[g.solar_panel_temp_mean.idxmax(), "satellite_id"])
    return [str(i) for i in ids]


# ---------------------------------------------------------------- Q2
def ranked(score):
    # Descending by score; ties keep table order (satellite_id, orbit_number).
    return np.argsort(-score, kind="stable")


def roc_auc_full(y, score):
    ys = y[ranked(score)]
    pos_before = np.cumsum(ys)
    npos, nneg = ys.sum(), len(ys) - ys.sum()
    return pos_before[ys == 0].sum() / (npos * nneg)


def avg_precision_full(y, score):
    ys = y[ranked(score)]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return prec[ys == 1].sum() / ys.sum()


def q2(df):
    y = df.failure_in_12h.to_numpy()
    auc = {i + 1: roc_auc_full(y, df[m].to_numpy()) for i, m in enumerate(MODELS)}
    ap = {i + 1: avg_precision_full(y, df[m].to_numpy()) for i, m in enumerate(MODELS)}
    print("Q2 ROC-AUC:", {k: round(v, 4) for k, v in auc.items()})
    print("Q2 PR-AUC (AP), alternative:", {k: round(v, 4) for k, v in ap.items()})
    top = sorted(auc, key=lambda k: (-auc[k], k))[:5]
    top_ap = sorted(ap, key=lambda k: (-ap[k], k))[:5]
    print("Q2 alt (AP):", top_ap, [fmt(ap[k], 2) for k in top_ap])
    return [",".join(map(str, top)), ",".join(fmt(auc[k], 2) for k in top)]


# ---------------------------------------------------------------- Q3
def q3(df):
    g = df.groupby("satellite_id", sort=False)
    hr = df.solar_panel_temp_mean - g.solar_panel_temp_mean.shift(3)
    hr_abs = hr.abs()
    before = hr_abs.groupby(df.satellite_id, sort=False).shift(3)  # |hr| at k-3
    y = df.failure_in_12h
    no_fail = hr_abs[(y == 0) & hr_abs.notna()].mean()
    pre_fail = before[(y == 1) & before.notna()].mean()
    ratio = pre_fail / no_fail
    print(f"Q3 pre-failure={pre_fail:.5f} no-failure={no_fail:.5f} ratio={ratio:.4f}")
    return [str(int(round(ratio)))]


# ---------------------------------------------------------------- Q4
def q4(df):
    y = df.failure_in_12h.to_numpy()
    best = None
    for i, m in enumerate(MODELS, 1):
        p = df[m].to_numpy()
        t = p[y == 1].min()
        pred = p >= t
        fp = (pred & (y == 0)).sum()
        tn = (~pred & (y == 0)).sum()
        fpr = fp / (fp + tn)
        print(f"Q4 model {i}: t={t:.6f} FPR={fpr:.4f}")
        if best is None or fpr < best[1]:
            best = (i, fpr)
    return [str(best[0]), fmt(best[1], 2)]


# ---------------------------------------------------------------- Q5
def q5(df):
    res = {}
    for lag in range(-5, 6):
        rs = []
        for _, g in df.groupby("satellite_id", sort=False):
            s = g.solar_panel_temp_mean.to_numpy()
            b = g.battery_temp_mean.to_numpy()
            if lag > 0:
                x, z = s[:-lag], b[lag:]
            elif lag < 0:
                x, z = s[-lag:], b[:lag]
            else:
                x, z = s, b
            if len(x) > 1:
                rs.append(np.corrcoef(x, z)[0, 1])
        res[lag] = np.nanmean(rs)
    print("Q5:", {k: round(v, 4) for k, v in res.items()})
    lag = max(res, key=res.get)
    return [str(lag), fmt(res[lag], 2)]


# ---------------------------------------------------------------- Q6
def partial_corr(x, y, z):
    rxy, rxz, ryz = np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def q6(df):
    s, c, v = df.solar_panel_temp_mean, df.battery_current_mean, df.battery_voltage_mean
    p1 = partial_corr(s, c, v)
    p2 = partial_corr(c, v, s)
    print(f"Q6 r(S,I|V)={p1:.4f} r(I,V|S)={p2:.4f}")
    return [fmt(max(abs(p1), abs(p2)), 2)]


# ---------------------------------------------------------------- Q7
def best_f1(y, score):
    order = np.argsort(-score, kind="stable")
    s, ys = score[order], y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    cut = np.r_[s[1:] != s[:-1], True]  # positives = top k where a distinct value ends
    f1 = 2 * tp / (k + ys.sum())
    return f1[cut].max()


def q7(df):
    y = df.failure_in_12h.to_numpy()
    a = df.battery_temp_mean.to_numpy() / 40
    b = df.reaction_wheel_current_mean.to_numpy() / 2.0
    c = df.attitude_error_mean.to_numpy() / 1.5
    best = (-1, None)
    for i in range(101):
        for j in range(101 - i):
            w1, w2 = i / 100, j / 100
            w3 = (100 - i - j) / 100
            f = best_f1(y, w1 * a + w2 * b + w3 * c)
            if f > best[0]:
                best = (f, (w1, w2, w3))
    print(f"Q7 best F1={best[0]:.5f} weights={best[1]}")
    return [fmt(best[0], 3)]


# ---------------------------------------------------------------- Q8
def q8(df):
    rows = []
    for sid, g in df.groupby("satellite_id", sort=False):
        x = g.reaction_wheel_current_std.to_numpy()
        thr = x.mean() + 3 * x.std(ddof=0)
        rows.append(((x > thr).mean(), g.failure_in_12h.sum()))
    arr = np.array(rows, dtype=float)
    r = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
    print(f"Q8 r={r:.5f}")
    return [fmt(r, 3)]


# ---------------------------------------------------------------- Q9
def q9(df):
    X = np.column_stack([np.ones(len(df)), df.attitude_error_mean, df.reaction_wheel_current_mean])
    yv = df.battery_temp_mean.to_numpy()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ beta
    f = df.failure_in_12h.to_numpy()
    r1 = res[f == 1].std(ddof=1) / res[f == 0].std(ddof=1)
    r0 = res[f == 1].std(ddof=0) / res[f == 0].std(ddof=0)
    print(f"Q9 ratio ddof1={r1:.5f} ddof0={r0:.5f}")
    return [fmt(r1, 2)]


# ---------------------------------------------------------------- Q10
C_FN, C_FP = 20_000_000, 100_000


def q10(df, step, triples=None):
    y = df.failure_in_12h.to_numpy().astype(np.int64)
    grp = df.orbit_type.to_numpy()
    preds = {i: df[m].to_numpy() for i, m in enumerate(MODELS, 1)}
    n, npos = len(y), y.sum()
    gmask = [grp == g for g in (0, 1, 2)]
    gn = [m.sum() for m in gmask]
    gpos = [(y[m]).sum() for m in gmask]

    units = int(round(1 / step))
    weights = [(i / units, j / units, (units - i - j) / units)
               for i in range(1, units) for j in range(1, units - i)]

    best = None  # (loss, t, triple, weights)
    t0 = time.time()
    for triple in triples or itertools.combinations(range(1, 11), 3):
        pa, pb, pc = (preds[k] for k in triple)
        for w in weights:
            p = w[0] * pa + w[1] * pb + w[2] * pc
            order = np.argsort(-p, kind="stable")
            ps, ys = p[order], y[order]
            # Top k predicted dangerous, t = ps[k] (smallest t giving that split).
            k = np.arange(1, n)
            valid = ps[1:] < ps[:-1]
            tp = np.cumsum(ys)[:-1]
            fp = k - tp
            fn = npos - tp
            ok = valid & (fn <= 0.10 * npos) & (k <= 0.25 * n)
            if not ok.any():
                continue
            for gi in range(3):
                gm = gmask[gi][order]
                gk = np.cumsum(gm)[:-1]
                gtp = np.cumsum(gm & (ys == 1))[:-1]
                ok &= ((gpos[gi] - gtp) <= 0.15 * gpos[gi]) & (gk <= 0.30 * gn[gi])
            if not ok.any():
                continue
            loss = C_FN * fn + C_FP * fp
            loss = np.where(ok, loss, np.iinfo(np.int64).max)
            lmin = loss.min()
            idx = np.nonzero(loss == lmin)[0]
            t = ps[idx + 1].min()
            cand = (int(lmin), t, triple, w)
            if best is None or (cand[0], cand[1], cand[2]) < (best[0], best[1], best[2]):
                best = cand
        print(f"Q10 {triple} done, best so far {best[:3] if best else None} ({time.time() - t0:.0f}s)")
    print(f"Q10 best: loss={best[0]} t={best[1]:.6f} models={best[2]} weights={best[3]}")
    return [",".join(map(str, best[2])), fmt(best[1], 2), str(best[0])]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.join(here, "space_risk_model.csv"))
    ap.add_argument("--q10-step", type=float, default=0.1)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--q10-triples", type=str, default="",
                    help="restrict Q10 search, e.g. 7-9-10;5-9-10")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values(["satellite_id", "orbit_number"], kind="stable").reset_index(drop=True)
    print("rows:", len(df), "failure rate:", df.failure_in_12h.mean())

    triples = [tuple(int(x) for x in t.split("-")) for t in args.q10_triples.split(";") if t]
    funcs = [q1, q2, q3, q4, q5, q6, q7, q8, q9, lambda d: q10(d, args.q10_step, triples)]
    only = {int(x) for x in args.only.split(",") if x}
    rows = []
    for qi, f in enumerate(funcs, 1):
        if only and qi not in only:
            rows.append([qi, "", "", ""])
            continue
        t = time.time()
        ans = f(df)
        print(f"Q{qi}: {ans}  ({time.time() - t:.1f}s)")
        rows.append([qi] + ans + [""] * (3 - len(ans)))

    out = os.path.join(here, "submission.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question_id", "answer_1", "answer_2", "answer_3"])
        w.writerows(rows)
    print("written", out)


if __name__ == "__main__":
    main()
