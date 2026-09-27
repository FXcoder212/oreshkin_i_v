import numpy as np, pandas as pd

SENS = ['t_battery', 'v_battery', 'v_bus_aux', 'i_solar', 'flux_sensor', 't_engine', 't_radiator',
        'p_fuel', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z', 'signal_strength']
EDGES = np.array([0, .25, .5, .75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 11, 15, 25, np.inf])


def norm_diffs(df, per_session=True):
    """|первая разность| / робастный масштаб шума; NaN там, где разность не определена."""
    g = df.session_id.values
    X = df[SENS].values.astype(float)
    D = np.diff(X, axis=0, prepend=np.nan)
    D[np.r_[True, g[1:] != g[:-1]]] = np.nan
    Z = np.empty_like(D)
    for s in np.unique(g):
        m = g == s
        d = D[m]
        ns = 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d, 0)), 0) + 1e-12
        Z[m] = np.abs(d / ns)
    return Z


def binned(Z):
    B = np.digitize(np.nan_to_num(Z, nan=-1), EDGES) - 1        # -1 для NaN
    B[np.isnan(Z)] = -1
    return B


def fit_emissions(B, mask, alpha=1.0):
    """log-вероятности бинов для каждого датчика по строкам mask."""
    nb = len(EDGES) - 1
    P = np.zeros((B.shape[1], nb))
    for j in range(B.shape[1]):
        b = B[mask, j]; b = b[b >= 0]
        P[j] = np.log((np.bincount(b, minlength=nb) + alpha) / (len(b) + alpha * nb))
    return P


def loglik(B, P):
    """Сумма по датчикам log p(bin); NaN-датчики пропускаются."""
    L = np.zeros(len(B))
    for j in range(B.shape[1]):
        b = B[:, j]; ok = b >= 0
        L[ok] += P[j, b[ok]]
    return L


def hazards(lengths, K=60, smooth=2.0):
    """h_k = P(L=k | L>=k) для k=1..K по эмпирическим длинам (сглаженная гистограмма)."""
    cnt = np.bincount(np.clip(lengths, 1, K), minlength=K + 1)[1:].astype(float)
    ker = np.exp(-0.5 * (np.arange(-6, 7) / smooth) ** 2)
    pm = np.convolve(cnt, ker / ker.sum(), mode='same') + 1e-3
    pm /= pm.sum()
    surv = np.cumsum(pm[::-1])[::-1]
    h = pm / surv
    h[-1] = 0.2                                   # хвост: остаёмся в последнем состоянии
    return np.clip(h, 1e-6, 1 - 1e-6)


def posterior(E_N, E_S, E_A, E_E, h, q):
    """Прямой-обратный проход. Состояния: 0=N, 1..K = A_k (k-я строка эпизода), K+1 = E
    (первая штатная строка после эпизода). E_* — логарифмы эмиссий по строкам.
    Возвращает P(строка внутри эпизода)."""
    T = len(E_N); K = len(h); nS = K + 2
    le = np.empty((T, nS))
    le[:, 0] = E_N; le[:, 1] = E_S; le[:, 2:K + 1] = E_A[:, None]; le[:, K + 1] = E_E
    le -= le.max(1, keepdims=True)
    em = np.exp(le)
    stay = 1 - h

    def step(p):
        n = np.zeros(nS)
        n[0] = p[0] * (1 - q) + p[K + 1]
        n[1] = p[0] * q
        n[2:K + 1] = p[1:K] * stay[:K - 1]
        n[K] += p[K] * stay[K - 1]
        n[K + 1] = (p[1:K + 1] * h).sum()
        return n

    def bstep(b):                                  # b_t(i) = sum_j A_ij b_{t+1}(j) (эмиссии уже учтены)
        n = np.zeros(nS)
        n[0] = (1 - q) * b[0] + q * b[1]
        n[1:K] = stay[:K - 1] * b[2:K + 1] + h[:K - 1] * b[K + 1]
        n[K] = stay[K - 1] * b[K] + h[K - 1] * b[K + 1]
        n[K + 1] = b[0]
        return n

    F = np.empty((T, nS)); p = np.zeros(nS); p[0] = 1.0
    for t in range(T):
        p = (step(p) if t else p) * em[t]; p /= p.sum(); F[t] = p
    Bk = np.empty((T, nS)); b = np.ones(nS); Bk[-1] = b
    for t in range(T - 2, -1, -1):
        b = bstep(b * em[t + 1]); b /= b.sum(); Bk[t] = b
    post = F * Bk; post /= post.sum(1, keepdims=True)
    return post[:, 1:K + 1].sum(1)
