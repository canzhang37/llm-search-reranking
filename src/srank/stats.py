"""Paired tests, agreement, rank correlation."""
import numpy as np
from scipy import stats as st


def boot_ci(diff, n=10_000, seed=2027, level=0.95):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def signflip_p(diff, n=10_000, seed=2027):
    """Two-sided paired sign-flip test."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, d.size))
    null = np.abs((signs * d).mean(axis=1))
    # +1 both sides so p is never 0
    return float((np.sum(null >= obs - 1e-12) + 1) / (n + 1))


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (p.size - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def weighted_kappa(a, b, k=4, weights="linear", w=None):
    """Cohen's kappa for grades 0..k-1, optional weights."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    w = np.ones(a.size) if w is None else np.asarray(w, float)
    obs = np.zeros((k, k))
    np.add.at(obs, (a, b), w)
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    if weights == "linear":
        dis = np.abs(i - j) / (k - 1)
    elif weights == "quadratic":
        dis = ((i - j) / (k - 1)) ** 2
    else:
        dis = (i != j).astype(float)
    return float(1 - (dis * obs).sum() / (dis * exp).sum())


def cluster_boot(fn, groups, n=2_000, seed=2027, level=0.95):
    """Percentile interval, resampling whole queries."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        vals.append(fn(np.concatenate([members[g] for g in pick])))
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def kendall(x, y):
    return float(st.kendalltau(x, y).statistic)
