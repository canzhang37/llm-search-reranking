"""Cascade clicks, team-draft interleaving, sizes."""
import itertools

import numpy as np
from scipy.stats import norm

USERS = {
    "perfect": ((0.00, 0.20, 0.60, 1.00), (0.0, 0.0, 0.0, 0.0)),
    "navigational": ((0.05, 0.30, 0.70, 0.95), (0.20, 0.30, 0.70, 0.90)),
    "informational": ((0.40, 0.60, 0.80, 0.90), (0.10, 0.20, 0.30, 0.50)),
}
K = 10


def grades(ranking, qrel, k=K):
    return np.array([qrel.get(p, 0) for p in ranking[:k]], dtype=int)


def success_prob(g, user):
    """P(click then stop), and P(any click)."""
    c, s = (np.asarray(x)[g] for x in USERS[user])
    reach = np.concatenate([[1.0], np.cumprod(1 - c * s)[:-1]])
    return float(np.sum(reach * c * s)), float(1 - np.prod(1 - c))


def team_draft(a, b, flips, k=K):
    """Merged top k and team per slot (0 = a, 1 = b)."""
    merged, team, seen = [], [], set()
    na = nb = 0
    ia = ib = 0
    flips = iter(flips)
    while len(merged) < k:
        while ia < len(a) and a[ia] in seen:
            ia += 1
        while ib < len(b) and b[ib] in seen:
            ib += 1
        if ia >= len(a) and ib >= len(b):
            break
        a_turn = na < nb or (na == nb and next(flips) == 0)
        if (a_turn and ia < len(a)) or ib >= len(b):
            merged.append(a[ia]); team.append(0); seen.add(a[ia]); na += 1
        else:
            merged.append(b[ib]); team.append(1); seen.add(b[ib]); nb += 1
    return merged, np.array(team, dtype=int)


# 5 flips fill 10 slots, all equally likely
FLIPS = list(itertools.product((0, 1), repeat=K // 2))


def interleave_outcome(g, team, user):
    """Exact P(a wins), P(tie), P(b wins)."""
    c, s = (np.asarray(x)[g] for x in USERS[user])
    off = K
    alive = np.zeros(2 * K + 1)
    alive[off] = 1.0
    done = np.zeros(2 * K + 1)
    for i in range(g.size):
        step = 1 if team[i] == 0 else -1
        clicked = np.roll(alive, step) * c[i]
        alive = alive * (1 - c[i]) + clicked * (1 - s[i])
        done += clicked * s[i]
    final = alive + done
    return float(final[off + 1:].sum()), float(final[off]), float(final[:off].sum())


def query_tables(run_a, run_b, qrels, qids, user):
    """Per-query A/B rates and interleaving outcomes."""
    ab = np.array([[*success_prob(grades(run_a[q], qrels[q]), user),
                    *success_prob(grades(run_b[q], qrels[q]), user)] for q in qids])
    il = np.zeros((len(qids), len(FLIPS), 3))
    for qi, q in enumerate(qids):
        for fi, f in enumerate(FLIPS):
            merged, team = team_draft(run_a[q], run_b[q], f)
            il[qi, fi] = interleave_outcome(grades(merged, qrels[q]), team, user)
    return ab, il


def needed_ab(p1, p2, alpha=0.05, power=0.8):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    d = p1 - p2
    return np.inf if d == 0 else 2 * z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / d ** 2


def needed_il(mean, var, alpha=0.05, power=0.8):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return np.inf if mean == 0 else z ** 2 * var / mean ** 2


def exact_summary(ab, il):
    """Means and variances, queries equally likely."""
    pa, pb = ab[:, 0].mean(), ab[:, 2].mean()
    ca, cb = ab[:, 1].mean(), ab[:, 3].mean()
    probs = il.reshape(-1, 3).mean(axis=0)
    mean = probs[0] - probs[2]
    var = probs[0] + probs[2] - mean ** 2
    return {"succ_a": pa, "succ_b": pb, "click_a": ca, "click_b": cb, "il_mean": mean, "il_var": var}


def simulate(ab, il, n=200_000, seed=2027):
    """Simulated sessions per arm (plan estimate)."""
    rng = np.random.default_rng(seed)
    nq = ab.shape[0]
    qa, qb = rng.integers(0, nq, n), rng.integers(0, nq, n)
    succ_a = rng.random(n) < ab[qa, 0]
    succ_b = rng.random(n) < ab[qb, 2]
    click_a = rng.random(n) < ab[qa, 1]
    click_b = rng.random(n) < ab[qb, 3]
    qi, fi = rng.integers(0, nq, n), rng.integers(0, len(FLIPS), n)
    cum = np.cumsum(il[qi, fi], axis=1)
    u = rng.random(n)
    o = np.where(u < cum[:, 0], 1, np.where(u < cum[:, 1], 0, -1))
    return {"succ_a": succ_a.mean(), "succ_b": succ_b.mean(), "click_a": click_a.mean(),
            "click_b": click_b.mean(), "il_mean": o.mean(), "il_var": o.var()}


def power_check(ab, il, n_ab, n_il, reps=1_000, seed=2027, alpha=0.05):
    """Share of simulated tests that reject."""
    rng = np.random.default_rng(seed)
    nq = ab.shape[0]
    z = norm.ppf(1 - alpha / 2)
    half = int(np.ceil(n_ab / 2))
    rej_ab = rej_il = 0
    cells = il.reshape(-1, 3)
    for _ in range(reps):
        na, nb = rng.multinomial(half, np.full(nq, 1 / nq)), rng.multinomial(half, np.full(nq, 1 / nq))
        xa, xb = rng.binomial(na, ab[:, 0]).sum(), rng.binomial(nb, ab[:, 2]).sum()
        p1, p2 = xa / half, xb / half
        pp = (xa + xb) / (2 * half)
        se = np.sqrt(pp * (1 - pp) * 2 / half)
        rej_ab += se > 0 and abs(p1 - p2) / se > z
        m = int(np.ceil(n_il))
        counts = rng.multinomial(m, np.full(len(cells), 1 / len(cells)))
        # one multinomial per (query, flips) cell
        out = rng.multinomial(counts, cells).sum(axis=0)
        wins, losses = out[0], out[2]
        mean = (wins - losses) / m
        var = (wins + losses) / m - mean ** 2
        rej_il += var > 0 and abs(mean) / np.sqrt(var / m) > z
    return rej_ab / reps, rej_il / reps
