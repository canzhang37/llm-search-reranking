import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srank import bm25, clicks, metrics, stats  # noqa: E402


def test_ndcg_by_hand():
    qrel = {"a": 3, "b": 2, "c": 1, "z": 2}
    run = ["b", "x", "a"]
    got = metrics.ndcg(run, qrel)
    want = (2 / 1 + 0 + 3 / 2) / (3 / 1 + 2 / math.log2(3) + 2 / 2 + 1 / math.log2(5))
    assert math.isclose(got, want)


def test_mrr_and_recall_use_grade_two():
    qrel = {"a": 1, "b": 2, "c": 3}
    assert metrics.mrr(["a", "b"], qrel) == 0.5
    assert metrics.recall(["c", "a"], qrel, k=100) == 0.5


def test_bm25_prefers_rarer_matching_term():
    texts = {"1": "cats and dogs", "2": "dogs dogs dogs", "3": "cats like fish", "4": "dogs bark"}
    m = bm25.BM25(texts)
    # cats is rarer than dogs here
    order = m.rank("cats dogs", ["2", "1", "3", "4"])
    assert order[0] == "1"
    assert m.idf("cats") > m.idf("dogs") > 0


def test_holm_matches_definition():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_signflip_detects_shift_and_not_noise():
    rng = np.random.default_rng(0)
    assert stats.signflip_p(rng.normal(0.5, 1, 80), n=2000) < 0.01
    assert stats.signflip_p(rng.normal(0.0, 1, 80), n=2000) > 0.05


def test_weighted_kappa_perfect_and_chance():
    a = np.array([0, 1, 2, 3] * 25)
    assert math.isclose(stats.weighted_kappa(a, a), 1.0)
    rng = np.random.default_rng(1)
    assert abs(stats.weighted_kappa(rng.integers(0, 4, 4000), rng.integers(0, 4, 4000))) < 0.05


def test_team_draft_alternates_and_credits():
    a = ["a1", "a2", "a3", "s", "a4", "a5"]
    b = ["s", "b1", "b2", "b3", "b4", "b5"]
    merged, team = clicks.team_draft(a, b, (0, 0, 0, 0, 0))
    assert merged[:4] == ["a1", "s", "a2", "b1"]
    assert list(team[:4]) == [0, 1, 0, 1]
    assert len(set(merged)) == len(merged) == 10


def test_interleaving_dp_matches_simulation():
    g = np.array([3, 0, 2, 1, 0, 3, 0, 0, 1, 2])
    team = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    exact = clicks.interleave_outcome(g, team, "informational")
    c, s = (np.asarray(x)[g] for x in clicks.USERS["informational"])
    rng = np.random.default_rng(3)
    n = 100_000
    out = np.zeros(3)
    for _ in range(n // 1000):
        click = rng.random((1000, 10)) < c
        stop = rng.random((1000, 10)) < s
        alive = np.ones(1000, bool)
        diff = np.zeros(1000, int)
        for i in range(10):
            hit = alive & click[:, i]
            diff += np.where(hit, 1 if team[i] == 0 else -1, 0)
            alive &= ~(hit & stop[:, i])
        out += [(diff > 0).sum(), (diff == 0).sum(), (diff < 0).sum()]
    assert np.allclose(out / n, exact, atol=0.006)
    assert math.isclose(sum(exact), 1.0)


def test_cascade_success_probability():
    g = np.array([3, 3])
    succ, anyc = clicks.success_prob(g, "navigational")
    c, s = 0.95, 0.90
    assert math.isclose(succ, c * s + (1 - c * s) * c * s)
    assert math.isclose(anyc, 1 - (1 - c) ** 2)


def test_sample_size_formulas():
    assert math.isinf(clicks.needed_ab(0.3, 0.3))
    n = clicks.needed_il(0.1, 0.5)
    assert math.isclose(n, (1.959964 + 0.841621) ** 2 * 0.5 / 0.01, rel_tol=1e-5)


def test_simulation_and_power_check_agree_with_exact():
    rng = np.random.default_rng(5)
    pids = [f"p{i}" for i in range(30)]
    # sparse relevance, two close rankings
    qrels = {f"q{j}": {p: int(rng.choice(4, p=[0.75, 0.15, 0.07, 0.03])) for p in pids} for j in range(40)}
    good = {q: sorted(pids, key=lambda p: -qrels[q][p] + rng.normal(0, 1.5)) for q in qrels}
    bad = {q: sorted(pids, key=lambda p: -qrels[q][p] + rng.normal(0, 2.5)) for q in qrels}
    ab, il = clicks.query_tables(good, bad, qrels, list(qrels), "navigational")
    ex = clicks.exact_summary(ab, il)
    sim = clicks.simulate(ab, il, n=200_000)
    assert abs(sim["il_mean"] - ex["il_mean"]) < 0.01
    assert ex["il_mean"] > 0 and ex["succ_a"] > ex["succ_b"]
    n_ab = clicks.needed_ab(ex["succ_a"], ex["succ_b"])
    n_il = clicks.needed_il(ex["il_mean"], ex["il_var"])
    pab, pil = clicks.power_check(ab, il, n_ab, n_il, reps=400)
    assert 0.7 < pab < 0.9 and 0.7 < pil < 0.9
