"""Which (query, passage) pairs each model grades."""
import numpy as np

SAMPLE_PER_GRADE = 250
SEED = 2027


def pool(d):
    """Judged pairs among the candidates (label pool)."""
    out = []
    for q in d["judged"]:
        cand = set(d["cand"][q])
        out += [(q, p) for p in sorted(d["qrels"][q], key=int) if p in cand]
    return out


def opus_sample(d, per_grade=SAMPLE_PER_GRADE, seed=SEED):
    rng = np.random.default_rng(seed)
    pl = pool(d)
    out = []
    for g in range(4):
        idx = [i for i, (q, p) in enumerate(pl) if d["qrels"][q][p] == g]
        out += [pl[i] for i in sorted(rng.choice(idx, size=per_grade, replace=False))]
    return out


def top(runs, qids, k):
    return [(q, p) for q in qids for p in runs[q][:k]]


def plan(d, bm25_run, ce_run, per_grade=SAMPLE_PER_GRADE):
    """Pairs per model, deduplicated, stable order."""
    def uniq(xs):
        return list(dict.fromkeys(xs))
    q = d["judged"]
    haiku = uniq(top(bm25_run, q, 100) + top(ce_run, q, 20) + pool(d))
    opus = uniq(top(ce_run, q, 20) + opus_sample(d, per_grade))
    return {"haiku": haiku, "opus": opus}
