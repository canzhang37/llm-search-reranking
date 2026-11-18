"""NDCG@10, MRR@10, Recall@100 as in trec_eval."""
import numpy as np

REL = 2  # grade >= 2 is relevant for MRR and recall


def dcg(gains):
    g = np.asarray(gains, dtype=float)
    return float(np.sum(g / np.log2(np.arange(2, g.size + 2))))


def ndcg(ranking, qrel, k=10):
    # linear gain, as in trec_eval
    gains = [qrel.get(p, 0) for p in ranking[:k]]
    ideal = dcg(sorted(qrel.values(), reverse=True)[:k])
    return dcg(gains) / ideal if ideal > 0 else 0.0


def mrr(ranking, qrel, k=10):
    for i, p in enumerate(ranking[:k]):
        if qrel.get(p, 0) >= REL:
            return 1.0 / (i + 1)
    return 0.0


def recall(ranking, qrel, k=100):
    total = sum(1 for g in qrel.values() if g >= REL)
    if total == 0:
        return 0.0
    return sum(1 for p in ranking[:k] if qrel.get(p, 0) >= REL) / total


def per_query(run, qrels, fn, qids):
    return np.array([fn(run[q], qrels[q]) for q in qids])
