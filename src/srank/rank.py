"""Turn scores and grades into rankings."""
import json

import numpy as np

from srank import DERIVED


def by_score(pids, score):
    s = np.asarray(score, dtype=float)
    return [pids[i] for i in np.argsort(-s, kind="stable")]


def rerank_top(base, k, key):
    """Reorder top k by key; ties keep base order."""
    top = base[:k]
    order = sorted(range(len(top)), key=lambda i: (-key(top[i]), i))
    return [top[i] for i in order] + base[k:]


def save_runs(runs, name):
    DERIVED.mkdir(parents=True, exist_ok=True)
    (DERIVED / f"{name}.json").write_text(json.dumps(runs), encoding="utf-8")


def load_runs(name):
    return json.loads((DERIVED / f"{name}.json").read_text(encoding="utf-8"))
