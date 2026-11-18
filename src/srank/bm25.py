"""Okapi BM25 over the candidate pool."""
import math
import re
from collections import Counter

import numpy as np

STOP = frozenset("""a an and are as at be been but by can did do does for from had has have he her his how i
if in into is it its me my of on or our she so than that the their them then there these they this to
was we were what when where which who whom why will with would you your""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, texts):
        self.df = Counter()
        lengths = []
        for t in texts.values():
            tok = tokens(t)
            lengths.append(len(tok))
            self.df.update(set(tok))
        self.n = len(lengths)
        self.avgdl = float(np.mean(lengths))
        self.texts = texts

    def idf(self, term):
        # Lucene's variant, never negative
        d = self.df.get(term, 0)
        return math.log(1 + (self.n - d + 0.5) / (d + 0.5))

    def score(self, query, pids, k1=0.9, b=0.4):
        q = Counter(tokens(query))
        idf = {t: self.idf(t) for t in q}
        out = np.zeros(len(pids))
        for i, pid in enumerate(pids):
            tok = tokens(self.texts[pid])
            tf = Counter(tok)
            norm = k1 * (1 - b + b * len(tok) / self.avgdl)
            out[i] = sum(qn * idf[t] * tf[t] * (k1 + 1) / (tf[t] + norm) for t, qn in q.items() if tf[t])
        return out

    def rank(self, query, pids, k1=0.9, b=0.4):
        s = self.score(query, pids, k1, b)
        # stable sort on -score keeps file order among ties
        return [pids[i] for i in np.argsort(-s, kind="stable")]
