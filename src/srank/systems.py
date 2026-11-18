"""Rankings for Parts 1 to 3."""
import csv
from collections import defaultdict

from srank import DERIVED, RES, rank

MAIN = ["BM25", "CE", "Haiku-100", "CE>Haiku-20", "CE>Opus-20"]
COMPARISONS = [("P1", "CE", "BM25"), ("P2", "Haiku-100", "CE"), ("P3", "CE>Haiku-20", "CE"),
               ("P4", "CE>Opus-20", "CE"), ("P5", "CE>Opus-20", "CE>Haiku-20")]


def grades(path=RES / "llm_grades.csv"):
    """Grades and costs by model; failures are 0."""
    g, usd = defaultdict(dict), defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g[r["model"]][(r["qid"], r["pid"])] = int(r["grade"])
            usd[r["model"]][(r["qid"], r["pid"])] = float(r["usd_batch"])
    return g, usd


def ce_scores():
    s = defaultdict(dict)
    with open(DERIVED / "ce_scores.tsv", encoding="utf-8") as f:
        for line in f:
            q, p, x = line.rstrip("\n").split("\t")
            s[q][p] = float(x)
    return s


def build(qids, with_llm=True):
    bm = rank.load_runs("bm25_runs")
    ce = rank.load_runs("ce_run")
    cs = ce_scores()
    runs = {"BM25": {q: bm["bm25"][q] for q in qids}, "CE": {q: ce[q] for q in qids},
            "BM25 k1=1.2 b=0.75": {q: bm["bm25_k12_b75"][q] for q in qids},
            "BM25 k1=0.6 b=0.2": {q: bm["bm25_k06_b20"][q] for q in qids}}
    for k in (10, 20, 50, 100):
        runs[f"CE on BM25 top {k}"] = {q: rank.rerank_top(bm["bm25"][q], k, lambda p, q=q: cs[q][p]) for q in qids}
    if with_llm:
        g, _ = grades()
        h, o = g["haiku"], g["opus"]
        runs["Haiku-100"] = {q: rank.rerank_top(bm["bm25"][q], 100, lambda p, q=q: h[(q, p)]) for q in qids}
        runs["CE>Haiku-20"] = {q: rank.rerank_top(ce[q], 20, lambda p, q=q: h[(q, p)]) for q in qids}
        runs["CE>Opus-20"] = {q: rank.rerank_top(ce[q], 20, lambda p, q=q: o[(q, p)]) for q in qids}
    return runs


def arm_cost(qids):
    """USD per query for each LLM arm, batch prices."""
    bm = rank.load_runs("bm25_runs")["bm25"]
    ce = rank.load_runs("ce_run")
    _, usd = grades()
    def per_q(model, run, k):
        return sum(usd[model][(q, p)] for q in qids for p in run[q][:k]) / len(qids)
    return {"Haiku-100": per_q("haiku", bm, 100), "CE>Haiku-20": per_q("haiku", ce, 20),
            "CE>Opus-20": per_q("opus", ce, 20)}
