"""TREC DL 2019/2020 queries, candidates, qrels."""
import gzip
from collections import defaultdict

from srank import RAW

YEARS = ("2019", "2020")


def queries(year):
    out = {}
    with gzip.open(RAW / f"msmarco-test{year}-queries.tsv.gz", "rt", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t")
            out[qid] = text
    return out


def qrels(year):
    out = defaultdict(dict)
    for line in open(RAW / f"{year}qrels-pass.txt", encoding="utf-8"):
        qid, _, pid, grade = line.split()
        out[qid][pid] = int(grade)
    return dict(out)


def candidates(year):
    """Candidate pids in file order, and texts."""
    cand, text = defaultdict(list), {}
    with gzip.open(RAW / f"msmarco-passagetest{year}-top1000.tsv.gz", "rt", encoding="utf-8") as f:
        for line in f:
            qid, pid, _, passage = line.rstrip("\n").split("\t")
            cand[qid].append(pid)
            text[pid] = passage
    return dict(cand), text


def load():
    """Both years together; query ids are unique."""
    d = {"query": {}, "year": {}, "cand": {}, "text": {}, "qrels": {}}
    for y in YEARS:
        q = queries(y)
        cand, text = candidates(y)
        d["query"].update(q)
        d["year"].update({qid: y for qid in q})
        d["cand"].update(cand)
        d["text"].update(text)
        d["qrels"].update(qrels(y))
    d["judged"] = sorted(d["qrels"], key=lambda q: (d["year"][q], int(q)))
    return d
