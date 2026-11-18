"""Post hoc: Haiku vs CE on the same BM25 top 100."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import RES, data, rank, stats, systems  # noqa: E402


def main():
    d = data.load()
    qids = d["judged"]
    pq = pd.read_csv(RES / "part1_per_query.csv", dtype={"qid": str})
    nd = pq[pq.measure == "ndcg10"].pivot(index="qid", columns="system", values="value").loc[qids]
    diff = (nd["Haiku-100"] - nd["CE on BM25 top 100"]).to_numpy()
    lo, hi = stats.boot_ci(diff)
    g, _ = systems.grades()
    runs = systems.build(qids)
    tied = []
    for q in qids:
        top = [g["haiku"][(q, p)] for p in runs["Haiku-100"][q][:10]]
        n3 = sum(g["haiku"][(q, p)] == 3 for p in rank.load_runs("bm25_runs")["bm25"][q][:100])
        tied.append({"qid": q, "grade3_in_top100": n3, "distinct_grades_in_top10": len(set(top))})
    t = pd.DataFrame(tied)
    lines = [f"Haiku-100 minus CE on BM25 top 100, NDCG@10: {diff.mean():.4f} (95% CI {lo:.4f} to {hi:.4f}, "
             f"p = {stats.signflip_p(diff):.4f}, wins {int((diff > 0).sum())}, losses {int((diff < 0).sum())})",
             f"passages Haiku grades 3 in the BM25 top 100: median {t.grade3_in_top100.median():.0f}, "
             f"queries with more than 10: {(t.grade3_in_top100 > 10).sum()} of {len(t)}",
             f"queries whose Haiku-100 top 10 is one grade throughout: {(t.distinct_grades_in_top10 == 1).sum()}"]
    (RES / "part1_posthoc.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
