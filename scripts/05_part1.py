"""Part 1: metrics per system, planned comparisons."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import RES, data, metrics, stats, systems  # noqa: E402

MEASURES = {"ndcg10": metrics.ndcg, "mrr10": metrics.mrr, "recall100": metrics.recall}
RULES = {"P1": lambda lo: lo >= 0.10, "P2": lambda lo: lo > -0.03, "P3": lambda lo: lo > 0,
         "P4": lambda lo: lo > 0, "P5": lambda lo: lo > 0}


def main():
    d = data.load()
    qids = d["judged"]
    runs = systems.build(qids)
    rows = []
    for name, run in runs.items():
        for m, fn in MEASURES.items():
            for q, v in zip(qids, metrics.per_query(run, d["qrels"], fn, qids)):
                rows.append({"system": name, "measure": m, "qid": q, "year": d["year"][q], "value": v})
    pq = pd.DataFrame(rows)
    pq.to_csv(RES / "part1_per_query.csv", index=False)
    summ = pq.groupby(["system", "measure"])["value"].mean().unstack()
    for y in ("2019", "2020"):
        summ[f"ndcg10_{y}"] = pq[(pq.measure == "ndcg10") & (pq.year == y)].groupby("system")["value"].mean()
    cost = systems.arm_cost(qids)
    summ["usd_per_1000_queries"] = pd.Series({k: v * 1000 for k, v in cost.items()})
    summ = summ.loc[list(runs)]
    summ.round(4).to_csv(RES / "part1_systems.csv")
    nd = pq[pq.measure == "ndcg10"].pivot(index="qid", columns="system", values="value").loc[qids]
    comp = []
    for key, a, b in systems.COMPARISONS:
        diff = (nd[a] - nd[b]).to_numpy()
        lo, hi = stats.boot_ci(diff)
        comp.append({"id": key, "comparison": f"{a} minus {b}", "mean_diff": diff.mean(), "ci_low": lo,
                     "ci_high": hi, "p": stats.signflip_p(diff), "wins": int((diff > 0).sum()),
                     "losses": int((diff < 0).sum()), "met": RULES[key](lo)})
    comp = pd.DataFrame(comp)
    comp["p_holm"] = stats.holm(comp["p"])
    comp.round(4).to_csv(RES / "part1_comparisons.csv", index=False)
    print(summ.round(4).to_string())
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
