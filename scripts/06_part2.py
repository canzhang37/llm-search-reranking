"""Part 2: Claude grades vs NIST grades."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import RES, data, metrics, pairs, stats, systems  # noqa: E402


def main():
    d = data.load()
    qids = d["judged"]
    g, _ = systems.grades()
    pool = pairs.pool(d)
    nist = np.array([d["qrels"][q][p] for q, p in pool])
    haiku = np.array([g["haiku"][(q, p)] for q, p in pool])
    grp = np.array([q for q, _ in pool])
    out = []

    # L1: Haiku on the whole pool
    k = stats.weighted_kappa(nist, haiku)
    lo, hi = stats.cluster_boot(lambda i: stats.weighted_kappa(nist[i], haiku[i]), grp)
    kb = stats.weighted_kappa(nist >= 2, haiku >= 2, k=2, weights="none")
    out.append({"id": "L1", "what": "Haiku vs NIST, linear-weighted kappa, 10,298 pairs", "value": k, "ci_low": lo,
                "ci_high": hi, "met": k >= 0.40})
    out.append({"id": "-", "what": "Haiku vs NIST, kappa for grade 2 or more", "value": kb})
    for y in ("2019", "2020"):
        m = np.array([d["year"][q] == y for q in grp])
        out.append({"id": "-", "what": f"Haiku vs NIST, linear-weighted kappa, {y}",
                    "value": stats.weighted_kappa(nist[m], haiku[m])})
    conf = pd.crosstab(pd.Series(nist, name="NIST"), pd.Series(haiku, name="Haiku"))
    conf.to_csv(RES / "part2_confusion_haiku.csv")

    # L2: Opus vs Haiku, reweighted to the pool mix
    samp = pairs.opus_sample(d)
    sn = np.array([d["qrels"][q][p] for q, p in samp])
    so = np.array([g["opus"][(q, p)] for q, p in samp])
    sh = np.array([g["haiku"][(q, p)] for q, p in samp])
    sg = np.array([q for q, _ in samp])
    mix = np.bincount(nist, minlength=4) / nist.size
    w = mix[sn] / (np.bincount(sn, minlength=4)[sn] / sn.size)
    ko = stats.weighted_kappa(sn, so, w=w)
    kh = stats.weighted_kappa(sn, sh, w=w)
    lo, hi = stats.cluster_boot(lambda i: stats.weighted_kappa(sn[i], so[i], w=w[i])
                                - stats.weighted_kappa(sn[i], sh[i], w=w[i]), sg)
    out.append({"id": "L2", "what": "Opus minus Haiku kappa, 1,000-pair sample, reweighted", "value": ko - kh,
                "ci_low": lo, "ci_high": hi, "met": lo > 0})
    out.append({"id": "-", "what": "Opus vs NIST, linear-weighted kappa, sample reweighted", "value": ko})
    out.append({"id": "-", "what": "Haiku vs NIST, linear-weighted kappa, sample reweighted", "value": kh})
    pd.crosstab(pd.Series(sn, name="NIST"), pd.Series(so, name="Opus")).to_csv(RES / "part2_confusion_opus_sample.csv")

    # L3 and L4: the same pool judged two ways
    q_nist, q_haiku = defaultdict(dict), defaultdict(dict)
    for (q, p), a, b in zip(pool, nist, haiku):
        q_nist[q][p] = int(a)
        q_haiku[q][p] = int(b)
    runs = systems.build(qids)
    names = list(runs)
    by_nist = {s: metrics.per_query(runs[s], q_nist, metrics.ndcg, qids) for s in names}
    by_haiku = {s: metrics.per_query(runs[s], q_haiku, metrics.ndcg, qids) for s in names}
    tab = pd.DataFrame({"ndcg10_nist_pool": {s: by_nist[s].mean() for s in names},
                        "ndcg10_haiku_labels": {s: by_haiku[s].mean() for s in names}})
    tab["rank_nist"] = tab.ndcg10_nist_pool.rank(ascending=False)
    tab["rank_haiku"] = tab.ndcg10_haiku_labels.rank(ascending=False)
    tab.round(4).to_csv(RES / "part2_system_orderings.csv")
    tau = stats.kendall(tab.ndcg10_nist_pool, tab.ndcg10_haiku_labels)
    out.append({"id": "L3", "what": "Kendall tau, 11 systems, NIST vs Haiku labels on the pool", "value": tau,
                "met": tau >= 0.80})
    gap = (by_haiku["CE>Haiku-20"] - by_haiku["CE"]) - (by_nist["CE>Haiku-20"] - by_nist["CE"])
    lo, hi = stats.boot_ci(gap)
    out.append({"id": "L4", "what": "gain of CE>Haiku-20 over CE, Haiku labels minus NIST labels", "value": gap.mean(),
                "ci_low": lo, "ci_high": hi, "met": lo > 0})
    gap_o = (by_haiku["CE>Opus-20"] - by_haiku["CE"]) - (by_nist["CE>Opus-20"] - by_nist["CE"])
    lo, hi = stats.boot_ci(gap_o)
    out.append({"id": "post hoc", "what": "same for CE>Opus-20 (a ranker not built on Haiku)", "value": gap_o.mean(),
                "ci_low": lo, "ci_high": hi})
    res = pd.DataFrame(out)
    res.round(4).to_csv(RES / "part2_results.csv", index=False)
    print(res.round(4).to_string(index=False))
    print(conf.to_string())
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
