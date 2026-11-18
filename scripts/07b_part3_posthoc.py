"""Post hoc: exact-rate sizes, order-aware A/B."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import RES, clicks, data, systems  # noqa: E402


def success_top(run, qrels, qids, user, k=3):
    """Mean P(click and stop in the top k)."""
    c, s = (np.asarray(x) for x in clicks.USERS[user])
    out = []
    for q in qids:
        g = clicks.grades(run[q], qrels[q], k)
        out.append(1 - np.prod(1 - c[g] * s[g]))
    return float(np.mean(out))


def main():
    d = data.load()
    qids = d["judged"]
    runs = systems.build(qids)
    comp = pd.read_csv(RES / "part1_comparisons.csv").set_index("id")
    rows = []
    for key, a, b in systems.COMPARISONS:
        for user in ("navigational", "informational"):
            ab, il = clicks.query_tables(runs[a], runs[b], d["qrels"], qids, user)
            ex = clicks.exact_summary(ab, il)
            n_ab = clicks.needed_ab(ex["succ_a"], ex["succ_b"])
            n_il = clicks.needed_il(ex["il_mean"], ex["il_var"])
            pw_ab, pw_il = clicks.power_check(ab, il, n_ab, n_il)
            nd = comp.loc[key, "mean_diff"]
            t_a = success_top(runs[a], d["qrels"], qids, user)
            t_b = success_top(runs[b], d["qrels"], qids, user)
            rows.append({"id": key, "user": user, "n_ab_exact": n_ab, "n_il_exact": n_il, "ratio": n_ab / n_il,
                         "power_ab": pw_ab, "power_il": pw_il,
                         "ab_sign_agrees_ndcg": np.sign(ex["succ_a"] - ex["succ_b"]) == np.sign(nd),
                         "succ_diff_exact": ex["succ_a"] - ex["succ_b"], "top3_diff_exact": t_a - t_b,
                         "n_ab_top3": clicks.needed_ab(t_a, t_b),
                         "top3_sign_agrees_ndcg": np.sign(t_a - t_b) == np.sign(nd)})
            print(f"{key} {user}: A/B {n_ab:,.0f}  interleaving {n_il:,.0f}  power {pw_ab:.3f}/{pw_il:.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "part3_posthoc.csv", index=False)
    k = res[["power_ab", "power_il"]].stack()
    print(f"power at exact sizes: {k.min():.3f}-{k.max():.3f}, within 0.75-0.85: {bool(((k >= .75) & (k <= .85)).all())}")


if __name__ == "__main__":
    main()
