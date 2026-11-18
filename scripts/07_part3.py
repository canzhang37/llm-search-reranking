"""Part 3: sessions for A/B and interleaving."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import RES, clicks, data, systems  # noqa: E402


def main():
    d = data.load()
    qids = d["judged"]
    runs = systems.build(qids)
    comp = pd.read_csv(RES / "part1_comparisons.csv").set_index("id")
    rows = []
    for key, a, b in systems.COMPARISONS:
        clear = comp.loc[key, "ci_low"] > 0 or comp.loc[key, "ci_high"] < 0
        for user in clicks.USERS:
            ab, il = clicks.query_tables(runs[a], runs[b], d["qrels"], qids, user)
            ex = clicks.exact_summary(ab, il)
            sim = clicks.simulate(ab, il)
            n_ab = clicks.needed_ab(sim["succ_a"], sim["succ_b"])
            n_click = clicks.needed_ab(sim["click_a"], sim["click_b"])
            n_il = clicks.needed_il(sim["il_mean"], sim["il_var"])
            pw_ab, pw_il = (clicks.power_check(ab, il, n_ab, n_il) if np.isfinite(n_ab) and np.isfinite(n_il)
                            and max(n_ab, n_il) < 5e8 else (np.nan, np.nan))
            rows.append({"id": key, "comparison": f"{a} vs {b}", "user": user, "ndcg_diff": comp.loc[key, "mean_diff"],
                         "ndcg_clear": clear, "succ_a": sim["succ_a"], "succ_b": sim["succ_b"],
                         "succ_a_exact": ex["succ_a"], "succ_b_exact": ex["succ_b"], "il_mean": sim["il_mean"],
                         "il_mean_exact": ex["il_mean"], "n_ab_success": n_ab, "n_ab_click": n_click,
                         "n_interleaving": n_il, "ratio_ab_to_il": n_ab / n_il, "power_ab_at_n": pw_ab,
                         "power_il_at_n": pw_il,
                         "sign_agrees": np.sign(sim["il_mean"]) == np.sign(comp.loc[key, "mean_diff"])})
            print(f"{key} {user}: A/B {n_ab:,.0f}  interleaving {n_il:,.0f}  power {pw_ab:.3f}/{pw_il:.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "part3_sessions.csv", index=False)
    clear = res[res.ndcg_clear]
    o1 = bool((clear.ratio_ab_to_il >= 5).all()) if len(clear) else None
    o2 = bool(clear[clear.user.isin(["perfect", "navigational"])].sign_agrees.all()) if len(clear) else None
    k = res[["power_ab_at_n", "power_il_at_n"]].stack()
    check = bool(((k >= 0.75) & (k <= 0.85)).all())
    lines = [f"O1 (interleaving at most 1/5 of A/B sessions, every clear comparison, all users): {o1}",
             f"O2 (interleaving sign matches NDCG, clear comparisons, perfect and navigational): {o2}",
             f"power check at computed sizes within 0.75-0.85: {check} (range {k.min():.3f}-{k.max():.3f})"]
    (RES / "part3_verdicts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
