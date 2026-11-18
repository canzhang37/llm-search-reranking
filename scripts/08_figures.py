"""Readout figures."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import FIG, RES, stats, systems  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
INK, ACCENT, MUTED = "#1f2a36", "#b5452c", "#8a96a3"
FIG.mkdir(exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, metadata={"Software": None})
    plt.close(fig)


# 1. NDCG@10 and API cost
pq = pd.read_csv(RES / "part1_per_query.csv", dtype={"qid": str})
summ = pd.read_csv(RES / "part1_systems.csv", index_col=0)
nd = pq[pq.measure == "ndcg10"]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
for i, s in enumerate(systems.MAIN):
    v = nd[nd.system == s].value.to_numpy()
    lo, hi = stats.boot_ci(v)
    llm = s not in ("BM25", "CE")
    ax.errorbar(v.mean(), i, xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", ms=6, capsize=3,
                color=ACCENT if llm else INK)
    usd = summ.loc[s, "usd_per_1000_queries"]
    cost = "no API cost" if np.isnan(usd) else f"${usd:.2f} per 1,000 queries"
    ax.text(hi + 0.012, i, cost, va="center", fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(systems.MAIN)), systems.MAIN)
ax.invert_yaxis()
ax.set_xlim(0.4, 0.95)
ax.set_xlabel("NDCG@10, 97 TREC DL 2019-2020 queries (95% bootstrap interval)")
save(fig, "fig1_quality_and_cost.png")

# 2. Haiku vs NIST, judged pairs
conf = pd.read_csv(RES / "part2_confusion_haiku.csv", index_col=0)
conf = conf.reindex(index=range(4), columns=[str(c) for c in range(4)], fill_value=0)
share = conf.div(conf.sum(axis=1), axis=0).to_numpy()
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
for r in range(4):
    for c in range(4):
        ax.text(c, r, f"{share[r, c]:.0%}\n{conf.iat[r, c]:,}", ha="center", va="center", fontsize=7,
                color="white" if share[r, c] > 0.55 else INK)
ax.set_xticks(range(4), range(4))
ax.set_yticks(range(4), range(4))
ax.set_xlabel("Haiku grade")
ax.set_ylabel("NIST grade")
ax.set_title("Row shares and counts", fontsize=8.5, loc="left")
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)
save(fig, "fig2_haiku_vs_nist.png")

# 3. sessions for 80% power, exact rates
post = pd.read_csv(RES / "part3_posthoc.csv")
ids = [c[0] for c in systems.COMPARISONS]
labels = [f"{k}: {a} vs {b}" for k, a, b in systems.COMPARISONS]
clear = pd.read_csv(RES / "part1_comparisons.csv").set_index("id")
clear = (clear.ci_low > 0) | (clear.ci_high < 0)
XMAX = 3e6
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True, sharex=True)
for ax, u in zip(axes, ["navigational", "informational"]):
    sub = post[post.user == u].set_index("id").loc[ids]
    for i, k in enumerate(ids):
        r = sub.loc[k]
        xs = [min(x, XMAX) for x in (r.n_ab_exact, r.n_ab_top3, r.n_il_exact)]
        ax.plot([min(xs), max(xs)], [i, i], color=MUTED, lw=1, zorder=1)
        ax.plot(xs[0], i, "o", mfc="white", mec=INK, ms=6, zorder=2)
        ax.plot(xs[1], i, "s" if xs[1] < XMAX else ">", mfc="white", mec=ACCENT, ms=6, zorder=2)
        ax.plot(xs[2], i, "o", color=ACCENT, ms=6, zorder=3)
        if r.n_ab_top3 > XMAX:
            ax.annotate(f"{r.n_ab_top3 / 1e6:,.0f} million", (XMAX, i), xytext=(-2, 7),
                        textcoords="offset points", fontsize=6.5, color=MUTED, ha="right")
        if clear[k] and not r.ab_sign_agrees_ndcg:
            ax.annotate("A/B success favours CE", (xs[0], i), xytext=(0, -12), textcoords="offset points",
                        fontsize=6.5, color=INK, ha="center")
    ax.set_xscale("log")
    ax.set_xlim(30, XMAX * 1.6)
    ax.set_title(f"{u} user", fontsize=8.5)
    ax.set_xlabel("Sessions (log scale)")
axes[0].set_yticks(range(len(ids)), labels, fontsize=7.5)
axes[0].invert_yaxis()
handles = [plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, label="A/B test, session success (planned)"),
           plt.Line2D([], [], ls="", marker="s", mfc="white", mec=ACCENT, label="A/B test, success in the top 3 (added later)"),
           plt.Line2D([], [], ls="", marker="o", color=ACCENT, label="team-draft interleaving")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG / "fig3_sessions_needed.png", dpi=200, metadata={"Software": None})
plt.close(fig)
print("figures written to", FIG)
