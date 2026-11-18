"""BM25 runs for judged and pilot queries."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import bm25, data, rank  # noqa: E402

SETTINGS = {"bm25": (0.9, 0.4), "bm25_k12_b75": (1.2, 0.75), "bm25_k06_b20": (0.6, 0.2)}
PILOT = 2


def pilot_queries(d):
    # pilot: first two unjudged 2019 queries
    unjudged = sorted((q for q, y in d["year"].items() if y == "2019" and q not in d["qrels"]
                       and len(d["cand"].get(q, [])) >= 100), key=int)
    return unjudged[:PILOT]


def main():
    t0 = time.time()
    d = data.load()
    assert len(d["judged"]) == 97, len(d["judged"])
    model = bm25.BM25(d["text"])
    print(f"pool {model.n} passages, avgdl {model.avgdl:.1f} tokens ({time.time() - t0:.0f} s)")
    qids = d["judged"] + pilot_queries(d)
    runs = {name: {q: model.rank(d["query"][q], d["cand"][q], k1, b) for q in qids}
            for name, (k1, b) in SETTINGS.items()}
    rank.save_runs(runs, "bm25_runs")
    rank.save_runs({"pilot": pilot_queries(d)}, "pilot_queries")
    print(f"{len(qids)} queries ranked under {len(SETTINGS)} settings ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
