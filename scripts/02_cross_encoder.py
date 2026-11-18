"""Cross-encoder scores, and check K1."""
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import DERIVED, RES, data, metrics, rank  # noqa: E402

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
PUBLISHED_DL19 = 0.7430


def main():
    from sentence_transformers import CrossEncoder
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = data.load()
    pilot = rank.load_runs("pilot_queries")["pilot"]
    qids = d["judged"] + pilot
    ce = CrossEncoder(MODEL, device="cpu", max_length=512)
    out = DERIVED / "ce_scores.tsv"
    t0 = time.time()
    scores = {}
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for i, q in enumerate(qids):
            pids = d["cand"][q]
            s = ce.predict([(d["query"][q], d["text"][p]) for p in pids], batch_size=64, show_progress_bar=False)
            scores[q] = dict(zip(pids, map(float, s)))
            w.writerows((q, p, f"{x:.6f}") for p, x in scores[q].items())
            if i % 10 == 9:
                print(f"{i + 1}/{len(qids)} queries, {time.time() - t0:.0f} s", flush=True)
    secs = time.time() - t0
    runs = {q: rank.by_score(d["cand"][q], [scores[q][p] for p in d["cand"][q]]) for q in qids}
    rank.save_runs(runs, "ce_run")
    q19 = [q for q in d["judged"] if d["year"][q] == "2019"]
    got = metrics.per_query(runs, d["qrels"], metrics.ndcg, q19).mean()
    npairs = sum(len(d["cand"][q]) for q in qids)
    ok = abs(got - PUBLISHED_DL19) <= 0.02
    RES.mkdir(exist_ok=True)
    (RES / "k1_cross_encoder_check.txt").write_text(
        f"cross-encoder {MODEL}\nTREC DL 2019 NDCG@10 here: {got:.4f}\npublished on the model card: {PUBLISHED_DL19}\n"
        f"difference {got - PUBLISHED_DL19:+.4f} (K1 passes if within 0.02): {'pass' if ok else 'FAIL'}\n"
        f"{npairs} pairs scored on CPU in {secs:.0f} s\n", encoding="utf-8")
    print(f"K1: DL19 NDCG@10 {got:.4f} vs {PUBLISHED_DL19} -> {'pass' if ok else 'FAIL'}; {npairs} pairs, {secs:.0f} s")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
