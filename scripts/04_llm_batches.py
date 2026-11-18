"""LLM grades via the Batches API, one retry."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import RES, data, llm, pairs, rank  # noqa: E402

PER_GRADE = 250  # 125 only if the pilot breaks the cap (K4)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = data.load()
    bm = rank.load_runs("bm25_runs")["bm25"]
    ce = rank.load_runs("ce_run")
    plan = pairs.plan(d, bm, ce, per_grade=PER_GRADE)
    cl = llm.client()
    rows = []
    for model in ("haiku", "opus"):
        items = [(q, p, d["query"][q], d["text"][p]) for q, p in plan[model]]
        llm.run_batch(cl, model, items)
        n = llm.retry_errored(cl, model, items)
        print(f"{model}: {len(items)} pairs, {n} resubmitted")
        rows += llm.table(model, plan[model])
    RES.mkdir(exist_ok=True)
    llm.write_csv(rows, RES / "llm_grades.csv")
    for model in ("haiku", "opus"):
        r = [x for x in rows if x["model"] == model]
        bad = sum(x["status"] != "ok" for x in r)
        print(f"{model}: {len(r)} grades, {bad} counted as 0 after failure, ${sum(x['usd_batch'] for x in r):.2f}")


if __name__ == "__main__":
    main()
