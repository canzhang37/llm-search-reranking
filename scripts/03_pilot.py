"""K3 pilot: parse check and cost projection."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srank import RES, data, llm, pairs, rank  # noqa: E402

CAP = 30.0


def main():
    d = data.load()
    bm = rank.load_runs("bm25_runs")["bm25"]
    ce = rank.load_runs("ce_run")
    pilot = rank.load_runs("pilot_queries")["pilot"]
    items = [(q, p, d["query"][q], d["text"][p]) for q in pilot for p in bm[q][:20]]
    cl = llm.client()
    lines = [f"pilot queries {pilot}: {[d['query'][q] for q in pilot]}"]
    per_req = {}
    for model in ("haiku", "opus"):
        todo = [it for it in items if llm.cached(model, it[0], it[1]) is None]
        llm.run_sync(cl, model, todo)
        rows = llm.table(model, [(q, p) for q, p, _, _ in items])
        status = {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}
        grades = np.bincount([r["grade"] for r in rows], minlength=4)
        tin = np.mean([r["input_tokens"] for r in rows])
        tout = np.mean([r["output_tokens"] for r in rows])
        per_req[model] = np.mean([r["usd_batch"] for r in rows])
        lines.append(f"{model}: status {status}, grades 0-3 {grades.tolist()}, mean input {tin:.0f} and output "
                     f"{tout:.0f} tokens, ${per_req[model] * 1000:.3f} per 1,000 requests at batch prices")
    plan = pairs.plan(d, bm, ce)
    total = 0.0
    for model, ps in plan.items():
        usd = per_req[model] * len(ps)
        total += usd
        lines.append(f"main run {model}: {len(ps)} pairs, projected ${usd:.2f}")
    lines.append(f"projected total ${total:.2f} against the ${CAP:.0f} cap: {'within' if total <= CAP else 'OVER'}")
    if total > CAP:
        small = pairs.plan(d, bm, ce, per_grade=125)
        alt = sum(per_req[m] * len(ps) for m, ps in small.items())
        lines.append(f"with the Opus sample at 500 pairs (K4): ${alt:.2f}")
    RES.mkdir(exist_ok=True)
    (RES / "k3_pilot.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
