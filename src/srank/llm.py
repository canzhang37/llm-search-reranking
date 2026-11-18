"""Claude grades, one request per pair."""
import csv
import json
import logging
import os
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from srank import DERIVED, ROOT

log = logging.getLogger("srank.llm")
MODELS = {"haiku": "claude-haiku-5-5", "opus": "claude-opus-5-5"}
TAG = {"haiku": "h", "opus": "o"}
PRICE = {"haiku": (0.10, 0.50), "opus": (4.00, 20.00)}  # USD per million tokens, standard
EFFORT = "low"
MAX_TOKENS = 4000

SYSTEM = "You judge how well a passage answers a web search query, as a search quality rater would."

# wording from the TREC DL assessor guidelines
SCALE = """3 = Perfectly relevant: the passage is dedicated to the query and contains the exact answer.
2 = Highly relevant: the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
1 = Related: the passage seems related to the query but does not answer it.
0 = Irrelevant: the passage has nothing to do with the query."""

SCHEMA = {
    "type": "object",
    "properties": {"grade": {"type": "integer", "enum": [0, 1, 2, 3]}},
    "required": ["grade"],
    "additionalProperties": False,
}


def user_message(query, passage):
    return f"Query: {query}\n\nPassage: {passage}\n\nGrade the passage on this scale:\n{SCALE}\n\nReturn the grade."


def params(model, query, passage):
    return {
        "model": MODELS[model],
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": user_message(query, passage)}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def cid(model, qid, pid):
    return f"{TAG[model]}-{qid}-{pid}"


def store(model):
    p = DERIVED / "llm" / model
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse(rec):
    """Grade or None, usage, status."""
    if rec.get("type") != "succeeded":
        return None, {}, rec.get("type", "missing")
    m = rec["message"]
    usage = m.get("usage", {})
    if m.get("stop_reason") != "end_turn":
        return None, usage, m.get("stop_reason") or "no_stop"
    text = next((b["text"] for b in m["content"] if b["type"] == "text"), "")
    try:
        g = int(json.loads(text)["grade"])
    except (ValueError, KeyError, TypeError):
        return None, usage, "invalid"
    return (g, usage, "ok") if g in (0, 1, 2, 3) else (None, usage, "invalid")


def cached(model, qid, pid):
    f = store(model) / f"{cid(model, qid, pid)}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def load_env(path=ROOT / ".env"):
    if "ANTHROPIC_API_KEY" in os.environ or not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
            os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"')


def client():
    load_env()
    return anthropic.Anthropic(max_retries=4)


def run_sync(cl, model, items):
    """Pilot: one request at a time."""
    for qid, pid, query, passage in items:
        msg = cl.messages.create(**params(model, query, passage))
        rec = {"custom_id": cid(model, qid, pid), "type": "succeeded", "message": msg.to_dict(), "sync": True}
        (store(model) / f"{rec['custom_id']}.json").write_text(json.dumps(rec), encoding="utf-8")


def run_batch(cl, model, items, every=60):
    """Submit unstored pairs, wait, store results."""
    todo = [it for it in items if cached(model, it[0], it[1]) is None]
    if not todo:
        return None
    reqs = [Request(custom_id=cid(model, q, p), params=MessageCreateParamsNonStreaming(**params(model, qt, pt)))
            for q, p, qt, pt in todo]
    batch = cl.messages.batches.create(requests=reqs)
    log.info("batch %s: %d %s requests", batch.id, len(reqs), model)
    while True:
        b = cl.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        log.info("batch %s: %d processing", batch.id, b.request_counts.processing)
        time.sleep(every)
    for r in cl.messages.batches.results(batch.id):
        rec = {"custom_id": r.custom_id, "type": r.result.type, "batch": batch.id}
        if r.result.type == "succeeded":
            rec["message"] = r.result.message.to_dict()
        elif r.result.type == "errored":
            rec["error"] = r.result.error.to_dict()
        (store(model) / f"{r.custom_id}.json").write_text(json.dumps(rec), encoding="utf-8")
    return batch.id


def retry_errored(cl, model, items):
    """Resubmit errored or expired pairs once."""
    bad = []
    for it in items:
        rec = cached(model, it[0], it[1])
        if rec is not None and rec.get("type") in ("errored", "expired", "canceled") and not rec.get("retried"):
            (store(model) / f"{cid(model, it[0], it[1])}.json").unlink()
            bad.append(it)
    if bad:
        run_batch(cl, model, bad)
        for it in bad:
            f = store(model) / f"{cid(model, it[0], it[1])}.json"
            rec = json.loads(f.read_text(encoding="utf-8"))
            rec["retried"] = True
            f.write_text(json.dumps(rec), encoding="utf-8")
    return len(bad)


def usd(model, usage, batch=True):
    pin, pout = PRICE[model]
    tin = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    x = (tin * pin + usage.get("output_tokens", 0) * pout) / 1e6
    return x / 2 if batch else x


def table(model, pairs):
    """One row per pair; failures as grade 0."""
    rows = []
    for qid, pid in pairs:
        rec = cached(model, qid, pid)
        g, usage, status = parse(rec) if rec else (None, {}, "missing")
        rows.append({"model": model, "qid": qid, "pid": pid, "grade": 0 if g is None else g, "status": status,
                     "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
                     "usd_batch": round(usd(model, usage), 7)})
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
