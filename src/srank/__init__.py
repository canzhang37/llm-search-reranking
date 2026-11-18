"""TREC DL reranking: BM25, cross-encoder, Claude."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
RES = ROOT / "results"
FIG = ROOT / "figures"
