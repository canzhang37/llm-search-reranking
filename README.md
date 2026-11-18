# Does an LLM reranker earn its cost? Offline gains, judge bias and online test size on TREC DL

A search team that adds a large language model to its ranking stack has three questions to answer:

- Where in the stack does the model help, and is the gain worth the bill?
- Can the same kind of model be trusted to judge the result?
- How much traffic would an online test need to confirm it?

This project asks all three on the 97 judged queries of the TREC Deep Learning 2019 and 2020 passage tracks. It compares BM25, a small trained cross-encoder (MiniLM, MS MARCO), and Claude Haiku 5.5 and Opus 5.5 grading passages through the Message Batches API. The plan and eleven predictions were committed before any ranking was scored ([docs/design.md](docs/design.md)). The results are in [docs/readout.md](docs/readout.md).

What came out:

- **Claude helps on a short list, not on a cheap first stage.** Haiku regrading the cross-encoder's top 20 raised NDCG@10 from 0.741 to 0.768 (+0.027, 95% CI 0.013 to 0.042) for $0.59 per 1,000 queries. Opus did no better (+0.002 over Haiku, CI -0.014 to 0.020) at $23 per 1,000 queries. Haiku on BM25's top 100 scored 0.685, below the cross-encoder alone. Most of that loss came from BM25 missing passages the cross-encoder found deeper in the list.
- **Claude labels rank systems well but favour Claude's own ranker.** Haiku agreed with NIST assessors only moderately (weighted kappa 0.35, mostly from calling irrelevant passages "related"). Even so, it ordered 11 systems almost as NIST did (Kendall tau 0.89). But it tripled the measured gain of the Haiku reranker (0.081 against 0.028 under NIST labels). The Opus reranker, judged by Haiku, was not inflated.
- **The online metric mattered more than the test design.** In a cascade-click simulation, the planned A/B metric (did the session end in a satisfied click) depends only on which ten results are shown, not their order. For one comparison it favoured the worse ranker. An order-aware A/B metric and team-draft interleaving both pointed the right way. Interleaving was never the least sensitive of the three, but it needed a fifth of the A/B sessions in only 2 of 8 cases, so the planned prediction failed.

Four of the eleven predictions failed and are reported as such. Analyses added after the planned run are labelled post hoc. The API bill for the whole project was about $3.60.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows path; bin/ elsewhere
mkdir -p data/raw && cd data/raw
curl -O https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
curl -O https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
curl -O https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz
curl -O https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv.gz
curl -O https://trec.nist.gov/data/deep/2019qrels-pass.txt
curl -O https://trec.nist.gov/data/deep/2020qrels-pass.txt
cd ../..
.venv/Scripts/python scripts/01_first_stage.py     # BM25 over the candidate pool, about 15 seconds
.venv/Scripts/python scripts/02_cross_encoder.py   # 93,066 pairs, about 10 minutes on a laptop CPU
# ANTHROPIC_API_KEY=... in a .env file at the project root (never committed)
.venv/Scripts/python scripts/03_pilot.py           # 80 requests, projects the cost
.venv/Scripts/python scripts/04_llm_batches.py     # 17,767 batch requests, about 20 minutes, about $3.50
.venv/Scripts/python scripts/05_part1.py
.venv/Scripts/python scripts/05b_part1_posthoc.py
.venv/Scripts/python scripts/06_part2.py
.venv/Scripts/python scripts/07_part3.py
.venv/Scripts/python scripts/07b_part3_posthoc.py
.venv/Scripts/python scripts/08_figures.py
.venv/Scripts/python -m pytest tests
```

Claude's grades are in `results/llm_grades.csv`. With that file in place, scripts 05 to 08 rerun without an API key. A fresh run of 03 and 04 will not reproduce every grade, because the models do not return identical output on every call.

## Data

The data are the MS MARCO passage collection test queries and top-1000 candidate files, and the TREC Deep Learning 2019 and 2020 relevance judgments from NIST. MS MARCO is released for non-commercial research. No passage text is in this repository, and the only query text is the two pilot queries in `results/k3_pilot.txt`. The other results files hold query and passage ids, grades and scores.

## References

- Craswell, Mitra, Yilmaz, Campos, Voorhees. Overview of the TREC 2019 deep learning track. arXiv:2003.07820. The 2020 overview is arXiv:2102.07662.
- Thomas, Spielman, Craswell, Mitra. Large language models can accurately predict searcher preferences. SIGIR 2024. arXiv:2309.10621.
- Upadhyay, Pradeep, Thakur, Craswell, Lin. UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor. arXiv:2406.06519.
- Clarke, Dietz. LLM-based relevance assessment still can't replace human relevance assessment. arXiv:2412.17156.
- Craswell, Zoeter, Taylor, Ramsey. An experimental comparison of click position-bias models. WSDM 2008.
- Radlinski, Kurup, Joachims. How does clickthrough data reflect retrieval quality? CIKM 2008.
- Chapelle, Joachims, Radlinski, Yue. Large-scale validation and analysis of interleaved search evaluation. ACM TOIS 30(1), 2012.
- Hofmann. Fast and reliable online learning to rank for information retrieval. PhD thesis, University of Amsterdam, 2013.

Can Zhang, zhan8639@umn.edu. MIT license for the code.
