# Design

Written after reading the file formats and counting queries, judgments and candidate passages (numbers below). No ranking had been scored against the judgments, and no model had been run on any query, before this plan was committed.

## Questions

A search team that adds a large language model (LLM) to its ranking stack has to answer three questions, and this project asks them on public data:

1. Does an LLM reranker beat a small trained cross-encoder, used alone on a cheap first stage or on top of the cross-encoder, and is the gain worth its cost?
2. Can LLM relevance labels stand in for human labels when comparing rankers, and do they favour a ranker built on the same LLM?
3. How many user sessions would an online test need to see the offline differences, for an A/B test and for team-draft interleaving?

## Data

TREC Deep Learning 2019 and 2020, passage ranking (MS MARCO v1 passage collection), from the MS MARCO and TREC sites:

- Queries: 200 test queries per year. The 43 (2019) and 54 (2020) queries with NIST judgments are the evaluation set, 97 queries in all.
- Candidates: the official top-1000 file of each year (query, passage, text), 189,877 and 190,699 rows. Judged queries have a median of 1,000 candidates (fewest: 5 in 2019, 28 in 2020). The files carry no rank or score.
- Judgments: grades 0 to 3. 2019: 9,260 judgments, 4,612 of them on candidate passages. 2020: 11,386 and 5,686. Every judged query has at least one passage at grade 2 or more. Of the grade 2 and 3 passages, 1,612 of 2,501 (2019) and 959 of 1,666 (2020) are among the candidates, so no reranker of these lists can reach an NDCG of 1.

The 103 test queries per year without judgments are used only for the pilot.

## Metrics

- Primary: NDCG@10 with gain equal to the grade (0, 1, 2, 3), as in trec_eval and the TREC DL overviews. The ideal ranking uses all judgments of the query, including passages outside the candidates. Unjudged passages have grade 0.
- Secondary: MRR@10 and Recall@100, with grade 2 or more counted as relevant (the TREC DL passage convention).
- Results pooled over the 97 queries (primary) and per year.

## Systems

- **BM25**: Okapi BM25 recomputed over the union of both years' candidate passages (inverse document frequencies from that pool), k1 = 0.9 and b = 0.4 (Anserini's MS MARCO defaults), lowercase alphanumeric tokens, a fixed English stopword list, no stemming. It orders all candidates of a query.
- **CE**: the cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2` scores every candidate of a query (so it reranks the BM25 top 1,000), ordered by score.
- **Haiku-100**: Claude Haiku 5.5 grades each passage in the BM25 top 100. Passages are ordered by grade, ties by BM25 rank; ranks after 100 keep the BM25 order. This is an LLM placed directly on a cheap first stage.
- **CE→Haiku-20**: Haiku grades the CE top 20, ordered by grade, ties by CE score; ranks after 20 keep the CE order.
- **CE→Opus-20**: the same with Claude Opus 5.5.

LLM requests: one request per (query, passage) pair, models `claude-haiku-5-5` and `claude-opus-5-5`, effort `low`, thinking at the model default, no sampling parameters, structured JSON output with a single integer grade from 0 to 3, through the Message Batches API. The prompt gives the query, the passage and the four grade definitions from the TREC DL assessor guidelines (perfectly relevant, highly relevant, related, irrelevant). A request that errors is resubmitted once. A request that still fails, is refused or does not parse counts as grade 0, and the number of such requests is reported. Cost is reported per 1,000 queries at batch list prices. The cross-encoder runs on a laptop CPU; its time is reported, not priced.

## Part 1: comparisons and predictions

Each comparison is paired over the 97 queries: the mean difference in NDCG@10, a 95% interval from 10,000 bootstrap resamples of queries (percentile), and a two-sided p-value from a paired sign-flip randomization test (10,000 flips), with Holm's correction over the five comparisons.

| | Comparison | Met if |
|---|---|---|
| P1 | CE minus BM25 | lower 95% bound at least 0.10 |
| P2 | Haiku-100 minus CE | lower 95% bound above -0.03 (an LLM on BM25's top 100 is not worse than the trained reranker by more than 0.03) |
| P3 | CE→Haiku-20 minus CE | lower 95% bound above 0 |
| P4 | CE→Opus-20 minus CE | lower 95% bound above 0 |
| P5 | CE→Opus-20 minus CE→Haiku-20 | lower 95% bound above 0 |

## Part 2: LLM labels against NIST labels

- Label pool: the 10,298 judged pairs whose passage is among the candidates. Haiku grades all of them with the Part 1 prompt and settings (a pair already graded in Part 1 reuses that grade, since the request is identical). Opus grades a stratified random sample of 250 pairs per NIST grade (1,000 pairs, seed 2027).
- Agreement: linear-weighted Cohen's kappa over the four grades, Cohen's kappa for grade 2 or more, and the confusion matrix. Opus and Haiku are compared on the same 1,000 pairs, weighted back to the NIST grade mix of the pool. Intervals come from bootstrap resamples of queries.
- System ranking: 11 systems are scored by NDCG@10 twice on the same pool, once with NIST grades and once with Haiku grades (passages outside the pool have grade 0 under both). The systems: BM25 with (k1, b) = (0.9, 0.4), (1.2, 0.75) and (0.6, 0.2); the cross-encoder applied to the BM25 top 10, 20, 50, 100 and 1,000 (the last is CE); Haiku-100, CE→Haiku-20 and CE→Opus-20.

| | Prediction | Met if |
|---|---|---|
| L1 | Haiku agrees with NIST | linear-weighted kappa at least 0.40 on the 10,298 pairs |
| L2 | Opus agrees better than Haiku | kappa difference on the 1,000-pair sample has a lower 95% bound above 0 |
| L3 | Haiku labels rank the systems as NIST labels do | Kendall's tau between the two orderings of the 11 systems at least 0.80 |
| L4 | Haiku labels favour the Haiku reranker | the gain of CE→Haiku-20 over CE is larger under Haiku labels than under NIST labels on the same pool, lower 95% bound of the difference above 0 |

## Part 3: online tests in simulation

- Users follow a cascade click model (Craswell et al., 2008) over the top 10 shown: at each rank, a passage of grade g is clicked with probability c_g; after a click the user stops with probability s_g, otherwise continues; the session ends after rank 10. Grades are NIST grades, unjudged passages grade 0. Three user types, with values set here in the spirit of the perfect, navigational and informational users of earlier interleaving simulations (Hofmann et al.):

| User | c_0, c_1, c_2, c_3 | s_0, s_1, s_2, s_3 |
|---|---|---|
| perfect | 0.00, 0.20, 0.60, 1.00 | 0, 0, 0, 0 |
| navigational | 0.05, 0.30, 0.70, 0.95 | 0.20, 0.30, 0.70, 0.90 |
| informational | 0.40, 0.60, 0.80, 0.90 | 0.10, 0.20, 0.30, 0.50 |

- Each session draws one of the 97 queries uniformly.
- A/B test: sessions split evenly between two rankers. Primary metric: session success, a session that ends with a click after which the user stops. Secondary: sessions with any click.
- Team-draft interleaving (Radlinski et al., 2008): the two rankings are merged into one top 10, each pick credited to its ranker; a session scores +1, -1 or 0 by which ranker's results got more clicks.
- Sessions needed for 80% power at two-sided alpha 0.05 come from means and variances estimated on 200,000 simulated sessions per arm (two-proportion formula for the A/B test, one-sample formula for the interleaving score). The formula is checked by running 1,000 simulated experiments at the computed size; power between 0.75 and 0.85 passes.
- Comparisons: the five of Part 1.

| | Prediction | Met if |
|---|---|---|
| O1 | Interleaving needs far fewer sessions | for every Part 1 comparison whose NDCG@10 interval excludes 0, interleaving needs at most one fifth of the A/B sessions (session success), for all three user types |
| O2 | Interleaving points the same way as NDCG | for the same comparisons, the interleaving preference has the sign of the NDCG@10 difference for perfect and navigational users |

## Checks on the pipeline

- K1: CE NDCG@10 on TREC DL 2019 within 0.02 of the 0.7430 published on its model card. If not, the pipeline is checked before any LLM request is sent.
- K2: unit tests for NDCG@10, MRR@10, Recall@100, BM25, the bootstrap, the randomization test, weighted kappa, the cascade model and team-draft interleaving.
- K3: a pilot on two unjudged 2019 queries checks the prompt, parsing and token counts and projects the full cost before the main batches.
- K4: API spending is capped at $30. If the pilot projects more, the Opus label sample shrinks from 1,000 to 500 pairs first.

## Not done

No training or fine-tuning, no dense retrieval over the full collection of 8.8 million passages, no listwise prompting, no real users. The click simulation inherits the NIST judgment pool and a uniform query mix, which real traffic does not have.

## Amendments

Entries below, each written before the results it affects. There are none.
