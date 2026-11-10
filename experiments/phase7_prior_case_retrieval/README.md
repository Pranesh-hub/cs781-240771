# Phase 7: experiments

**Best leaderboard result:** micro-F1@7 0.3916 (rank 65). The notebook behind it is in the main repository folder for this phase.

This folder collects everything tried in this phase, including runs that failed or were superseded.

* `notebooks/`: 8 distinct experiment notebooks (stored outputs removed; duplicates removed by code). They are grouped by architecture in subfolders; see `notebooks/ARCHITECTURES.md`.
* `results/`: run ledgers, reports and per-run result files saved from Kaggle.
* `submissions/`: the prediction files uploaded to Codabench (and the scoring files where we saved them).

## Best runs of this phase


| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | E2_enc_partial | partial | dev micro-F1@7 0.3502 | Kaggle oipranesh/p7-nb2 ledger.csv |
| 2 | E4_enc_ia3 | ia3 | dev micro-F1@7 0.3480 | Kaggle oipranesh/p7-nb2 ledger.csv |
| 3 | E1_enc_full | full | dev micro-F1@7 0.3457 | Kaggle oipranesh/p7-nb2 ledger.csv |
| 4 | E3_enc_lorma | lorma+ | dev micro-F1@7 0.3446 | Kaggle oipranesh/p7-nb2 ledger.csv. compulsory technique |
| 5 | E6_enc_half_kd | kd | dev micro-F1@7 0.3435 | Kaggle oipranesh/p7-nb2 ledger.csv. student has 64% of teacher params |
| 6 | E5_enc_prompt | prompt | dev micro-F1@7 0.3146 | Kaggle oipranesh/p7-nb2 ledger.csv |

Best system, not a single run: F_final_rule_safe 0.3702 = local Phase 7/prediction3.zip (hash match). H_final_hybrid 0.3969 (prediction4.zip) uses lexical BM25, which is not a Phase 1 model: ask the TAs first. Plain 6-model ensemble 0.3635.

## Codabench submissions (micro-F1@7)

| submission id | date | score |
|---|---|---|
| 1042845 | 2026-11-08 23:53 | 0.3461 |
| 1042765 | 2026-11-08 23:13 | 0.3916 |
| 1042764 | 2026-11-08 23:13 | 0.3467 |
| 1042760 | 2026-11-08 23:13 | 0.3396 |
| 1042759 | 2026-11-08 23:13 | 0.3284 |

## Notebooks

| file | cells | size | first line |
|---|---|---|---|
| `p7__modernbert__pcr_six_experiments_modernbert_biencoder__16525fdb51.ipynb` | 21 | 86 KB | # CS781 Phase 7 — PRIOR CASE RETRIEVAL: one notebook, Phase 1 ModernBERT bi-encoder, six experiments, prediction.json (micro-F1@7) **Deadlin |
| `00_BEST__p7__bilstm__bilstm_retriever.ipynb` | 22 | 100 KB | # CS781 Phase 7 — BiLSTM RETRIEVER: implementation, a new Kaggle dataset that stores the model, and the predictions **What this notebook is. |
| `p7__modernbert__fast_predict.ipynb` | 23 | 99 KB | # CS781 Phase 7 — FAST PREDICTIONS from the saved checkpoints (about 20-35 minutes; nothing is trained) For when time is short. It uses the  |
| `00_BEST__p7__modernbert__nb2.ipynb` | 27 | 104 KB | # CS781 Phase 7 — PART 2: pool-aware re-ranking, n-gram BM25, ensembles, more chunks, round-2 hard negatives, citation context, cross-encode |
| `p7__modernbert__more_predictions.ipynb` | 10 | 33 KB | # CS781 Phase 7 — MORE PREDICTION FILES in about 2 minutes (CPU only, nothing is trained, no GPU needed) The fast notebook wrote one file (t |
| `p7__modernbert__part2_improvements.ipynb` | 26 | 113 KB | # CS781 Phase 7 — PART 2: pool-aware re-ranking, n-gram BM25, ensembles, more chunks, round-2 hard negatives, citation context, cross-encode |
| `p7__modernbert__part2_improvements_pack.ipynb` | 26 | 113 KB | # CS781 Phase 7 — PART 2: pool-aware re-ranking, n-gram BM25, ensembles, more chunks, round-2 hard negatives, citation context, cross-encode |
| `p7__modernbert__pcr_retrieval.ipynb` | 20 | 82 KB | # CS781 Phase 7 — PRIOR CASE RETRIEVAL: one notebook, Phase 1 ModernBERT bi-encoder, six experiments, prediction.json (micro-F1@7) **Deadlin |
| `p7__modernbert__pcr_retrieval_pack.ipynb` | 20 | 82 KB | # CS781 Phase 7 — PRIOR CASE RETRIEVAL: one notebook, Phase 1 ModernBERT bi-encoder, six experiments, prediction.json (micro-F1@7) **Deadlin |
