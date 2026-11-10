# Phase 2: experiments

**Best leaderboard result:** strict macro-F1 0.3232 (rank 119). The notebook behind it is in the main repository folder for this phase.

This folder collects everything tried in this phase, including runs that failed or were superseded.

* `notebooks/`: 26 distinct experiment notebooks (stored outputs removed; duplicates removed by code). They are grouped by architecture in subfolders; see `notebooks/ARCHITECTURES.md`.
* `results/`: run ledgers, reports and per-run result files saved from Kaggle.
* `submissions/`: the prediction files uploaded to Codabench (and the scoring files where we saved them).

## Best runs of this phase


| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | prediction.zip (6a6752c271): champion + counsel + JUDGE rules | full + rules | LEADERBOARD strict macro-F1 0.2858 (dev ~0.2995) | local phase2_cs781 + scoring_result (1).zip. matched to score by timestamp (file 17:49, score 17:53) |
| 2 | prediction_7.zip (564bac316c) | full | LEADERBOARD 0.2854 | local phase2_cs781 + scoring_result.zip. matched by timestamp (file 06:47, score 06:48) |
| 3 | prediction_wit_hybrid_verified.json (4b5cfe6760) | full + WIT hybrid | not matched to a Codabench row | local phase2_cs781. see codabench_scores.csv |
| 4 | New folder (2)/prediction.json (f2d69fa79f) | full + rules | not matched to a Codabench row | local phase2_cs781. latest file (11 Oct 23:43) |
| 5 | prediction_6.zip (60999cc390) | full | not matched to a Codabench row | local phase2_cs781 |
| 6 | prediction_3.zip (a764bc8259) | full | not matched to a Codabench row | local phase2_cs781. same content as prediction_x.zip |

Codabench lists 14 submissions. The best (0.3232, 11 October 23:44) is a later refinement of the rule-based pipeline; the first rule version scored 0.2858. Six distinct experiments are not evidenced for Phase 2.

## Codabench submissions (strict macro-F1)

| submission id | date | score |
|---|---|---|
| 978743 | 2026-10-11 23:44 | 0.3232 |
| 977475 | 2026-10-11 17:51 | 0.2858 |
| 977469 | 2026-10-11 17:50 | 0.2858 |
| 976003 | 2026-10-11 06:47 | 0.2854 |
| 975971 | 2026-10-11 06:07 | 0.2149 |
| 975968 | 2026-10-11 06:02 | preparing |
| 975816 | 2026-10-11 04:14 | 0.2858 |
| 975770 | 2026-10-11 03:43 | cancelled |
| 975212 | 2026-10-10 22:41 | 0.1612 |
| 975208 | 2026-10-10 22:38 | 0.1552 |
| 975093 | 2026-10-10 21:36 | 0.1591 |
| 974835 | 2026-10-10 19:57 | 0.0879 |
| 974778 | 2026-10-10 19:21 | 0.0707 |

## Notebooks

| file | cells | size | first line |
|---|---|---|---|
| `p2__gpt2__experiments_clean.ipynb` | 14 | 36 KB | # CS781 Legal NER — Clean Experiment Notebook **Purpose:** one clean, reproducible notebook for checkpoint/data inventory, a locked exact-sp |
| `p2__gpt2__champion_plus_rules_run_cells.ipynb` | 52 | 149 KB | # Champion + counsel + JUDGE inference pipeline This notebook is extracted from the supplied CS781 notebook. It preserves the original imple |
| `p2__gpt2__new_self_contained_experiments.ipynb` | 13 | 72 KB | # CS781 Legal NER — Self-contained new experiments This notebook reconstructs the known champion + rules pipeline, checks the established va |
| `p2__gpt2__ner_clean_experiments__finalv1_c01.ipynb` | 52 | 167 KB | # CS781 Legal NER — Clean Experiment Notebook **Purpose:** one clean, reproducible notebook for checkpoint/data inventory, a locked exact-sp |
| `p2__gpt2__ner_clean_experiments__finalv1_c00.ipynb` | 48 | 143 KB | # CS781 Legal NER — Clean Experiment Notebook **Purpose:** one clean, reproducible notebook for checkpoint/data inventory, a locked exact-sp |
| `p2__gpt2__ner_clean_experiments_with_exp15__finalv1.ipynb` | 53 | 199 KB | # CS781 Legal NER — Clean Experiment Notebook **Purpose:** one clean, reproducible notebook for checkpoint/data inventory, a locked exact-sp |
| `p2__gpt2__ner_self_contained_experiments__finalv2_c01.ipynb` | 35 | 105 KB | # CS781 Legal NER — Self-contained new experiments This notebook reconstructs the known champion + rules pipeline, checks the established va |
| `p2__gpt2__ner_self_contained_experiments__finalv2_c02.ipynb` | 47 | 182 KB | # CS781 Legal NER — Self-contained new experiments This notebook reconstructs the known champion + rules pipeline, checks the established va |
| `00_BEST__p2__gpt2__ner_self_contained_experiments__finalv2_c03.ipynb` | 50 | 200 KB | # CS781 Legal NER — Self-contained new experiments This notebook reconstructs the known champion + rules pipeline, checks the established va |
| `p2__gpt2__ner_self_contained_experiments__finalv2_c04.ipynb` | 32 | 100 KB | # CS781 Legal NER — Self-contained new experiments This notebook reconstructs the known champion + rules pipeline, checks the established va |
| `p2__gpt2__ner_self_contained_experiments__finalv2_c05.ipynb` | 108 | 339 KB | # CS781 Legal NER — Self-contained new experiments This notebook reconstructs the known champion + rules pipeline, checks the established va |
| `p2__gpt2__ner_self_contained_experiments__finalv2_c00.ipynb` | 30 | 90 KB | # CS781 Legal NER — Self-contained new experiments This notebook reconstructs the known champion + rules pipeline, checks the established va |
| `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c10.ipynb` | 385 | 709 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c04.ipynb` | 294 | 482 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c05.ipynb` | 50 | 94 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c06.ipynb` | 222 | 364 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c07.ipynb` | 321 | 537 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c08.ipynb` | 356 | 614 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c09.ipynb` | 380 | 690 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c01.ipynb` | 80 | 184 KB | # Champion + counsel + JUDGE inference pipeline This notebook is extracted from the supplied CS781 notebook. It preserves the original imple |
| `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c02.ipynb` | 88 | 212 KB | # Champion + counsel + JUDGE inference pipeline This notebook is extracted from the supplied CS781 notebook. It preserves the original imple |
| `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c03.ipynb` | 97 | 239 KB | # Champion + counsel + JUDGE inference pipeline This notebook is extracted from the supplied CS781 notebook. It preserves the original imple |
| `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c04.ipynb` | 162 | 417 KB | # Champion + counsel + JUDGE inference pipeline This notebook is extracted from the supplied CS781 notebook. It preserves the original imple |
| `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c05.ipynb` | 164 | 425 KB | # Champion + counsel + JUDGE inference pipeline This notebook is extracted from the supplied CS781 notebook. It preserves the original imple |
| `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c06.ipynb` | 170 | 436 KB | # Champion + counsel + JUDGE inference pipeline This notebook is extracted from the supplied CS781 notebook. It preserves the original imple |
| `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c00.ipynb` | 52 | 126 KB | # Champion + counsel + JUDGE inference pipeline This notebook is extracted from the supplied CS781 notebook. It preserves the original imple |
