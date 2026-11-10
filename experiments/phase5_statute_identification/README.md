# Phase 5: experiments

**Best leaderboard result:** macro-F1 0.4129 (rank 27). The notebook behind it is in the main repository folder for this phase.

This folder collects everything tried in this phase, including runs that failed or were superseded.

* `notebooks/`: 8 distinct experiment notebooks (stored outputs removed; duplicates removed by code). They are grouped by architecture in subfolders; see `notebooks/ARCHITECTURES.md`.
* `results/`: run ledgers, reports and per-run result files saved from Kaggle.
* `submissions/`: the prediction files uploaded to Codabench (and the scoring files where we saved them).

## Best runs of this phase


| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | 8-member encoder mean ensemble (Part 3) | ensemble | val macro-F1 0.4128 (95%% CI 0.405-0.420) | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_*. TF-IDF baseline 0.376; E4 BiLSTM never ran |
| 2 | E6_enc_full | full | val macro-F1 0.3930 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |
| 3 | E2_lorma_plus | lorma+ | val macro-F1 0.3710 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |
| 4 | E5_distilled_student | kd | val macro-F1 0.3610 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |
| 5 | E1_gpt2_full | full | val macro-F1 0.3580 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |
| 6 | E3_t5 | full | val macro-F1 0.1100 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |

E4 BiLSTM never ran. Part 4 improvements are built but their run was never seen. Codabench lists 18 scored files; the best is 0.4129 (prediction18, 27 October).

## Codabench submissions (macro-F1)

| submission id | date | score |
|---|---|---|
| 1011268 | 2026-10-27 06:27 | 0.4129 |
| 1011267 | 2026-10-27 06:26 | 0.4108 |
| 1011266 | 2026-10-27 06:26 | 0.4068 |
| 1011264 | 2026-10-27 06:26 | 0.4129 |
| 1010475 | 2026-10-27 00:24 | 0.4049 |
| 1010474 | 2026-10-27 00:24 | 0.4048 |
| 1010472 | 2026-10-27 00:24 | 0.3944 |
| 1010471 | 2026-10-27 00:24 | 0.3974 |
| 1010469 | 2026-10-27 00:24 | 0.4049 |
| 1010468 | 2026-10-27 00:24 | 0.3866 |
| 1010466 | 2026-10-27 00:23 | 0.3407 |
| 1010463 | 2026-10-27 00:23 | 0.3531 |
| 1010461 | 2026-10-27 00:23 | 0.1066 |
| 1010460 | 2026-10-27 00:23 | 0.3679 |
| 1010453 | 2026-10-27 00:22 | 0.3526 |
| 1010452 | 2026-10-27 00:22 | 0.3717 |
| 1010451 | 2026-10-27 00:22 | 0.3875 |
| 1010450 | 2026-10-27 00:22 | 0.3875 |

## Notebooks

| file | cells | size | first line |
|---|---|---|---|
| `p5__gpt2+modernbert+t5+bilstm__lsi_part2_experiments_e4_e5__cs781_phase5.ipynb` | 19 | 106 KB | # CS781 Phase 5 (statute identification) — PART 2 of 2: restore Part 1, experiments E4 and E5, ensemble, rule check, final prediction.json * |
| `p5__gpt2+modernbert+t5+bilstm__lsi_part3_better_predictions__cs781_phase5.ipynb` | 23 | 108 KB | # CS781 Phase 5 — PART 3: better predictions from the finished experiments (stacker, more seeds, longer training, rare-statute losses, 8 win |
| `p5__gpt2+modernbert+t5+bilstm__lsi_part1_baselines_window_test__cs781_phase5.ipynb` | 22 | 110 KB | # CS781 Phase 5 (statute identification) — PART 1 of 2: baselines, window test, experiments E1, E6, E2, E3, backup **Why two parts:** the si |
| `p5__gpt2+modernbert+t5+bilstm__lsi_all_experiments.ipynb` | 28 | 124 KB | # CS781 Phase 5 — Legal Statute Identification (LSI): multi-label prediction of statute ids, with every capstone requirement built in **Data |
| `p5__gpt2+modernbert+t5+bilstm__lsi_all_experiments_alt.ipynb` | 28 | 126 KB | # CS781 Phase 5 — Legal Statute Identification (LSI): multi-label prediction of statute ids, with every capstone requirement built in **Data |
| `p5__gpt2+modernbert+t5+bilstm__lsi_part1_baselines_e1_e6_e2_e3_alt.ipynb` | 22 | 110 KB | # CS781 Phase 5 (statute identification) — PART 1 of 2: baselines, window test, experiments E1, E6, E2, E3, backup **Why two parts:** the si |
| `p5__gpt2+modernbert+t5+bilstm__lsi_part2_e4_e5_final.ipynb` | 19 | 103 KB | # CS781 Phase 5 (statute identification) — PART 2 of 2: restore Part 1, experiments E4 and E5, ensemble, rule check, final prediction.json * |
| `p5__gpt2+modernbert+t5+bilstm__lsi_part4_rules_safe_improvements.ipynb` | 22 | 109 KB | # CS781 Phase 5 — PART 4: rules-safe improvements (label-wise attention, multi-window end-to-end training, repeat-factor sampling, layer-wis |
