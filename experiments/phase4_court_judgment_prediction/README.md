# Phase 4: experiments

**Best leaderboard result:** macro-F1 0.7455 (rank 51). The notebook behind it is in the main repository folder for this phase.

This folder collects everything tried in this phase, including runs that failed or were superseded.

* `notebooks/`: 14 distinct experiment notebooks (stored outputs removed; duplicates removed by code). They are grouped by architecture in subfolders; see `notebooks/ARCHITECTURES.md`.
* `results/`: run ledgers, reports and per-run result files saved from Kaggle.
* `submissions/`: the prediction files uploaded to Codabench (and the scoring files where we saved them).

## Best runs of this phase


| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | Ensemble (GPT-2 full + encoder full), fullRun.ipynb | ensemble | dev macro-F1 0.767, AUC 0.842 | context pack README; local CS781 phase4/Didnt submit/fullRun.ipynb. submitted as prediction_best: Codabench 0.7455 |
| 2 | P2_enc_full_W8 | full | dev AUC 0.8327; F1 rate-matched 0.7545 / tuned 0.7643 | Kaggle armankrsingh/notebook3393974db1 ledger.csv. = prediction_best_single.json = local Didnt submit prediction2/3 |
| 3 | P2_enc_full | full | dev AUC 0.8278; F1 rate-matched 0.7485 / tuned 0.7526 | same ledger. = local Didnt submit prediction4 |
| 4 | P1_gpt2_full | full | dev AUC 0.8091; F1 rate-matched 0.7243 / tuned 0.7298 | same ledger |
| 5 | P6_gpt2_ia3 | ia3 | dev AUC 0.7674 (F1 not recorded) | context pack summary. undertrained; checkpoint not on disk |
| 6 | G1_gpt2_tail512 (submitted 19 October) | full | dev macro-F1 0.6709; LEADERBOARD 0.6613 | Kaggle cs781-gpt2-pred-phase4; local CS781 phase4/prediction1.zip + scoring_result1.zip. Codabench scored the 19 October files at 0.6613 (G1) and 0.6585 (the second file) |

Codabench: the best file (prediction_best, 26 October) scored 0.7455; the encoder files scored 0.7350 and 0.7376; the early GPT-2 runs scored 0.6585 and 0.6613.

## Codabench submissions (macro-F1)

| submission id | date | score |
|---|---|---|
| 1008896 | 2026-10-26 16:32 | 0.7455 |
| 1008875 | 2026-10-26 16:26 | 0.7376 |
| 1008871 | 2026-10-26 16:25 | 0.735 |
| 1008863 | 2026-10-26 16:23 | cancelled |
| 1008854 | 2026-10-26 16:22 | 0.7376 |
| 1003369 | 2026-10-19 21:49 | 0.6585 |
| 1003367 | 2026-10-19 21:47 | 0.6613 |

## Notebooks

| file | cells | size | first line |
|---|---|---|---|
| `00_BEST__p4__gpt2__gpt2-pred-phase4.ipynb` | 11 | 33 KB | # CS781 Phase 4 (CJPE) — predictions with the NEW GPT-2 model (decoder-only Phase 1 model) **Task:** predict each Indian Supreme Court case' |
| `00_BEST__p4__gpt2+modernbert+t5+bilstm__didnt_submit_fullrun.ipynb` | 15 | 77 KB | # CS781 Phase 4 (CJPE) — BEST RUNS ONLY: GPT-2, ModernBERT (encoder), T5 Extracted from the v3 notebook: the shared code is the same cells ( |
| `p4__gpt2__final-arch.ipynb` | 21 | 43 KB | # ============================================================ # CS781 PHASE 1 — RETNET ARCHITECTURE IMPLEMENTATION # ====================== |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_v3_required_techniques__3393974db1_c01.ipynb` | 21 | 83 KB | # CS781 Phase 4 (CJPE) — v3: the REQUIRED techniques (full, partial, LoRA, LoRMA, IA3, knowledge distillation) across all four Phase 1 archi |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_v3_required_techniques__3393974db1_c00.ipynb` | 19 | 72 KB | # CS781 Phase 4 (CJPE) — v3: the REQUIRED techniques (full, partial, LoRA, LoRMA, IA3, knowledge distillation) across all four Phase 1 archi |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_best_runs.ipynb` | 11 | 66 KB | # CS781 Phase 4 (CJPE) — BEST RUNS ONLY: GPT-2, ModernBERT (encoder), T5 Extracted from the v3 notebook: the shared code is byte-for-byte th |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_best_runs1.ipynb` | 13 | 70 KB | # CS781 Phase 4 (CJPE) — BEST RUNS ONLY: GPT-2, ModernBERT (encoder), T5 Extracted from the v3 notebook: the shared code is byte-for-byte th |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_best_runs2.ipynb` | 14 | 76 KB | # CS781 Phase 4 (CJPE) — BEST RUNS ONLY: GPT-2, ModernBERT (encoder), T5 Extracted from the v3 notebook: the shared code is the same cells ( |
| `p4__gpt2__cjpe_gpt2_new.ipynb` | 10 | 32 KB | # CS781 Phase 4 (CJPE) — predictions with the NEW GPT-2 model (decoder-only Phase 1 model) **Task:** predict each Indian Supreme Court case' |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_improve_max5h.ipynb` | 22 | 88 KB | # CS781 Phase 4 (CJPE) — IMPROVEMENT NOTEBOOK: a window-aware stacker, more ModernBERT members, the missing LoRMA and distillation experimen |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_predictions.ipynb` | 13 | 34 KB | # CS781 Phase 4 — Court Judgment Prediction (CJPE): fine-tune the Phase 1 models and generate the six submission files **Task:** given the t |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_v2_all_models.ipynb` | 17 | 51 KB | # CS781 Phase 4 (CJPE) — v2: fine-tune ALL Phase 1 models, ensemble them, and predict (built from the weaknesses seen in the first GPT-2 run |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_v3_techniques_pack.ipynb` | 19 | 79 KB | # CS781 Phase 4 (CJPE) — v3: the REQUIRED techniques (full, partial, LoRA, LoRMA, IA3, knowledge distillation) across all four Phase 1 archi |
| `p4__gpt2+modernbert+t5+bilstm__cjpe_v4_final.ipynb` | 21 | 100 KB | # CS781 Phase 4 (CJPE: court judgment prediction) — the full notebook: restore, the remaining required experiments, long-document study, ens |
