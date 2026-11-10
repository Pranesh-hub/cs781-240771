# Phase 3: experiments

**Best leaderboard result:** macro-F1 0.3922 (rank 127). The notebook behind it is in the main repository folder for this phase.

This folder collects everything tried in this phase, including runs that failed or were superseded.

* `notebooks/`: 32 distinct experiment notebooks (stored outputs removed; duplicates removed by code). They are grouped by architecture in subfolders; see `notebooks/ARCHITECTURES.md`.
* `results/`: run ledgers, reports and per-run result files saved from Kaggle.
* `submissions/`: the prediction files uploaded to Codabench (and the scoring files where we saved them).

## Best runs of this phase


| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | E4_qlora | qlora | dev macro-F1 CL 0.290 / IT 0.339 (mean 0.315) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |
| 2 | E6_kd_student | kd | dev macro-F1 CL 0.284 / IT 0.328 (mean 0.306) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |
| 3 | E2_lorma_plus | lorma+ | dev macro-F1 CL 0.238 / IT 0.344 (mean 0.291) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |
| 4 | E1_lora | lora | dev macro-F1 CL 0.239 / IT 0.323 (mean 0.281) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger. scores from the context pack summary because the log parse was ambiguous |
| 5 | E0a_full_sent | full | dev macro-F1 CL 0.220 / IT 0.341 (mean 0.280) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |
| 6 | E3_ia3 | ia3 | dev macro-F1 CL 0.220 / IT 0.267 (mean 0.243) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |

Ensemble with smoothing and class biases scored CL 0.4205 / IT 0.4504, tuned on the same 5 dev docs (improved 2 of 5, worsened 3 on CL): uncertain. 7th run: E0b_full_ctx 0.2189 / 0.2929. Codabench: best 0.3922 (14 October 03:43), from six scored submissions.

## Codabench submissions (macro-F1)

| submission id | date | score |
|---|---|---|
| 985950 | 2026-10-14 04:13 | 0.3859 |
| 985940 | 2026-10-14 03:59 | 0.3828 |
| 985929 | 2026-10-14 03:43 | 0.3922 |
| 985910 | 2026-10-14 03:13 | 0.3748 |
| 985905 | 2026-10-14 03:06 | failed |
| 985896 | 2026-10-14 03:03 | 0.2966 |
| 981536 | 2026-10-13 07:07 | 0.2765 |

## Notebooks

| file | cells | size | first line |
|---|---|---|---|
| `p3__gpt2__phase3_audited_fixed.ipynb` | 19 | 67 KB | # CS781 Phase 3 — Rhetorical Role Prediction (CL + IT): training pipeline **What this notebook does:** loads the Phase 3 data, builds senten |
| `p3__gpt2__phase3_full_fixed_3000step.ipynb` | 20 | 65 KB | # CS781 Phase 3 — Rhetorical Role Prediction (CL + IT): training pipeline **What this notebook does:** loads the Phase 3 data, builds senten |
| `p3__gpt2__phase3_periodic_kaggle_publishing.ipynb` | 22 | 89 KB | ## Durable recovery publishing This version periodically publishes `latest.pt`, `best.pt`, experiment `result.json`, the ledger, and predict |
| `p3__gpt2__phase3_periodic_publishing_fixed.ipynb` | 23 | 82 KB | ## Durable recovery publishing This version periodically publishes `latest.pt`, `best.pt`, experiment `result.json`, the ledger, and predict |
| `p3__gpt2__phase3_rr_colab_t4.ipynb` | 38 | 47 KB | # CS781 Phase 3 — Rhetorical Role Prediction (Colab T4 starter pipeline) **Runtime target:** Google Colab with an NVIDIA T4 GPU (16 GB VRAM, |
| `p3__gpt2__phase3_rr_colab_tpu.ipynb` | 38 | 52 KB | # CS781 Phase 3 — Rhetorical Role Prediction (Colab TPU starter pipeline) **Runtime target:** Google Colab TPU runtime, where available. **I |
| `p3__gpt2__phase3_rr_starter_pipeline.ipynb` | 37 | 46 KB | # CS781 Phase 3 — Rhetorical Role Prediction (starter training pipeline) **Purpose:** train and compare sentence-level rhetorical-role class |
| `p3__gpt2__phase3_rr_stopiteration_fixed.ipynb` | 19 | 57 KB | # CS781 Phase 3 — Rhetorical Role Prediction (CL + IT): training pipeline **What this notebook does:** loads the Phase 3 data, builds senten |
| `p3__gpt2__phase3_resumable_checkpoints.ipynb` | 20 | 72 KB | # CS781 Phase 3 — Rhetorical Role Prediction (CL + IT): training pipeline **What this notebook does:** loads the Phase 3 data, builds senten |
| `p3__gpt2__diskfull_publisher_fix.ipynb` | 24 | 91 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__incremental_sharded_recovery.ipynb` | 24 | 107 KB | ## Incremental sharded recovery + rolling checkpoints `latest.pt` is atomically replaced after each completed epoch; `best.pt` is updated on |
| `p3__gpt2__notebook_kd_logits_recovery_fix.ipynb` | 24 | 93 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rolling_checkpoints_fixed.ipynb` | 24 | 93 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c10.ipynb` | 22 | 78 KB | ## Durable recovery publishing This version periodically publishes `latest.pt`, `best.pt`, experiment `result.json`, the ledger, and predict |
| `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c12.ipynb` | 23 | 79 KB | ## Durable recovery publishing This version periodically publishes `latest.pt`, `best.pt`, experiment `result.json`, the ledger, and predict |
| `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c13.ipynb` | 24 | 84 KB | ## Durable recovery publishing This version periodically publishes `latest.pt`, `best.pt`, experiment `result.json`, the ledger, and predict |
| `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c14.ipynb` | 24 | 84 KB | ## Durable recovery publishing This version periodically publishes `latest.pt`, `best.pt`, experiment `result.json`, the ledger, and predict |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c18.ipynb` | 25 | 94 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c21.ipynb` | 25 | 96 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c22.ipynb` | 27 | 108 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c23.ipynb` | 76 | 215 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c24.ipynb` | 77 | 217 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c25.ipynb` | 97 | 290 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c27.ipynb` | 105 | 334 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c28.ipynb` | 107 | 337 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c30.ipynb` | 108 | 343 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_training_pipeline__6d390d6f69_c04.ipynb` | 19 | 53 KB | # CS781 Phase 3 — Rhetorical Role Prediction (CL + IT): training pipeline **What this notebook does:** loads the Phase 3 data, builds senten |
| `p3__gpt2__rr_training_pipeline__6d390d6f69_c05.ipynb` | 19 | 55 KB | # CS781 Phase 3 — Rhetorical Role Prediction (CL + IT): training pipeline **What this notebook does:** loads the Phase 3 data, builds senten |
| `p3__gpt2__rr_training_pipeline__6d390d6f69_c06.ipynb` | 20 | 62 KB | # CS781 Phase 3 — Rhetorical Role Prediction (CL + IT): training pipeline **What this notebook does:** loads the Phase 3 data, builds senten |
| `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c08.ipynb` | 23 | 78 KB | ## Durable recovery publishing This version periodically publishes `latest.pt`, `best.pt`, experiment `result.json`, the ledger, and predict |
| `00_BEST__p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c00.ipynb` | 102 | 318 KB | ## Fixed rolling checkpoints + controlled durable publishing This notebook uses a **fixed local checkpoint set** per run: `latest.pt` is ato |
| `p3__gpt2__rr_pipeline.ipynb` | 18 | 65 KB | # CS781 Phase 3 — Rhetorical Role Prediction (CL + IT): training pipeline **What this notebook does:** loads the Phase 3 data, builds senten |
