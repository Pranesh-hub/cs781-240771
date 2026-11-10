# CS781: best 6 runs per phase (built 9 Nov 2026)

Scores are dev / validation numbers from saved outputs unless marked LEADERBOARD (a real test score). They are not comparable across phases. `runs_index.csv` has the same table.


## Phase 2: Legal NER (metric: strict macro-F1)

| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | prediction.zip (6a6752c271): champion + counsel + JUDGE rules | full + rules | LEADERBOARD strict macro-F1 0.2858 (dev ~0.2995) | local phase2_cs781 + scoring_result (1).zip. matched to score by timestamp (file 17:49, score 17:53) |
| 2 | prediction_7.zip (564bac316c) | full | LEADERBOARD 0.2854 | local phase2_cs781 + scoring_result.zip. matched by timestamp (file 06:47, score 06:48) |
| 3 | prediction_wit_hybrid_verified.json (4b5cfe6760) | full + WIT hybrid | not matched to a Codabench row | local phase2_cs781. see codabench_scores.csv |
| 4 | New folder (2)/prediction.json (f2d69fa79f) | full + rules | not matched to a Codabench row | local phase2_cs781. latest file (11 Oct 23:43) |
| 5 | prediction_6.zip (60999cc390) | full | not matched to a Codabench row | local phase2_cs781 |
| 6 | prediction_3.zip (a764bc8259) | full | not matched to a Codabench row | local phase2_cs781. same content as prediction_x.zip |

Codabench lists 14 submissions. The best (0.3232, 11 October 23:44) is a later refinement of the rule-based pipeline; the first rule version scored 0.2858. Six distinct experiments are not evidenced for Phase 2.


## Phase 3: Rhetorical Roles (metric: macro-F1 (CL / IT))

| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | E4_qlora | qlora | dev macro-F1 CL 0.290 / IT 0.339 (mean 0.315) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |
| 2 | E6_kd_student | kd | dev macro-F1 CL 0.284 / IT 0.328 (mean 0.306) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |
| 3 | E2_lorma_plus | lorma+ | dev macro-F1 CL 0.238 / IT 0.344 (mean 0.291) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |
| 4 | E1_lora | lora | dev macro-F1 CL 0.239 / IT 0.323 (mean 0.281) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger. scores from the context pack summary because the log parse was ambiguous |
| 5 | E0a_full_sent | full | dev macro-F1 CL 0.220 / IT 0.341 (mean 0.280) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |
| 6 | E3_ia3 | ia3 | dev macro-F1 CL 0.220 / IT 0.267 (mean 0.243) | Phase 3 notebook logs (notebook6d390d6f69 series) + recovery ledger |

Ensemble with smoothing and class biases scored CL 0.4205 / IT 0.4504, tuned on the same 5 dev docs (improved 2 of 5, worsened 3 on CL): uncertain. 7th run: E0b_full_ctx 0.2189 / 0.2929. Codabench: best 0.3922 (14 October 03:43), from six scored submissions.


## Phase 4: CJPE (metric: macro-F1)

| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | Ensemble (GPT-2 full + encoder full), fullRun.ipynb | ensemble | dev macro-F1 0.767, AUC 0.842 | context pack README; local CS781 phase4/Didnt submit/fullRun.ipynb. submitted as prediction_best: Codabench 0.7455 |
| 2 | P2_enc_full_W8 | full | dev AUC 0.8327; F1 rate-matched 0.7545 / tuned 0.7643 | Kaggle armankrsingh/notebook3393974db1 ledger.csv. = prediction_best_single.json = local Didnt submit prediction2/3 |
| 3 | P2_enc_full | full | dev AUC 0.8278; F1 rate-matched 0.7485 / tuned 0.7526 | same ledger. = local Didnt submit prediction4 |
| 4 | P1_gpt2_full | full | dev AUC 0.8091; F1 rate-matched 0.7243 / tuned 0.7298 | same ledger |
| 5 | P6_gpt2_ia3 | ia3 | dev AUC 0.7674 (F1 not recorded) | context pack summary. undertrained; checkpoint not on disk |
| 6 | G1_gpt2_tail512 (submitted 19 October) | full | dev macro-F1 0.6709; LEADERBOARD 0.6613 | Kaggle cs781-gpt2-pred-phase4; local CS781 phase4/prediction1.zip + scoring_result1.zip. Codabench scored the 19 October files at 0.6613 (G1) and 0.6585 (the second file) |

Codabench: the best file (prediction_best, 26 October) scored 0.7455; the encoder files scored 0.7350 and 0.7376; the early GPT-2 runs scored 0.6585 and 0.6613.


## Phase 5: Statute ID (metric: macro-F1)

| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | 8-member encoder mean ensemble (Part 3) | ensemble | val macro-F1 0.4128 (95%% CI 0.405-0.420) | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_*. TF-IDF baseline 0.376; E4 BiLSTM never ran |
| 2 | E6_enc_full | full | val macro-F1 0.3930 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |
| 3 | E2_lorma_plus | lorma+ | val macro-F1 0.3710 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |
| 4 | E5_distilled_student | kd | val macro-F1 0.3610 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |
| 5 | E1_gpt2_full | full | val macro-F1 0.3580 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |
| 6 | E3_t5 | full | val macro-F1 0.1100 | context pack 00b summary (measured on Kaggle); notebooks phase5_lsi_* |

E4 BiLSTM never ran. Part 4 improvements are built but their run was never seen. Codabench lists 18 scored files; the best is 0.4129 (prediction18, 27 October).


## Phase 6: Summarisation (metric: BERTScore F1)

| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | FINAL: S5h multi-chunk encoder + GPT-2 decoder, beam 4, byte-level decoder fix | houlsby | BERTScore F1 0.8499 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local Phase6/submissions/prediction10.zip = Kaggle p6-final-nb prediction.json (hash match) |
| 2 | S9_encdec_lorma | lorma+ | BERTScore F1 0.8213 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local prediction3.zip (hash match) |
| 3 | S8_gpt2_full | full | BERTScore F1 0.8175 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local prediction4.zip (hash match) |
| 4 | S6_gpt2_6L_kd | kd | BERTScore F1 0.8165 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local prediction5.zip; no-KD control 0.8150 |
| 5 | S3_gpt2_lora | lora | BERTScore F1 0.8151 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local prediction6.zip (hash match) |
| 6 | S4_gpt2_prompt | prompt | BERTScore F1 0.8133 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv |

Highest scores before the distinct-technique filter: S5h8 (8 chunks) 0.8485, S5c head-tail 0.8448, S5h 0.8346, S5i 0.8255. Rule gap: no BiLSTM run (S7).


## Phase 7: Prior Case Retrieval (metric: micro-F1@7)

| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | E2_enc_partial | partial | dev micro-F1@7 0.3502 | Kaggle oipranesh/p7-nb2 ledger.csv |
| 2 | E4_enc_ia3 | ia3 | dev micro-F1@7 0.3480 | Kaggle oipranesh/p7-nb2 ledger.csv |
| 3 | E1_enc_full | full | dev micro-F1@7 0.3457 | Kaggle oipranesh/p7-nb2 ledger.csv |
| 4 | E3_enc_lorma | lorma+ | dev micro-F1@7 0.3446 | Kaggle oipranesh/p7-nb2 ledger.csv. compulsory technique |
| 5 | E6_enc_half_kd | kd | dev micro-F1@7 0.3435 | Kaggle oipranesh/p7-nb2 ledger.csv. student has 64% of teacher params |
| 6 | E5_enc_prompt | prompt | dev micro-F1@7 0.3146 | Kaggle oipranesh/p7-nb2 ledger.csv |

Best system, not a single run: F_final_rule_safe 0.3702 = local Phase 7/prediction3.zip (hash match). H_final_hybrid 0.3969 (prediction4.zip) uses lexical BM25, which is not a Phase 1 model: ask the TAs first. Plain 6-model ensemble 0.3635.


## Phase 1

Not ranked: pretraining losses are not comparable across architectures. GPT-2 decoder val loss 3.244, ModernBERT-style encoder 1.744, T5 1,520 steps, BiLSTM 560 steps (loss 5.96).