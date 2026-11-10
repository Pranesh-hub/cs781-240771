# Phase 6: experiments

**Best leaderboard result:** BERTScore F1 0.8528 (rank 12). The notebook behind it is in the main repository folder for this phase.

This folder collects everything tried in this phase, including runs that failed or were superseded.

* `notebooks/`: 17 distinct experiment notebooks (stored outputs removed; duplicates removed by code). They are grouped by architecture in subfolders; see `notebooks/ARCHITECTURES.md`.
* `results/`: run ledgers, reports and per-run result files saved from Kaggle.
* `submissions/`: the prediction files uploaded to Codabench (and the scoring files where we saved them).

## Best runs of this phase


| # | run | technique | score | source / note |
|---|---|---|---|---|
| 1 | FINAL: S5h multi-chunk encoder + GPT-2 decoder, beam 4, byte-level decoder fix | houlsby | BERTScore F1 0.8499 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local Phase6/submissions/prediction10.zip = Kaggle p6-final-nb prediction.json (hash match) |
| 2 | S9_encdec_lorma | lorma+ | BERTScore F1 0.8213 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local prediction3.zip (hash match) |
| 3 | S8_gpt2_full | full | BERTScore F1 0.8175 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local prediction4.zip (hash match) |
| 4 | S6_gpt2_6L_kd | kd | BERTScore F1 0.8165 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local prediction5.zip; no-KD control 0.8150 |
| 5 | S3_gpt2_lora | lora | BERTScore F1 0.8151 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv. local prediction6.zip (hash match) |
| 6 | S4_gpt2_prompt | prompt | BERTScore F1 0.8133 | Kaggle oipranesh/p6-final-nb ledger.csv + decode_fix_comparison.csv |

Highest scores before the distinct-technique filter: S5h8 (8 chunks) 0.8485, S5c head-tail 0.8448, S5h 0.8346, S5i 0.8255. Rule gap: no BiLSTM run (S7).

## Codabench submissions (BERTScore F1)

| submission id | date | score |
|---|---|---|
| 1030777 | 2026-11-03 22:07 | 0.8528 |
| 1029824 | 2026-11-03 14:59 | 0.8501 |
| 1029649 | 2026-11-03 13:33 | 0.8397 |
| 1029021 | 2026-11-03 06:39 | 0.7791 |
| 1029020 | 2026-11-03 06:39 | 0.8135 |
| 1029019 | 2026-11-03 06:39 | 0.8158 |
| 1029018 | 2026-11-03 06:38 | 0.8162 |
| 1029017 | 2026-11-03 06:38 | 0.8183 |
| 1029016 | 2026-11-03 06:38 | 0.8233 |
| 1028432 | 2026-11-02 23:23 | 0.8186 |
| 1028225 | 2026-11-02 21:49 | 0.8186 |

## Notebooks

| file | cells | size | first line |
|---|---|---|---|
| `p6__gpt2+modernbert+t5+bilstm__followup_decoding_fix.ipynb` | 20 | 147 KB | # CS781 Phase 6 — FOLLOW-UP: fix the decoding of the generated text and regenerate the final prediction.json **Why this notebook exists.** P |
| `p6__gpt2+modernbert+t5+bilstm__part0_selftest.ipynb` | 28 | 207 KB | # CS781 Phase 6 — PART 0: SELF-TEST on tiny synthetic models (run this FIRST; Kaggle CPU, no GPU quota) Runs **the same code as Parts 1-5**  |
| `p6__gpt2+modernbert+t5+bilstm__part0_selftest_2.ipynb` | 28 | 207 KB | # CS781 Phase 6 — PART 0: SELF-TEST on tiny synthetic models (run this FIRST; Kaggle CPU, no GPU quota) Runs **the same code as Parts 1-5**  |
| `p6__gpt2+modernbert+t5+bilstm__part1_data_longdoc.ipynb` | 17 | 144 KB | # CS781 Phase 6 (abstractive legal summarisation) — PART 1: data, schema, validation split, compatibility checks, baselines, long-document s |
| `p6__gpt2+modernbert+t5+bilstm__part1_data_longdoc_2.ipynb` | 17 | 144 KB | # CS781 Phase 6 (abstractive legal summarisation) — PART 1: data, schema, validation split, compatibility checks, baselines, long-document s |
| `00_BEST__p6__t5__part2_t5_s1_s2.ipynb` | 15 | 127 KB | # CS781 Phase 6 — PART 2: experiments S1 (T5 full fine-tuning) and S2 (T5 LoRMA+, compulsory) **Attach:** Part 1's notebook output (Add Inpu |
| `p6__t5__part2_t5_s1_s2_2.ipynb` | 15 | 127 KB | # CS781 Phase 6 — PART 2: experiments S1 (T5 full fine-tuning) and S2 (T5 LoRMA+, compulsory) **Attach:** Part 1's notebook output (Add Inpu |
| `00_BEST__p6__gpt2+modernbert__part3_gpt2_encdec_s3_s4_s5_s8_s9.ipynb` | 15 | 127 KB | # CS781 Phase 6 — PART 3: GPT-2 experiments (S8 full, S3 LoRA, S4 prompt tuning) and the ModernBERT encoder + GPT-2 decoder (S5 Houlsby adap |
| `p6__gpt2+modernbert__part3_gpt2_encdec_s3_s4_s5_s8_s9_2.ipynb` | 15 | 127 KB | # CS781 Phase 6 — PART 3: GPT-2 experiments (S8 full, S3 LoRA, S4 prompt tuning) and the ModernBERT encoder + GPT-2 decoder (S5 Houlsby adap |
| `00_BEST__p6__gpt2+bilstm__part4_kd_bilstm_s6_s7.ipynb` | 15 | 130 KB | # CS781 Phase 6 — PART 4: S6 knowledge distillation (mandatory) and S7 the BiLSTM sequence-to-sequence model **Attach:** Part 2's AND/OR Par |
| `p6__gpt2+bilstm__part4_kd_bilstm_s6_s7_2.ipynb` | 15 | 130 KB | # CS781 Phase 6 — PART 4: S6 knowledge distillation (mandatory) and S7 the BiLSTM sequence-to-sequence model **Attach:** Part 2's AND/OR Par |
| `p6__gpt2+modernbert+t5+bilstm__part5_final.ipynb` | 19 | 153 KB | # CS781 Phase 6 — PART 5 (FINAL): comparison, rule check, best checkpoint, generation-parameter study, test inference, `prediction.json`, qu |
| `p6__gpt2+modernbert+t5+bilstm__part5_final_2.ipynb` | 19 | 153 KB | # CS781 Phase 6 — PART 5 (FINAL): comparison, rule check, best checkpoint, generation-parameter study, test inference, `prediction.json`, qu |
| `p6__gpt2+modernbert+t5+bilstm__part6_improvements.ipynb` | 26 | 179 KB | # CS781 Phase 6 — PART 6: improvement notebook (BiLSTM experiment, learned detokeniser, multi-chunk encoder, better input strategy, better d |
| `00_BEST__p6__gpt2+modernbert+t5+bilstm__part7_combined_under7h.ipynb` | 23 | 156 KB | # CS781 Phase 6 — PART 7 (combined, under 7 hours): quick wins + retraining of the best system One notebook instead of 7a + 7b, with a **tim |
| `p6__gpt2+modernbert+t5+bilstm__part7a_quick_wins.ipynb` | 22 | 149 KB | # CS781 Phase 6 — PART 7a: inference-only quick wins on the finished best system (chunk count at inference, length penalty, trimming) No tra |
| `p6__gpt2+modernbert+t5+bilstm__part7b_retraining.ipynb` | 20 | 149 KB | # CS781 Phase 6 — PART 7b: retraining variants (8 chunks, compressed targets, LoRMA on the multi-chunk encoder, optional full fine-tuning an |
