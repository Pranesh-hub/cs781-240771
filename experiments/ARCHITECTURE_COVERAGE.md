# Which of the four Phase 1 architectures each phase actually used

**Correction.** An earlier version of this table said BiLSTM was trained in Phases 4 to 7. The ranking notes show it was not: Phase 5 E4 BiLSTM never ran and Phase 6 has no BiLSTM run. The table below is corrected.

Checked from the notebook code (model classes that are built, trained or evaluated), the ranking ledgers and the review notes. "Loader only" means the notebook can load that model but the notebook states or shows it was not run.

| Phase | GPT-2 decoder | ModernBERT encoder | T5 encoder-decoder | BiLSTM | Where |
|---|---|---|---|---|---|
| 1 | trained | trained | trained | trained | `phase1_pretraining/notebooks/1..4_*` and `6_multi_architecture_pipelines/phase1_four_architectures` |
| 2 | trained (all runs) | **no notebook** | **no notebook** | all-phases BiLSTM notebook | `1_gpt2_decoder`, `4_bilstm` |
| 3 | trained (all runs) | **no notebook** | **no notebook** | all-phases BiLSTM notebook | `1_gpt2_decoder`, `4_bilstm` |
| 4 | trained | trained | trained (LoRA run broke, AUC about 0.50) | code only, no result seen | `6_multi_architecture_pipelines` (13 of 14 notebooks) |
| 5 | trained | trained (best single model, validation macro-F1 0.393) | trained (weak, 0.110) | code only: E4 BiLSTM never ran | `6_multi_architecture_pipelines` |
| 6 | trained (own decoder + cross-attention) | trained (encoder side) | trained (`part2_t5`) | code only: KD/BiLSTM notebook exists, no BiLSTM run (S7) | `1_gpt2_decoder`, `3_t5_encoder_decoder`, `4_bilstm`, `6_multi_architecture_pipelines` |
| 7 | loader only | trained (all six ledger runs) | loader only | notebook exists (`bilstm_retriever`), no ranked result seen | `2_modernbert_encoder`, `4_bilstm` |

The `+` lists in notebook names (`p4__gpt2+modernbert+t5+bilstm__…`) show the models whose code is in the notebook, not which of them gave a result; this table says which did.

Gaps that remain: Phases 2 and 3 have no ModernBERT or T5 notebook (their code registers only the GPT-2 family), Phase 7 did not run GPT-2 or T5, and BiLSTM has no ranked result in Phases 4 to 7. No such notebook exists on the laptop, in the context zips or on the three Kaggle accounts.

## Notebooks written for coverage (not run)

Where a phase had no notebook for an architecture, one was added so that every phase has code for it. They have no results and no leaderboard score. Each starts with a "NOT RUN" notice.

| Phase | Added notebook | What was checked |
|---|---|---|
| 2 | ModernBERT and T5 NER tagger + the same counsel/JUDGE rule layer | run end to end on CPU with a tiny random model and synthetic data; wrote a schema-valid `prediction.json` |
| 3 | ModernBERT and T5 rhetorical-role pipeline (the Phase 3 cells that ran on Kaggle, backbone swapped, full fine-tuning only) | backbone registration and classifier forward pass on CPU with a tiny random model; full pipeline not run |
| 7 | GPT-2 and T5 bi-encoder (the submitted Phase 7 pipeline, backbone swapped) | all five training techniques built, forward and backward pass on CPU with a tiny random model; full pipeline not run |

T5 trains in fp32 in the Phase 2 notebook because T5 can overflow in fp16; the Phase 3 and Phase 7 notebooks keep the original mixed-precision setting, which may need switching off for T5.

BiLSTM in Phases 4 to 7 already has notebook code (see the table above); it was not given new notebooks.
