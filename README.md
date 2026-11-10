# CS781 capstone: in-house legal intelligence system

Self-hosted legal language models for seven tasks on Indian court documents: pretraining, named entity recognition, rhetorical roles, judgment prediction, statute identification, summarization and prior case retrieval. Every later phase starts from our own pretrained checkpoints (no outside weights).

## Results (Codabench, best valid submission per phase)

| Phase | Task | Best result | Folder |
|---|---|---|---|
| 1 | Legal language-model pretraining | BERTScore F1 0.7973 (ROUGE-L 0.1782, BLEU-4 0.0112), rank 140 | [`phase1_pretraining/`](phase1_pretraining/) |
| 2 | Legal named entity recognition | strict macro-F1 0.3232 (precision 0.3954, recall 0.2902), rank 119 | [`phase2_legal_ner/`](phase2_legal_ner/) |
| 3 | Rhetorical role prediction | macro-F1 0.3922 (precision 0.5177, recall 0.3768), rank 127 | [`phase3_rhetorical_roles/`](phase3_rhetorical_roles/) |
| 4 | Court judgment prediction (CJPE) | macro-F1 0.7455 (precision 0.7480, recall 0.7441), rank 51 | [`phase4_court_judgment_prediction/`](phase4_court_judgment_prediction/) |
| 5 | Legal statute identification | macro-F1 0.4129 (micro-F1 0.5166, subset accuracy 0.0405), rank 27 | [`phase5_statute_identification/`](phase5_statute_identification/) |
| 6 | Abstractive summarization | BERTScore F1 0.8528 (ROUGE-1 0.5066, ROUGE-L 0.2634), rank 12 | [`phase6_summarization/`](phase6_summarization/) |
| 7 | Prior case retrieval | micro-F1@7 0.3916 (micro-P@7 0.3996, micro-R@7 0.3839), rank 65 | [`phase7_prior_case_retrieval/`](phase7_prior_case_retrieval/) |

## Repository layout

* `experiments/`: every experiment notebook we ran or wrote, grouped by phase and by architecture (GPT-2 decoder, ModernBERT encoder, T5, BiLSTM), with ranking notes, Codabench scores and the submitted predictions. Start with `experiments/README.md` and `experiments/ARCHITECTURE_COVERAGE.md`. Notebooks marked `not_run` were written for coverage and have no results.

* `phaseN_*/`: one notebook per phase (the notebook that produced the best submission) and a README with the approach, inputs and settings.
* `phase1_pretraining/scratch_models.py`: the encoder and decoder implemented from scratch in PyTorch, with an equivalence test against the reference library.
* `report/`: the LaTeX source of the report (`report.tex`, `figs/`). The compiled PDF is `report/report.pdf`.

## Running the notebooks

The notebooks are written for Kaggle (T4 GPU, Internet on). The course data are gated on Hugging Face: accept the dataset conditions, create a read token and add it as the Kaggle secret `HF_TOKEN`. Phases 2 to 7 need the Phase 1 checkpoints and the tokenizer (`legal_bpe_32k_mask.json`) attached as a Kaggle dataset. Phases 5, 6 and 7 restore the outputs of their earlier notebooks, as described in each notebook's header.

## Notes

* Notebooks 2 to 7 are exported as they ran on Kaggle, with stored outputs removed.
* Model weights are not stored in this repository.
