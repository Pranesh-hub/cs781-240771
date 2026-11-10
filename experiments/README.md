# CS781 experiments, phase by phase

Every experiment we ran, collected per phase. The best code of each phase is in the main repository folders; this folder is the full record.

| folder | what it holds |
|---|---|
| [`phase1_pretraining/`](phase1_pretraining/) | Phase 1: 45 notebooks, results, submissions |
| [`phase2_legal_ner/`](phase2_legal_ner/) | Phase 2: 26 notebooks, results, submissions |
| [`phase3_rhetorical_roles/`](phase3_rhetorical_roles/) | Phase 3: 32 notebooks, results, submissions |
| [`phase4_court_judgment_prediction/`](phase4_court_judgment_prediction/) | Phase 4: 14 notebooks, results, submissions |
| [`phase5_statute_identification/`](phase5_statute_identification/) | Phase 5: 8 notebooks, results, submissions |
| [`phase6_summarization/`](phase6_summarization/) | Phase 6: 17 notebooks, results, submissions |
| [`phase7_prior_case_retrieval/`](phase7_prior_case_retrieval/) | Phase 7: 8 notebooks, results, submissions |
| [`data_analysis/`](data_analysis/) | dataset statistics computed for the report |
| [`extra/`](extra/) | capstone-series notebooks and the BiLSTM single-notebook run |
| `codabench_scores.csv` | every Codabench submission we made, with its score (transcribed from the submission pages on 9 November 2026) |
| `RANKING.md`, `runs_index.csv` | the best runs per phase |

## Notes

* Notebooks are exported as they ran on Kaggle, with stored outputs removed. Several are early or superseded versions.
* Model weights and checkpoints are not included.
* Scores are development or validation numbers unless a Codabench score is given.
* Phase 2: the six-experiment requirement is not evidenced; the Phase 2 record is the champion pipeline plus rule layers.

## Older Kaggle versions

Kaggle keeps every saved version of a notebook. The latest versions were collected first. A second pass pulled the code of all earlier versions of the CS781 notebooks (Kaggle API, version labels `v1`, `v2`, ...) and kept only versions whose code differs from every notebook already in this folder. Nine such versions were added, in `phaseN_*/notebooks/older_versions/` (index: `older_versions_index.csv`; file names are `<account>__<notebook>__v<N>.ipynb`). Versions that only changed outputs or whitespace, and notebooks from other courses on the same accounts, were left out. Outputs are not included for older versions.


See `ARCHITECTURE_COVERAGE.md` for which of the four architectures each phase used.

## File naming

Every notebook under `phaseN_*/notebooks/` is named `pN__<models>__<topic>.ipynb`:

* `pN`: the phase (`p4` = Phase 4).
* `<models>`: the architectures the notebook builds or trains, joined by `+`: `gpt2` (decoder), `modernbert` (encoder), `t5` (encoder-decoder), `bilstm`, and for Phase 1 also `mamba`, `rwkv`, `retnet`. A notebook that handles several models lists all of them, for example `gpt2+modernbert+t5+bilstm`. The list shows what the code contains. Whether a given model produced a ranked result is in `ARCHITECTURE_COVERAGE.md`.
* `<topic>`: what the notebook does. Kaggle's auto-generated names were replaced by a description plus the Kaggle id and copy number (`__6d390d6f69_c04`: copy `(4)` of that Kaggle notebook; the copy number is the download order, not the run order).

`RENAME_MAP.csv` maps original paths to current paths where the mapping could be recovered.
