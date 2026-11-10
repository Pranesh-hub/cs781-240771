# Phase 5: Legal statute identification

**Best leaderboard result:** macro-F1 0.4129 (micro-F1 0.5166, subset accuracy 0.0405), rank 27

**Approach.** Multi-label classification over 100 statutes: sigmoid heads with binary cross-entropy, mean of eight encoder runs, thresholds tuned on validation only.

**Notebook.** `Phase5_Statute_Identification.ipynb`

**Inputs.** Outputs of the earlier Phase 5 notebooks, Phase 1 checkpoints, tokenizer, Phase 5 data.

**How to run.** Open the notebook on Kaggle, attach the inputs above, turn Internet and a T4 GPU on, add the secret `HF_TOKEN` (needed to download the gated course data) and run all cells.
