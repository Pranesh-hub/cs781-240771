# Phase 4: Court judgment prediction (CJPE)

**Best leaderboard result:** macro-F1 0.7455 (precision 0.7480, recall 0.7441), rank 51

**Approach.** Window classifier over the last four 512-token windows of a judgment (eight at inference) with the Phase 1 decoder and the ModernBERT-style encoder, ensemble, and a test threshold rate-matched to the development positive rate.

**Notebook.** `Phase4_Court_Judgment_Prediction.ipynb`

**Inputs.** Phase 1 decoder and encoder checkpoints, tokenizer, Phase 4 data.

**How to run.** Open the notebook on Kaggle, attach the inputs above, turn Internet and a T4 GPU on, add the secret `HF_TOKEN` (needed to download the gated course data) and run all cells.
