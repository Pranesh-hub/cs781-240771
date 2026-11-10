# Phase 2: Legal named entity recognition

**Best leaderboard result:** strict macro-F1 0.3232 (precision 0.3954, recall 0.2902), rank 119

**Approach.** BIO token classification with the Phase 1 decoder over 512-token windows (stride 448), per-label confidence thresholds, and counsel / JUDGE rule layers applied after the model.

**Notebook.** `Phase2_Legal_NER.ipynb`

**Inputs.** Phase 1 decoder checkpoint, tokenizer, Phase 2 data.

**How to run.** Open the notebook on Kaggle, attach the inputs above, turn Internet and a T4 GPU on, add the secret `HF_TOKEN` (needed to download the gated course data) and run all cells.
