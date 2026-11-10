# Phase 7: Prior case retrieval

**Best leaderboard result:** micro-F1@7 0.3916 (micro-P@7 0.3996, micro-R@7 0.3839), rank 65

**Approach.** Bi-encoder on the Phase 1 encoder trained with InfoNCE and BM25 hard negatives; pool-aware re-ranking and a BM25 hybrid for the best submission.

**Notebook.** `Phase7_Prior_Case_Retrieval.ipynb`

**Inputs.** Outputs of the first Phase 7 notebook, Phase 1 encoder, tokenizer, Phase 7 data.

**How to run.** Open the notebook on Kaggle, attach the inputs above, turn Internet and a T4 GPU on, add the secret `HF_TOKEN` (needed to download the gated course data) and run all cells.
