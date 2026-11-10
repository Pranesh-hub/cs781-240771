# Phase 6: Abstractive summarization

**Best leaderboard result:** BERTScore F1 0.8528 (ROUGE-1 0.5066, ROUGE-L 0.2634), rank 12

**Approach.** Encoder-decoder from our own weights: the ModernBERT-style encoder reads four importance-selected chunks, the Phase 1 decoder gets new cross-attention layers; only Houlsby adapters and the cross-attention are trained. Beam search (4 beams) with a byte-level decoder attached to the tokenizer.

**Notebook.** `Phase6_Summarization.ipynb`

**Inputs.** Outputs of the earlier Phase 6 parts, Phase 1 checkpoints, tokenizer, Phase 6 data.

**How to run.** Open the notebook on Kaggle, attach the inputs above, turn Internet and a T4 GPU on, add the secret `HF_TOKEN` (needed to download the gated course data) and run all cells.
