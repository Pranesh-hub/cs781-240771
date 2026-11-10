# Phase 1: Legal language-model pretraining

**Best leaderboard result:** BERTScore F1 0.7973 (ROUGE-L 0.1782, BLEU-4 0.0112), rank 140

**Approach.** A GPT-2-style decoder (112.8M parameters, 12 layers x 768, 4,096 positions, 32,000-token BPE vocabulary) pre-trained for 1,000 steps produced the submission by generating 128-word continuations (temperature 0.8, top-k 50). The notebook also pre-trains the ModernBERT-style encoder (85.4M parameters) that the later phases build on. Both models are implemented from scratch in plain PyTorch (no model class from a library).

**Notebook.** `Phase1_pretraining_from_scratch.ipynb`

**Inputs.** Deduplicated legal corpus and the 32k BPE tokenizer (Part A); the Phase 1 decoder checkpoint and test.json (Part B).

**How to run.** Open the notebook on Kaggle, attach the inputs above, turn Internet and a T4 GPU on, add the secret `HF_TOKEN` (needed to download the gated course data) and run all cells.

**Extra files.** `scratch_models.py` holds the from-scratch ModernBERT-style encoder and GPT-2-style decoder; `test_scratch_equivalence.py` checks them against the reference library with the same weights (encoder logits differ by exactly 0.0, decoder logits by 2e-6, parameter counts are identical: 85,380,609 and 112,777,728). The test needs `torch` and `transformers`.
