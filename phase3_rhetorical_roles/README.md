# Phase 3: Rhetorical role prediction

**Best leaderboard result:** macro-F1 0.3922 (precision 0.5177, recall 0.3768), rank 127

**Approach.** Sentence classification into 13 roles for the CL and IT domains, with 3 previous and 1 following sentence as context. Compares full fine-tuning, LoRA, LoRMA+, IA3, QLoRA, prompt tuning and a distilled student; tunes ensembling, transition smoothing and class biases.

**Notebook.** `Phase3_Rhetorical_Roles.ipynb`

**Inputs.** Phase 1 decoder checkpoint, tokenizer, Phase 3 data.

**How to run.** Open the notebook on Kaggle, attach the inputs above, turn Internet and a T4 GPU on, add the secret `HF_TOKEN` (needed to download the gated course data) and run all cells.
