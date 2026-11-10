# Phase 1: experiments

**Best leaderboard result:** BERTScore F1 0.7973 (rank 140). The notebook behind it is in the main repository folder for this phase.

This folder collects everything tried in this phase, including runs that failed or were superseded.

* `notebooks/`: 45 distinct experiment notebooks (stored outputs removed; duplicates removed by code). They are grouped by architecture in subfolders; see `notebooks/ARCHITECTURES.md`.
* `results/`: run ledgers, reports and per-run result files saved from Kaggle.
* `submissions/`: the prediction files uploaded to Codabench (and the scoring files where we saved them).

## Codabench submissions (BERTScore F1)

| submission id | date | score |
|---|---|---|
| 965408 | 2026-10-06 23:33 | 0.7931 |
| 965382 | 2026-10-06 23:19 | 0.7933 |
| 963431 | 2026-10-06 06:38 | 0.7973 |
| 952981 | 2026-09-30 12:27 | 0.7662 |
| 952499 | 2026-09-30 05:44 | failed |
| 952481 | 2026-09-30 05:32 | failed |

## Notebooks

| file | cells | size | first line |
|---|---|---|---|
| `p1__gpt2__decoder_transformer_scaffold_v3__copy1.ipynb` | 62 | 75 KB | # CS781 Phase 1 — Decoder-only Transformer This notebook reuses the **verified Phase-4 data foundation** already prepared in the project: -  |
| `p1__gpt2__01_decoder_transformer.ipynb` | 9 | 11 KB | # CS781 Phase 1 — Decoder Transformer Standalone notebook using your verified 32K BPE + 4096 streaming corpus. |
| `p1__gpt2__01_decoder_transformer_completed_scaffold.ipynb` | 9 | 31 KB | # CS781 Phase 1 — Decoder-only Transformer This notebook reuses the **verified Phase-4 data foundation** already prepared in the project: -  |
| `p1__gpt2__01_decoder_transformer_completed_scaffold_v2.ipynb` | 9 | 35 KB | # CS781 Phase 1 — Decoder-only Transformer This notebook reuses the **verified Phase-4 data foundation** already prepared in the project: -  |
| `p1__gpt2__01_decoder_transformer_completed_scaffold_v3.ipynb` | 9 | 30 KB | # CS781 Phase 1 — Decoder-only Transformer This notebook reuses the **verified Phase-4 data foundation** already prepared in the project: -  |
| `p1__mamba__02_mamba_ssm.ipynb` | 9 | 11 KB | # CS781 Phase 1 — Mamba / SSM Standalone notebook using your verified 32K BPE + 4096 streaming corpus. |
| `p1__rwkv__03_rwkv.ipynb` | 9 | 11 KB | # CS781 Phase 1 — RWKV Standalone notebook using your verified 32K BPE + 4096 streaming corpus. |
| `p1__retnet__04_retnet.ipynb` | 9 | 11 KB | # CS781 Phase 1 — RetNet Standalone notebook using your verified 32K BPE + 4096 streaming corpus. |
| `p1__gpt2__phase1_starter_code.ipynb` | 45 | 19 KB | ## 1. Hugging Face Login #### This imports the token from Kaggle Secrets, so make sure the secret is set correctly before running the code. |
| `p1__rwkv__rwkv_training__arch2_cs781_1.ipynb` | 33 | 71 KB | import os import gc import json import math import time import random import zipfile from pathlib import Path import torch import torch.nn a |
| `p1__rwkv__rwkv_training__arch2_cs781_2.ipynb` | 41 | 83 KB | import os import gc import json import math import time import random import zipfile from pathlib import Path import torch import torch.nn a |
| `p1__rwkv__rwkv_training__arch2_cs781.ipynb` | 17 | 45 KB | import os import gc import json import math import time import random import zipfile from pathlib import Path import torch import torch.nn a |
| `p1__rwkv__rwkv_training__arch2_rwkv_checkpointed.ipynb` | 41 | 91 KB | import os import gc import json import math import time import random import zipfile from pathlib import Path import torch import torch.nn a |
| `p1__mamba__mamba_experiment_kaggle__cs781_c01.ipynb` | 65 | 371 KB | # ============================================================ # MAMBA EXPERIMENT — KAGGLE ENVIRONMENT SETUP # ============================= |
| `p1__mamba__mamba_experiment_kaggle__cs781_c02.ipynb` | 70 | 415 KB | # ============================================================ # MAMBA EXPERIMENT — KAGGLE ENVIRONMENT SETUP # ============================= |
| `p1__bilstm__bislstm.ipynb` | 8 | 30 KB | # CS781 Phase 1 - Notebook 4 of 4: RECURRENT language model (bidirectional LSTM, ELMo-style) / / / /---/---/ / **family** / recurrent networ |
| `p1__bilstm__bislstm_2.ipynb` | 10 | 40 KB | # CS781 Phase 1 - Notebook 4 of 4: RECURRENT language model (bidirectional LSTM, ELMo-style) / / / /---/---/ / **family** / recurrent networ |
| `p1__modernbert__modernbert_copy1.ipynb` | 12 | 52 KB | # CS781 Phase 1 — Notebook 2 of 4, VERSION 2: ENCODER-ONLY Transformer (ModernBERT-style, masked-language modelling) **Why a version 2:** th |
| `p1__modernbert__modernbert.ipynb` | 9 | 30 KB | # CS781 Phase 1 - Notebook 2 of 4: ENCODER-ONLY Transformer (bidirectional, masked-language modelling) / / / /---/---/ / **family** / encode |
| `p1__t5__t5.ipynb` | 9 | 38 KB | # CS781 Phase 1 - Notebook 3 of 4: ENCODER-DECODER Transformer (T5-style, span denoising) / / / /---/---/ / **family** / encoder-decoder Tra |
| `p1__mamba__mamba_experiment_kaggle__cs781_c00.ipynb` | 25 | 61 KB | # ============================================================ # MAMBA EXPERIMENT — KAGGLE ENVIRONMENT SETUP # ============================= |
| `p1__mamba__mamba_resumable_training.ipynb` | 14 | 47 KB | # CS781 Phase 1 — Mamba legal language-model training A compact training notebook reusing the existing Mamba architecture and legal-corpus s |
| `p1__gpt2__phase1_resumable_streaming_3000_steps.ipynb` | 16 | 42 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p1__rwkv__rwkv_clean_resumable.ipynb` | 17 | 38 KB | # CS781 Phase 1 — RWKV training and inference (cleaned) This version removes the conflicting experimental/debug cells and keeps one model de |
| `p1__gpt2__decoder_gpt2_nb1_of_4__gptlm_new.ipynb` | 12 | 49 KB | # CS781 Phase 1 - Notebook 1 of 4: DECODER-ONLY Transformer (causal language model) / / / /---/---/ / **family** / decoder-only Transformer  |
| `p1__mamba__mamba-cs781.ipynb` | 13 | 23 KB | import os import gc import json import math import time import random import zipfile from pathlib import Path import numpy as np import torc |
| `p1__gpt2__decoder_resumable_training__1fcc618882_c01.ipynb` | 16 | 34 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p1__gpt2__decoder_resumable_training__1fcc618882_c02.ipynb` | 26 | 44 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p1__gpt2__decoder_resumable_training__1fcc618882_c03.ipynb` | 23 | 42 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p1__gpt2__decoder_resumable_training__1fcc618882_c00.ipynb` | 11 | 25 KB | # Phase 1 Legal LM — resumable training with periodic Kaggle Dataset checkpoint backups This notebook initializes a decoder-only GPT-style l |
| `p1__gpt2__decoder_resumable_training__1fcc618882_c99.ipynb` | 296 | 492 KB | # CS781 Phase 1 — Resumable decoder-only Transformer training This version preserves the project's existing Phase-4 data foundation and stre |
| `p1__rwkv__rwkv_training__6d390d6f69_c01.ipynb` | 44 | 89 KB | import os import gc import json import math import time import random import zipfile from pathlib import Path import torch import torch.nn a |
| `p1__rwkv__rwkv_training_clean__6d390d6f69_c02.ipynb` | 18 | 37 KB | # CS781 Phase 1 — RWKV training and inference (cleaned) This version removes the conflicting experimental/debug cells and keeps one model de |
| `p1__mamba__mamba_training__6d390d6f69_c00.ipynb` | 18 | 39 KB | # CS781 Phase 1 — Mamba legal language-model training A compact training notebook reusing the existing Mamba architecture and legal-corpus s |
| `p1__gpt2__nb1_decoder_gpt2.ipynb` | 8 | 32 KB | # CS781 Phase 1 - Notebook 1 of 4: DECODER-ONLY Transformer (causal language model) / / / /---/---/ / **family** / decoder-only Transformer  |
| `p1__modernbert__nb2_encoder_modernbert.ipynb` | 8 | 33 KB | # CS781 Phase 1 - Notebook 2 of 4: ENCODER-ONLY Transformer (bidirectional, masked-language modelling) / / / /---/---/ / **family** / encode |
| `p1__modernbert__nb2_encoder_modernbert_v2.ipynb` | 10 | 45 KB | # CS781 Phase 1 — Notebook 2 of 4, VERSION 2: ENCODER-ONLY Transformer (ModernBERT-style, masked-language modelling) **Why a version 2:** th |
| `p1__t5+modernbert+gpt2__nb3_encoder_decoder_t5.ipynb` | 8 | 34 KB | # CS781 Phase 1 - Notebook 3 of 4: ENCODER-DECODER Transformer (T5-style, span denoising) / / / /---/---/ / **family** / encoder-decoder Tra |
| `p1__gpt2__1000_step_inference_only.ipynb` | 10 | 12 KB | # CS781 Phase 1 — Load the 1,000-step model and generate predictions This is a trimmed inference-only notebook extracted from the provided n |
| `p1__gpt2__continued_pretraining.ipynb` | 10 | 24 KB | # CS781 — continued pretraining of the Phase 1 language model (fixed recipe, held-out validation) **Why this notebook exists.** The 3,000-st |
| `p1__gpt2+modernbert+t5+bilstm__four_architectures.ipynb` | 13 | 46 KB | # CS781 Phase 1 v3 — four different foundation-model architectures on one shared tokenizer, corpus and recipe The spec asks for **four diffe |
| `p1__gpt2__resumable_training_2000_3000_steps.ipynb` | 10 | 29 KB | # Phase 1 Legal LM — resumable training with periodic Kaggle Dataset checkpoint backups This notebook initializes a decoder-only GPT-style l |
| `p1__gpt2__v2_gpt2_architecture.ipynb` | 11 | 32 KB | # CS781 Phase 1 v2 — better GPT-2-family language model (architecture ablation + a training recipe that fixes the original run's problems) # |
