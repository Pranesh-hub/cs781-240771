# Phase 1: notebooks grouped by architecture

**Coverage.** Phase 1 has pretraining notebooks for all four architectures: decoder (1), encoder (2), T5 (3), BiLSTM (4), plus Mamba/RWKV/RetNet (5).

## 1_gpt2_decoder
GPT-2-style decoder (our Phase 1 model)
**Best:** `p1__gpt2__1000_step_inference_only.ipynb`

* `p1__gpt2__decoder_transformer_scaffold_v3__copy1.ipynb`
* `p1__gpt2__01_decoder_transformer.ipynb`
* `p1__gpt2__01_decoder_transformer_completed_scaffold.ipynb`
* `p1__gpt2__01_decoder_transformer_completed_scaffold_v2.ipynb`
* `p1__gpt2__01_decoder_transformer_completed_scaffold_v3.ipynb`
* `p1__gpt2__phase1_starter_code.ipynb`
* `p1__gpt2__phase1_resumable_streaming_3000_steps.ipynb`
* `p1__gpt2__decoder_gpt2_nb1_of_4__gptlm_new.ipynb`
* `p1__gpt2__decoder_resumable_training__1fcc618882_c01.ipynb`
* `p1__gpt2__decoder_resumable_training__1fcc618882_c02.ipynb`
* `p1__gpt2__decoder_resumable_training__1fcc618882_c03.ipynb`
* `p1__gpt2__decoder_resumable_training__1fcc618882_c00.ipynb`
* `p1__gpt2__decoder_resumable_training__1fcc618882_c99.ipynb`
* `p1__gpt2__nb1_decoder_gpt2.ipynb`
* `p1__gpt2__1000_step_inference_only.ipynb`
* `p1__gpt2__continued_pretraining.ipynb`
* `p1__gpt2__resumable_training_2000_3000_steps.ipynb`
* `p1__gpt2__v2_gpt2_architecture.ipynb`

## 2_modernbert_encoder
ModernBERT-style encoder (our Phase 1 model)
**Best:** `p1__modernbert__nb2_encoder_modernbert_v2.ipynb`

* `p1__modernbert__modernbert_copy1.ipynb`
* `p1__modernbert__modernbert.ipynb`
* `p1__modernbert__nb2_encoder_modernbert.ipynb`
* `p1__modernbert__nb2_encoder_modernbert_v2.ipynb`

## 3_t5_encoder_decoder
T5-style encoder-decoder (our Phase 1 model)
**Best:** `p1__t5+modernbert+gpt2__nb3_encoder_decoder_t5.ipynb`

* `p1__t5__t5.ipynb`
* `p1__t5+modernbert+gpt2__nb3_encoder_decoder_t5.ipynb`

## 4_bilstm
BiLSTM (our Phase 1 model)
**Best:** `p1__bilstm__bislstm_2.ipynb`

* `p1__bilstm__bislstm.ipynb`
* `p1__bilstm__bislstm_2.ipynb`

## 5_other_architectures_mamba_rwkv_retnet
Other Phase 1 architectures (Mamba, RWKV, RetNet) and architecture-comparison notebooks
**Best:** `None.ipynb`(none; see folder README)

* `p1__mamba__02_mamba_ssm.ipynb`
* `p1__rwkv__03_rwkv.ipynb`
* `p1__retnet__04_retnet.ipynb`
* `p1__rwkv__rwkv_training__arch2_cs781_1.ipynb`
* `p1__rwkv__rwkv_training__arch2_cs781_2.ipynb`
* `p1__rwkv__rwkv_training__arch2_cs781.ipynb`
* `p1__rwkv__rwkv_training__arch2_rwkv_checkpointed.ipynb`
* `p1__mamba__mamba_experiment_kaggle__cs781_c01.ipynb`
* `p1__mamba__mamba_experiment_kaggle__cs781_c02.ipynb`
* `p1__mamba__mamba_experiment_kaggle__cs781_c00.ipynb`
* `p1__mamba__mamba_resumable_training.ipynb`
* `p1__rwkv__rwkv_clean_resumable.ipynb`
* `p1__mamba__mamba-cs781.ipynb`
* `p1__rwkv__rwkv_training__6d390d6f69_c01.ipynb`
* `p1__rwkv__rwkv_training_clean__6d390d6f69_c02.ipynb`
* `p1__mamba__mamba_training__6d390d6f69_c00.ipynb`

## 6_multi_architecture_pipelines
Pipelines that train or compare several architectures in one notebook

* `p1__gpt2+modernbert+t5+bilstm__four_architectures.ipynb`
