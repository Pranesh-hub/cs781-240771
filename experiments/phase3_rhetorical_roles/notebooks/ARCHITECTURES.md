# Phase 3: notebooks grouped by architecture

**Coverage.** Phase 3 was run only with the GPT-2-style decoder; the BiLSTM run is in the all-phases notebook. No separate ModernBERT or T5 notebook exists for this phase.

## 1_gpt2_decoder
GPT-2-style decoder (our Phase 1 model)

* `p3__gpt2__phase3_audited_fixed.ipynb`
* `p3__gpt2__phase3_full_fixed_3000step.ipynb`
* `p3__gpt2__phase3_periodic_kaggle_publishing.ipynb`
* `p3__gpt2__phase3_periodic_publishing_fixed.ipynb`
* `p3__gpt2__phase3_rr_colab_t4.ipynb`
* `p3__gpt2__phase3_rr_colab_tpu.ipynb`
* `p3__gpt2__phase3_rr_starter_pipeline.ipynb`
* `p3__gpt2__phase3_rr_stopiteration_fixed.ipynb`
* `p3__gpt2__phase3_resumable_checkpoints.ipynb`
* `p3__gpt2__diskfull_publisher_fix.ipynb`
* `p3__gpt2__incremental_sharded_recovery.ipynb`
* `p3__gpt2__notebook_kd_logits_recovery_fix.ipynb`
* `p3__gpt2__rolling_checkpoints_fixed.ipynb`
* `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c10.ipynb`
* `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c12.ipynb`
* `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c13.ipynb`
* `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c14.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c18.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c21.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c22.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c23.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c24.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c25.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c27.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c28.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c30.ipynb`
* `p3__gpt2__rr_training_pipeline__6d390d6f69_c04.ipynb`
* `p3__gpt2__rr_training_pipeline__6d390d6f69_c05.ipynb`
* `p3__gpt2__rr_training_pipeline__6d390d6f69_c06.ipynb`
* `p3__gpt2__rr_durable_recovery_publishing__6d390d6f69_c08.ipynb`
* `p3__gpt2__rr_rolling_checkpoints_publishing__6d390d6f69_c00.ipynb`
* `p3__gpt2__rr_pipeline.ipynb`

## 2_modernbert_encoder
ModernBERT-style encoder (our Phase 1 model)

* `p3__modernbert__rr_pipeline_full_finetune__not_run.ipynb` (written for coverage, NOT RUN: no results)

## 3_t5_encoder_decoder
T5-style encoder-decoder (our Phase 1 model)

* `p3__t5__rr_pipeline_full_finetune__not_run.ipynb` (written for coverage, NOT RUN: no results)

## 4_bilstm
BiLSTM (our Phase 1 model)

* `bilstm_all_phases__runs_phase_3_too.ipynb (copy of extra/bilstm_all_phases)`

## 5_other_architectures_mamba_rwkv_retnet
Other Phase 1 architectures (Mamba, RWKV, RetNet) and architecture-comparison notebooks

* none: no notebook for this architecture exists in this phase (see the phase README).

## 6_multi_architecture_pipelines
Pipelines that train or compare several architectures in one notebook

* none: no notebook for this architecture exists in this phase (see the phase README).
