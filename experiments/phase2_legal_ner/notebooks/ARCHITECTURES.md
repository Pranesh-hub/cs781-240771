# Phase 2: notebooks grouped by architecture

**Coverage.** Phase 2 was run only with the GPT-2-style decoder (plus rule layers); the BiLSTM run is in the all-phases notebook. No separate ModernBERT or T5 notebook exists for this phase.

## 1_gpt2_decoder
GPT-2-style decoder (our Phase 1 model)

* `p2__gpt2__experiments_clean.ipynb`
* `p2__gpt2__champion_plus_rules_run_cells.ipynb`
* `p2__gpt2__new_self_contained_experiments.ipynb`
* `p2__gpt2__ner_clean_experiments__finalv1_c01.ipynb`
* `p2__gpt2__ner_clean_experiments__finalv1_c00.ipynb`
* `p2__gpt2__ner_clean_experiments_with_exp15__finalv1.ipynb`
* `p2__gpt2__ner_self_contained_experiments__finalv2_c01.ipynb`
* `p2__gpt2__ner_self_contained_experiments__finalv2_c02.ipynb`
* `p2__gpt2__ner_self_contained_experiments__finalv2_c03.ipynb`
* `p2__gpt2__ner_self_contained_experiments__finalv2_c04.ipynb`
* `p2__gpt2__ner_self_contained_experiments__finalv2_c05.ipynb`
* `p2__gpt2__ner_self_contained_experiments__finalv2_c00.ipynb`
* `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c10.ipynb`
* `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c04.ipynb`
* `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c05.ipynb`
* `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c06.ipynb`
* `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c07.ipynb`
* `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c08.ipynb`
* `p2__gpt2__ner_decoder_training_plus_ner__1fcc618882_c09.ipynb`
* `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c01.ipynb`
* `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c02.ipynb`
* `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c03.ipynb`
* `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c04.ipynb`
* `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c05.ipynb`
* `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c06.ipynb`
* `p2__gpt2__ner_champion_counsel_judge_inference__a30dd7235a_c00.ipynb`

## 2_modernbert_encoder
ModernBERT-style encoder (our Phase 1 model)

* `p2__modernbert__ner_tagger_with_rules__not_run.ipynb` (written for coverage, NOT RUN: no results)

## 3_t5_encoder_decoder
T5-style encoder-decoder (our Phase 1 model)

* `p2__t5__ner_tagger_with_rules__not_run.ipynb` (written for coverage, NOT RUN: no results)

## 4_bilstm
BiLSTM (our Phase 1 model)

* `bilstm_all_phases__runs_phase_2_too.ipynb (copy of extra/bilstm_all_phases)`

## 5_other_architectures_mamba_rwkv_retnet
Other Phase 1 architectures (Mamba, RWKV, RetNet) and architecture-comparison notebooks

* none: no notebook for this architecture exists in this phase (see the phase README).

## 6_multi_architecture_pipelines
Pipelines that train or compare several architectures in one notebook

* none: no notebook for this architecture exists in this phase (see the phase README).
