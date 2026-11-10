# Phase 7 requirement check

**ALL PHASE 7 EXPERIMENT RULES SATISFIED**

- counted experiments: ['E5_enc_prompt', 'E4_enc_ia3', 'E6_enc_half_kd', 'E2_enc_partial', 'E1_enc_full', 'E3_enc_lorma']
- techniques: ['full', 'ia3', 'kd', 'lorma_plus', 'partial', 'prompt']
- every experiment starts from the Phase 1 ModernBERT encoder; no outside model, weights, corpora or APIs; test labels never used
- prediction.json: F_final_rule_safe (dev micro-F1@7 0.3702); exactly 7 unique candidates per test query, all from the TEST pool

Dev numbers are local evaluation, not official test scores.