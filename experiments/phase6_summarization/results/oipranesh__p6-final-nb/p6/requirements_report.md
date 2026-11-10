# Phase 6 (abstractive summarisation): requirement check

**NOT COMPLIANT: architecture not covered: bilstm**

| name                    | exp_id   | arch        | model      | technique   | family                 |   params_M |   trainable_M |   bertscore_f1 |   rouge1_f |   rougeL_f |
|:------------------------|:---------|:------------|:-----------|:------------|:-----------------------|-----------:|--------------:|---------------:|-----------:|-----------:|
| S5h_encdec_hier_houlsby | S5       | encdec_hier | modernbert | houlsby     | Houlsby adapters       |     227.91 |        33.125 |         0.8346 |     0.4784 |     0.2445 |
| S9_encdec_lorma         | S9       | encdec      | modernbert | lorma_plus  | LoRMA                  |     224.33 |        29.546 |         0.8213 |     0.3794 |     0.2051 |
| S8_gpt2_full            | S8       | gpt2        | gpt2       | full        | full fine-tuning       |     110.03 |       110.026 |         0.8175 |     0.3207 |     0.1821 |
| S6_gpt2_6L_kd           | S6       | gpt2        | gpt2       | kd          | knowledge distillation |      67.5  |        67.499 |         0.8165 |     0.3096 |     0.1805 |
| S3_gpt2_lora            | S3       | gpt2        | gpt2       | lora        | LoRA                   |     110.62 |         0.59  |         0.8151 |     0.3188 |     0.1835 |
| S1_t5_full              | S1       | t5          | t5         | full        | full fine-tuning       |      91.28 |        91.281 |         0.7819 |     0.3241 |     0.1919 |

- all four Phase 1 architectures: False (gpt2, modernbert, t5)
- LoRMA: True
- knowledge distillation: True
- technique families (5): Houlsby adapters, LoRA, LoRMA, full fine-tuning, knowledge distillation
- every experiment starts from our Phase 1 checkpoints; no outside pre-trained weights, corpora or APIs are used by any model (the BERTScore scoring model is a metric only)
- the old 3,000-step GPT-2 is not used anywhere

## Evaluation conventions
Validation = a held-out split of the public training data (703 documents; 400 scored per run). BERTScore F1: roberta-large, layer 17, rescale=False (library default, NOT necessarily the official setting). ROUGE: rouge_score, stemming True. Validation scores are not official test scores.

## Ablations / studies (not counted)
| name                      | arch   | technique   | strategy          |   bertscore_f1 |   rouge1_f |   rougeL_f |
|:--------------------------|:-------|:------------|:------------------|---------------:|-----------:|-----------:|
| LD_gpt2_concat_importance | gpt2   | full        | concat:importance |         0.8125 |     0.4049 |     0.211  |
| LD_gpt2_head              | gpt2   | full        | head              |         0.8153 |     0.3145 |     0.1798 |
| LD_gpt2_head_at128        | gpt2   | full        | head@128          |         0.8121 |     0.3051 |     0.1774 |
| LD_gpt2_head_tail         | gpt2   | full        | head_tail         |         0.8153 |     0.3173 |     0.18   |
| LD_gpt2_hier2_importance  | gpt2   | full        | hier2:importance  |         0.8123 |     0.3042 |     0.1736 |
| LD_gpt2_importance        | gpt2   | full        | importance        |         0.8196 |     0.3294 |     0.1851 |
| LD_gpt2_section           | gpt2   | full        | section           |         0.8188 |     0.3242 |     0.1837 |
| LD_gpt2_tail              | gpt2   | full        | tail              |         0.8172 |     0.324  |     0.1833 |
| S6c_gpt2_6L_nokd          | gpt2   | full        | head_tail         |         0.815  |     0.3171 |     0.1817 |
| Z0_gpt2_zero_shot         | gpt2   | zero-shot   | head              |         0.8033 |     0.2859 |     0.1689 |
