# Phase 6 analysis (validation numbers; not official test scores)

Selection metric: **bertscore_f1**. BERTScore setting: roberta-large, layer 17, rescale=False (library default, not necessarily the official one). ROUGE: rouge_score.

## Which techniques work best

| run | architecture | technique | trainable M | bertscore_f1 | ROUGE-1 | ROUGE-L | fact coverage |
|---|---|---|---|---|---|---|---|
| S5_encdec_houlsby | modernbert | Houlsby adapters | 33.12 | 0.8217 | 0.3787 | 0.2065 | 0.207 |
| S9_encdec_lorma | modernbert | LoRMA | 29.55 | 0.8213 | 0.3794 | 0.2051 | 0.191 |
| S8_gpt2_full | gpt2 | full fine-tuning | 110.03 | 0.8175 | 0.3207 | 0.1821 | 0.134 |
| S6_gpt2_6L_kd | gpt2 | knowledge distillation | 67.50 | 0.8165 | 0.3096 | 0.1805 | 0.120 |
| S3_gpt2_lora | gpt2 | LoRA | 0.59 | 0.8151 | 0.3188 | 0.1835 | 0.121 |
| S4_gpt2_prompt | gpt2 | prompt tuning | 0.01 | 0.8133 | 0.3179 | 0.1851 | 0.108 |
| S1_t5_full | t5 | full fine-tuning | 91.28 | 0.7819 | 0.3241 | 0.1919 | 0.116 |
| S2_t5_lorma | t5 | LoRMA | 0.74 | 0.7721 | 0.2889 | 0.1810 | 0.042 |

Best: **S5_encdec_houlsby** (Houlsby adapters on modernbert). Differences smaller than the paired-bootstrap intervals printed below are not evidence of a better technique.

Best vs second (bertscore_f1, 400 shared documents): +0.0004, 95% interval [-0.0003, +0.0013].

## Do parameter-efficient methods approach full fine-tuning?

- t5: best PEFT S2_t5_lorma (LoRMA, 0.74M trained) reaches 0.7721 vs full 0.7819 (98.7% of full) with 0.81% of the trainable parameters.
- gpt2: best PEFT S3_gpt2_lora (LoRA, 0.59M trained) reaches 0.8151 vs full 0.8175 (99.7% of full) with 0.54% of the trainable parameters.
- encdec: PEFT runs ['S5_encdec_houlsby', 'S9_encdec_lorma'] but no full fine-tune of this architecture to compare with.

## Long-document strategies

GPT-2 study (8 settings): best **importance** (0.8196), worst head@128 (0.8121). Strategy used by the experiments: head_tail.
- importance: 0.8196 | ROUGE-1 0.3294 | fact coverage 0.132 | length ratio 0.48 | ROUGE-1 short/long docs 0.3795 / 0.2728
- section: 0.8188 | ROUGE-1 0.3242 | fact coverage 0.143 | length ratio 0.47 | ROUGE-1 short/long docs 0.3761 / 0.2656
- tail: 0.8172 | ROUGE-1 0.3240 | fact coverage 0.140 | length ratio 0.48 | ROUGE-1 short/long docs 0.3780 / 0.2632
- head_tail: 0.8153 | ROUGE-1 0.3173 | fact coverage 0.127 | length ratio 0.49 | ROUGE-1 short/long docs 0.3699 / 0.2580
- head: 0.8153 | ROUGE-1 0.3145 | fact coverage 0.124 | length ratio 0.49 | ROUGE-1 short/long docs 0.3657 / 0.2568
- concat:importance: 0.8125 | ROUGE-1 0.4049 | fact coverage 0.228 | length ratio 1.33 | ROUGE-1 short/long docs 0.4204 / 0.3874
- hier2:importance: 0.8123 | ROUGE-1 0.3042 | fact coverage 0.111 | length ratio 0.47 | ROUGE-1 short/long docs 0.3555 / 0.2464
- head@128: 0.8121 | ROUGE-1 0.3051 | fact coverage 0.114 | length ratio 0.48 | ROUGE-1 short/long docs 0.3566 / 0.2470

Information lost before any model reads the text (budget 512 tokens): share of reference key facts present in the input: head 0.359, tail 0.281, head_tail 0.340, importance 0.323, section 0.348, oracle 0.378, overlapping chunks (all, <=6) 0.622. The rest is lost by truncation / selection, whatever the model.

## Are legal facts, reasoning and decisions preserved? Coverage, redundancy, hallucination

Final setting (S5_encdec_houlsby, 'beam 4', 200 documents): key-fact coverage 0.216 (short 0.260 / long 0.167); decision agreement 0.439 (documents whose reference states a decision); unsupported numbers / names 0.238 (hallucination proxy); repeated trigrams 0.021; length ratio 0.51; ROUGE-1 short/long documents 0.4019 / 0.2958.
Extractive reference points (not submissions): B1_lead_k: ROUGE-1 0.4957, fact coverage 0.475; B2_importance_k: ROUGE-1 0.5457, fact coverage 0.414; B3_ORACLE_k (upper bound, reads the reference): ROUGE-1 0.6182, fact coverage 0.496.

**Example 1** (doc 1548, 10465 tokens)
- reference: The Municipal Board of Pushkar passed a resolution on May 24, 1948, for the shifting of the bus stand on the sacred lake to another site near the Police Station where it resolved to construct a passenger shed and provide other facilities for the pilgrims. The Regional Transport Authority by a resolution dated December 3/4, 1959, accepted the suggestion and issued a public notification to that effect on June 28, 1960. Long before the notification, two residents of Pushkar moved the State Transport Authority under section 64A of the Motor Vehicles Act for revision of the decision of the Regional...
- generated: The appellant Municipality issued a notification under section 3 ( 1 ) of the , to the State Transport Authority under section 64 A of the Act . The appellant filed a writ petition in the High Court under article 226 of the Constitution . In the appeal to this Court , it was contended on behalf of the appellant that the Regional Transport Authority had no jurisdiction to entertain the petition . Held ( per Sarkar , C . J ., H id ayat ullah , J ., dissenting ), that the State Government had no power to entertain an application for a writ of certiorari to quash the notification issued under sect...

**Example 2** (doc 3847, 400 tokens)
- reference: The appellant paid the correct court fee under Section 37(2) of the Madras Court Fees and Suits Valuation Act clearly alleging in para 14 of the plaint that she is in joint possession and is seeking partition and separate possession of her half share in the suit properties as heir of deceased Paramayee. The preliminary objection as to the correct court fee payable raised and taken up resulted in the final appeal before this court. Allowing the appeal by Special Leave, the Court ^ HELD: (a) Courts should be anxious to grapple with the real issues and not spend their energies on peripheral ones....
- generated: The plaintiff respondent filed a suit for partition of his share in the suit properties for partition and for a declaration that the plaintiff respondent was entitled to partition in the properties of the deceased plaintiff . The trial court dismissed the suit on the ground that the suit was barred by section 17 ( b ) of the Central Civil Procedure Code . The High Court dismissed the plaintiff ' s suit . In the appeal to this Court it was contended on behalf of the appellant that the decree passed by the trial court was not binding on the plaintiff and the plaintiff was not entitled to the ben...

## Generation length, decoding and redundancy controls

- default (greedy, trigram block, dedupe): bertscore_f1 0.8198, ROUGE-1 0.3692, ROUGE-L 0.2007, length ratio 0.66, repetition 0.039, 0.69 s/doc
- no n-gram blocking: bertscore_f1 0.8077, ROUGE-1 0.1710, ROUGE-L 0.1206, length ratio 0.26, repetition 0.302, 0.69 s/doc
- no dedupe post-processing: bertscore_f1 0.8196, ROUGE-1 0.3700, ROUGE-L 0.2010, length ratio 0.66, repetition 0.040, 0.69 s/doc
- repetition penalty 1.2: bertscore_f1 0.8238, ROUGE-1 0.3572, ROUGE-L 0.1850, length ratio 0.53, repetition 0.014, 0.67 s/doc
- shorter: max_new x0.5: bertscore_f1 0.8224, ROUGE-1 0.3314, ROUGE-L 0.1894, length ratio 0.45, repetition 0.036, 0.35 s/doc
- max_new x0.75: bertscore_f1 0.8212, ROUGE-1 0.3604, ROUGE-L 0.1988, length ratio 0.58, repetition 0.038, 0.52 s/doc
- longer: min_new x0.5: bertscore_f1 0.8191, ROUGE-1 0.3918, ROUGE-L 0.2073, length ratio 0.74, repetition 0.040, 0.69 s/doc
- nucleus sampling p0.9 T0.7: bertscore_f1 0.8202, ROUGE-1 0.3934, ROUGE-L 0.2030, length ratio 0.71, repetition 0.023, 0.72 s/doc
- beam 4: bertscore_f1 0.8242, ROUGE-1 0.3520, ROUGE-L 0.1907, length ratio 0.51, repetition 0.021, 3.22 s/doc
- beam 4, length penalty 1.5 (longer): bertscore_f1 0.8234, ROUGE-1 0.3872, ROUGE-L 0.2009, length ratio 0.62, repetition 0.023, 3.21 s/doc
- beam 4, length penalty 0.7 (shorter): bertscore_f1 0.8242, ROUGE-1 0.3503, ROUGE-L 0.1906, length ratio 0.50, repetition 0.021, 3.20 s/doc
- hierarchical: chunk summaries -> second stage: bertscore_f1 0.8152, ROUGE-1 0.3906, ROUGE-L 0.2104, length ratio 0.90, repetition 0.049, 2.57 s/doc
- hierarchical: chunk summaries joined: bertscore_f1 0.8217, ROUGE-1 0.4386, ROUGE-L 0.2211, length ratio 1.66, repetition 0.199, 1.27 s/doc

Spearman correlation across settings between length ratio and bertscore_f1: -0.36; with ROUGE-1: 0.96.

## Domain adaptation and the usefulness of the Phase 1 representations

Phase 1 GPT-2 zero-shot (legal-domain pre-training only): ROUGE-1 0.2859 -> after summarisation fine-tuning (S8): 0.3207. No general-domain model may be used, so 'legal vs general pre-training' cannot be measured here; the evidence is fine-tuned vs zero-shot vs extractive baselines.
With the Phase 1 weights FROZEN, the best parameter-efficient run reaches 0.8217 (S5_encdec_houlsby): the Phase 1 representations carry most of what that model uses.
BiLSTM (barely pre-trained): not run (checkpoint not attached).

## Knowledge distillation

- S8_gpt2_full: 110.0M params | bertscore_f1 0.8175 | 0.18 s/doc | peak memory train / generation 9.40 / 0.94 GB | checkpoint 514 MiB
- S6_gpt2_6L_kd: 67.5M params | bertscore_f1 0.8165 | 0.11 s/doc | peak memory train / generation 9.02 / 0.62 GB | checkpoint 351 MiB
- S6c_gpt2_6L_nokd: 67.5M params | bertscore_f1 0.8150 | 0.11 s/doc | peak memory train / generation 6.92 / 1.87 GB | checkpoint 351 MiB
Distillation helps only if the KD student beats the same student without distillation (S6c) by more than the seed noise; compare the two rows above.

## Model size, memory and inference cost

             name  params_M  trainable_M  ckpt_mb  peak_mem_train_gb  peak_mem_gen_gb  gen_sec_per_doc  train_minutes
S5_encdec_houlsby    227.91       33.125    126.5               7.66             3.00            0.628           32.0
  S9_encdec_lorma    224.33       29.546    112.8               7.73             2.98            0.667           33.6
     S8_gpt2_full    110.03      110.026    513.5               9.40             0.94            0.184           20.5
    S6_gpt2_6L_kd     67.50       67.499    351.3               9.02             0.62            0.109           29.9
     S3_gpt2_lora    110.62        0.590      2.3               9.11             2.18            0.277           22.2
   S4_gpt2_prompt    110.04        0.015      0.1               8.71             2.19            0.203           20.1
       S1_t5_full     91.28       91.281    582.6               4.46             1.56            0.706           94.5
      S2_t5_lorma     92.02        0.737      2.8               4.15             2.92            0.697           95.8

## What was NOT done
- prefix tuning, AdaLoRA and QLoRA are not implemented (not needed for the rules; QLoRA needs bitsandbytes on a GPU).
- the importance selector is an unsupervised within-document heuristic, not a learned Phase 1 selector.
- validation numbers only; the official test scores are computed by the course evaluator.