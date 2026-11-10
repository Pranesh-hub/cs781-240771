# Part 6 (improvement notebook): what changed, in validation numbers (not official test scores)

- previous best counted run, as originally scored: S5_encdec_houlsby (BERTScore F1 0.8217, ROUGE-1 0.3787)
- new runs of this notebook (re-scored with the same scorer):

```
                         name        arch technique   strategy  in_tokens  bertscore_f1  rouge1_f  rougeL_f  fact_coverage  rouge1_f_short  rouge1_f_long
      S5h_encdec_hier_houlsby encdec_hier   houlsby importance     2048.0        0.8346    0.4784    0.2445         0.2993          0.5253         0.4310
S5i_encdec_houlsby_importance      encdec   houlsby importance      512.0        0.8255    0.4177    0.2186         0.2139          0.4537         0.3815
```
- BiLSTM (S7): did NOT run (checkpoint not attached): the architecture rule remains unmet
- final system: S5h_encdec_hier_houlsby with decoding 'beam 4' | requirement status: NOT COMPLIANT: architecture not covered: bilstm

Every number above is a validation number on a held-out part of the public training data; the official test scores come from the course evaluator.