# Combined Part 7: quick wins + retraining (validation numbers; not official test scores)

- time: 346 of 400 budget minutes
- best system: S5h_encdec_hier_houlsby; decoder v2b; chunk sweep winner 4 (trained with 4)
- new runs screened: ['S5h8_encdec_hier8_houlsby', 'S5c_encdec_hier_houlsby_headtail']; went to the beam stage: S5h8_encdec_hier8_houlsby
- winner: NEW: S5h8_encdec_hier8_houlsby (beam 4) (primary metric (BERTScore) beats the reference by the margin); trimming not adopted; final file: NEW: S5h8_encdec_hier8_houlsby (beam 4) (test outputs generated in the beam stage)

```
                             run         arch target_mode  in_tokens    sel              vs_old_greedy  rouge1  train_min
       S5h8_encdec_hier8_houlsby encdec_hier8       trunc       4096 0.8478 +0.0031 [+0.0015, +0.0046]  0.4780       99.8
S5c_encdec_hier_houlsby_headtail  encdec_hier    headtail       2048 0.8442 -0.0004 [-0.0017, +0.0008]  0.4587       62.9
```

```
                              candidate                   sel_diff                rouge1_diff  est_test_min
NEW: S5h8_encdec_hier8_houlsby (beam 4) +0.0031 [+0.0017, +0.0045] +0.0148 [+0.0047, +0.0249]           0.2
   4 chunks, beam 4, length penalty 1.5 -0.0002 [-0.0004, +0.0001] +0.0111 [+0.0068, +0.0157]          12.2
```

BiLSTM: not run in this notebook (the architecture rule stays unmet unless Part 6 is run with only S7 and the BiLSTM dataset attached).
Validation numbers only; the official test scores come from the course evaluator.