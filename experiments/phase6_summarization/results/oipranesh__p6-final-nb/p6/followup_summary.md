# Follow-up: the decoding fix (validation numbers; not official test scores)

- decoder candidate: v2b (byte-level decoder attached); the plain decode of the earlier parts joined every pair of tokens with a space (the tokenizer file has no decoder); decoder used for the final file: v2b; the fixed text is used for the final file
- best system: S5h_encdec_hier_houlsby; final decoding 'beam 4'; final file: S5h_encdec_hier_houlsby | decoding 'beam 4'; consensus not adopted (-0.0167 [-0.0184, -0.0150])

## Old versus new text, the same generated ids, 200 validation documents

```
                                   text  bertscore_f1  rouge1_f  rougeL_f  fact_coverage  repetition3  unsupported  len_ratio  pred_words
OLD (plain decode of the earlier parts)        0.8396    0.4862    0.2470         0.3015       0.0166       0.1492     0.7720     421.735
 NEW (v2b: byte-level decoder attached)        0.8499    0.4914    0.2504         0.3092       0.0163       0.2421     0.6716     367.335
```

paired difference of BERTScore F1 (new - old): +0.0103 [+0.0098, +0.0108] | ROUGE-1: +0.0053 [+0.0042, +0.0063]

## Requirement status (from Part 6; this notebook changes no experiment)

**NOT COMPLIANT: architecture not covered: bilstm**

Validation numbers of the OTHER runs in the ledger were scored on old-style text and are comparable with each other, not with the NEW row above; re-generating every run was not done.