# Results status at the documented snapshot

This is a compact status sheet. The detailed numbers, seeds, split rules, and receipts remain in `HANDOFF/05_EXPERIMENTS_AND_EVIDENCE.md` and the result directories.

| Result family | Current interpretation | Publication use |
|---|---|---|
| Exact range/null identities | Algebraic tests pass | Core theory/method evidence |
| Feasible-wrong-image certificate | Same record can correspond to a wrong cross-class image under the declared construction | Identifiability limitation; keep the certificate and assumptions visible |
| Gauge-GAN | Gauge equalization removes the row shortcut in the reported stress test | Case study, not a universal GAN theorem |
| VQAE/VQGAN fusion | Trained VQAE/VQGAN outputs are fused at test time; locked result improves perceptual metrics under the declared split | Positive result with the locked-scorer and split restrictions |
| FCC canary | Diagnostic compatibility result | Supplementary / diagnostic unless the paper states the exact scope |
| CASSI two-ledger | Transfer test with the released operator/data | Supplementary transfer evidence |
| fastMRI two-ledger | Requires declared external files/weights | Supplementary when the data are restored |
| G2R posterior sampling | Negative or dormant in the recorded pilot | Do not present as a positive contribution |

The status sheet is deliberately conservative: a metric improvement does not remove the need to state the operator, split, data access, and locked-evaluation rules.

