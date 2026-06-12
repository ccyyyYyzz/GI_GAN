# Phase 50 Second-Wave Plan

Prepare only. Do not run until the user explicitly approves.

| Candidate | Expected Runtime | Scientific Value | Colab Needed | Placement | Risk |
|---|---:|---|---|---|---|
| Rad-5 train_no_gate_no_final_audit | overnight | Tests combined dependence on P_N and final Pi_y after isolated ablations | yes | supplement first | May muddy isolated causal claims |
| Scr-5 train_no_gate_no_final_audit | overnight | Tests whether Scr-5 collapse is audit-only or combined-circuit dependent | yes | supplement first | Expensive and may duplicate Session 05 if it collapses |
| Rad sampling scaling 2.5%, 7.5%, 15%, 20% | multi-session overnight | Strengthens empirical limit curve for Rademacher | yes | main if clean | Adds many runs before mechanism is settled |
| Scr sampling scaling 2.5%, 7.5%, 15%, 20% | multi-session overnight | Tests anchor-assisted scaling vs Rad | yes | main if clean | Requires exact config parity and careful labeling |
| Stronger CS-TV/FISTA-TV/ADMM-TV subset baseline | eval/short | Gives a more conventional inverse-problem comparison | maybe | main or supplement | Hyperparameter tuning could become a side project |
| Mixed Rad/Scr model | overnight | Tests whether one learned prior can bridge operator families | yes | supplement | Could distract from operator-centered mechanism |

Recommendation: do not start Phase 50 immediately. First import and inspect Phase 48/49 Session 01 plus at least one Rad and one Scr train-time ablation.
