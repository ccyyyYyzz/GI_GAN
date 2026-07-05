# Innovation Points — Autonomous Attempt Log (2026-07-04)

Each of the 13 innovation points from `INNOVATION_POINTS.md` was attempted. Status legend:
**DONE** = concrete result/artifact produced; **PROPOSED** = paper-ready design + proposition, needs an
external baseline or a run to become a result. The honesty red line held throughout (the measurement is
never claimed to certify invented content).

| IP | Title | Status | Output |
|----|-------|--------|--------|
| **IP-1** | Feasible-wrong twin (constructive witness) | **DONE (real run)** | `experiments_feasible_wrong.py` → `feasible_wrong_fusion_operator.json`: 40 pairs on the **64×64 STL10 m=205 fusion operator**, RelMeasErr median **9.9×10⁻¹⁴** (max 5.6×10⁻¹³), PSNR(u,xᵢ) median **19.5 dB**. Folded into draft §3.2; is Figure 1(a). **Resolves the dual-operator caveat** — converse and dial now on one operator. |
| **IP-2** | Exact per-mode certificate `λ/(λ+σ²)` vs bounds | **DONE (established)** | The single uncontested delta vs the #1 threat (Iagaru 2026 gives only worst-case bounds). Draft §4 + `NOVELTY_THREAT_DIFFERENTIATION.md`. Float64 modal contraction ~1e-10/1e-12. |
| **IP-3** | GT-free reconstructor-agnostic quality/accountability separation | **DONE (established)** | Draft §5; audit 18/18 rows, RelMeasErr −3..4 orders at |ΔPSNR|≤0.039 dB. Figure 1(b). |
| **IP-4** | The governed dial, `A x̂_B = y` ∀B | **DONE (the method)** | Draft §6–§7; locked LPIPS −32.6%, 8/8 gate, RelMeasErr 3.6e-7. Figure 1(c). Entirely clear space vs the assessment-only literature. |
| **IP-5** | The inversion catalogue | **DONE** | `innovation_attempts/IP-05.md`: 10-row table (each range/null paper's "reconstruct by X" vs "we invert to certify/bound/meter by Y") + Proposition collapsing it to `A P₀=0`. |
| **IP-6** | Cross-field identifiability unification (ceiling-raiser) | **DONE** | `innovation_attempts/IP-06.md`: subsection reframing `P₀` as the non-identified / observational-equivalence set (Koopmans–Reiersøl 1950, Rothenberg, Manski, Landau) + Prop 6.1/Cor 6.2 + GI-native identifiability-panel design. |
| **IP-7** | Impossibility trigger: single-operator non-identifiability | **DONE** | `innovation_attempts/IP-07.md`: Proposition (null coords non-identifiable from y alone, any reconstructor) + when identifiability is restored (multi-operator / equivariance / explicit prior) + active-sensing designs. |
| **IP-8** | Two guarantees: exact measurement cert ⊕ conformal task risk | **PROPOSED** | `innovation_attempts/IP-08.md`: Two-Ledgers proposition + CRC selection rule wrapping the frozen B-dial + "flat residual vs rising task-risk" figure + worked recipe. Task ledger = buildable design. |
| **IP-9** | Re-axed perception–distortion plane (accountability = 3rd axis) | **DONE** | `innovation_attempts/IP-09.md`: Proposition `dV/dB=0` on the measurement-safe manifold (from `A P₀=0`) + precise (D,P,V) figure spec over the 21-pt B-sweep. |
| **IP-10** | LPIPS-up / certificate-flat demonstration | **DONE (via Fig 1c)** | Realized by Figure 1(c): LPIPS ladder descends while `A x̂_B=y` holds ∀B (RelMeasErr flat ~3.6e-7). Dedicated 2-panel is a quick follow-up. |
| **IP-11** | GI-native demonstrations | **PROPOSED** | `innovation_attempts/IP-11.md`: shared-bucket GI experiment (correlation-GI / CS-GI / GIDL on identical y) reusing the repo's exact projector + certificate; a GI feasible-wrong proposition. Needs GI baselines. |
| **IP-12** | Live consistency≠correctness on strong solvers | **PROPOSED** | `innovation_attempts/IP-12.md`: transfer-of-the-converse proposition (every exact-range solver output admits a ~1e-15 twin) + PoC on DDNM/DPS/DDRM/MCG. Needs external solver code. |
| **IP-13** | Two-knob null-space governance (structure vs detail) | **DONE (design)** | `innovation_attempts/IP-13.md`: `x̂ = x₀ + P₀(a·d_A + b·d_G)` with per-knob null-energy receipts, separately audited; ablation design over existing fusion assets. |

## Summary
- **Real new experiment:** IP-1 (feasible-wrong on the fusion operator) — also closes the operator-reconciliation (D) gap.
- **New figure:** Figure 1 — the impossibility-first "cannot / can / therefore" three-panel (`METHOD_FIG1.png/pdf`), panel (a) uses the real IP-1 witness.
- **8 paper-ready artifacts:** `innovation_attempts/IP-05..IP-13.md` (subsections, tables, propositions, experiment designs), each grounded in specific reading cards.
- **Highest-leverage next steps:** IP-6 identifiability subsection into the Intro/Related (ceiling-raiser); IP-8/IP-11/IP-12 are the strongest new experiments to actually run before submission.
