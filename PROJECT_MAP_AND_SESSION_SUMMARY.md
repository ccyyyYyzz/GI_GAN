# GAN / FCC Project Map & Session Summary

_Organization snapshot generated 2026-06-29. Covers (A) what is on GitHub, (B) the local
workspaces and project lines, and (C) what this Claude Code session produced._

---

## A. GitHub repositories (what is actually published)

| Repo | Default branch | What is really there | Other branches |
|---|---|---|---|
| **`ccyyyYyzz/GI_GAN`** | `main` (7 commits) | **Only early Jupyter notebooks** — `GI_GAN_12.ipynb`, `GI_GAN_13.ipynb`, `GI_GAN_4_report.ipynb`, `GI_GAN_for_report.ipynb`, `GI_GAN_learning_code_implementation.ipynb`, `Untitled14.ipynb`, `README.md`. This is the original learning/implementation, **not** the detail-fusion / FCC research code. | `codex/vqgan-multiseed-handoff` (at commit `3a5c866`, 2026-06-26), `codex/core-experiments` |
| **`ccyyyYyzz/GAN_FCC`** | `main` (1 commit) | **Near-empty handoff shell** — only `README.md` + open **issue #1** "Handoff: current ghost-imaging GAN/FCC work state". No code/results pushed. | (handoff names branch `pub-colab-runner`) |

**Bottom line:** the substantive research (detail-fusion, FCC diagnostics, Rad-5 paper) is **local only**; GitHub holds old notebooks (GI_GAN) and a handoff stub (GAN_FCC).

---

## B. Local workspaces & project lines (three distinct lines)

| Line | Local workspace | Git remote / branch | Topic | Status |
|---|---|---|---|---|
| **A — VQGAN detail-fusion + FCC** | `E:\ns_mc_gan_gi_code_fcc_phase1` (this dir) | `github → GI_GAN`, `backup`; branch `codex/vqgan-multiseed-handoff` | Measurement-consistent VQGAN null-space detail fusion; FCC row-null compatibility diagnostics | Fusion paper assembled (`BALANCED_VQGAN_FUSION_CONFIRMED`); FCC diagnostics done this session |
| **B — Rad-5 B/C gauge-GAN paper** | `E:\ns_mc_gan_gi_code` **(READ-ONLY)** | `backup` only; branch `pub-colab-runner` → maps to **GAN_FCC** repo | Soft-anchor `B_λy` + gauge discriminator; `paper/main.tex` | ~42 uncommitted files; nothing pushed to GAN_FCC. Source of truth = GAN_FCC issue #1 |
| **C — ZIFB iodine-passivation (COMSOL)** | `E:\zifb_final_9129_luck` | (separate) | Pure-simulation battery mechanism paper | Separate project — **not** on these GitHub repos, not part of this session |

> Hard rule in effect: **`E:\ns_mc_gan_gi_code` is read-only** — view/copy/run only; all new artifacts go to authorized working dirs (`..._fcc_phase1` / `..._phase2`).

---

## C. This session's work (2026-06-27) — FCC diagnostics on Line A

Goal of the session: per the user's `FCC_THEORY_AND_PROMPT_BUNDLE`, **diagnose** whether the
null-space content is learnably compatible with the row-space structure **beyond deployable
nuisance baselines** — explicitly NOT to improve reconstruction.

### C.1 Row-null FCC canary (clean re-implementation)
- **Result: `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`.** Real-pair retrieval is perfect (DEV Recall@1 = 1.00, 32× random) but it is fully explained by deployable nuisance statistics of `u = r+n` (TV, range, high-freq); FCC does not exceed the deployable baseline on nuisance-balanced negatives. `row-only = null-only = 0.5`.
- Independently **replicates** the prior Rad-5/96 + Phase-1.1 finding on a fresh 64×64 Rademacher-5% operator, now with corrected *deployable* (not oracle) controls.
- Code: `fcc_diagnostic_canary.py`, `src/fcc_canary.py`, `tests/test_fcc_canary.py`, `configs/compatibility/fcc_diagnostic_canary64{,_smoke}.yaml`.
- Outputs: `outputs/compatibility/fcc_diagnostic_canary64/` · package `FCC_DIAGNOSTIC_CANARY64_RESULT_PACKAGE.zip` (SHA256 `230FCBA8…`).
- Adversarial 4-agent verification: **PASS_WITH_NITS** (no blockers).

### C.2 Structure-Detail FCC (on the confirmed VQAE/VQGAN fusion data)
- Pairing `s = x_A` (VQAE structure), `d = P0(x_G − x_A) = d_G − d_A` (GAN null-space detail). Feasibility exact (`d ∈ null A`).
- **Result: `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`, 3/3 fusion seeds** — but **"warmer"** than row-null: negatives balance far better (SMD ~0.55 vs 1.48; detail is only ~12% energy), and the deployable *joint* classifier (0.97 bal AUC) beats pure-naturalness sum-image (~0.87) by **+0.09–0.13** → real structure-detail correlation beyond naturalness, yet still captured by a plain logistic and not exceeded by the learned critic (FCC bal AUC ~0.63).
- Code: `structure_detail_fcc.py`, `configs/compatibility/structure_detail_fcc.yaml` (reuses the FCC pipeline).
- Outputs: `outputs/compatibility/structure_detail_fcc/` (`STRUCTURE_DETAIL_FCC_REPORT.md` + per-seed reports) · package `STRUCTURE_DETAIL_FCC_RESULT_PACKAGE.zip` (SHA256 `60345B33…`).
- Adversarial 3-agent verification: **PASS**.

### C.3 Core mechanism figure — **set aside per user ("不画图了")**
- `core_mechanism_figure.py` → `outputs/.../detail_fusion_paper/CORE_MECHANISM_FIGURE.{png,pdf,svg}`. Concise version exists but the user has paused figure work. Kept on disk; not wired into the paper.

### C.4 Commit / sync status of this session
- Committed **locally** on `codex/vqgan-multiseed-handoff`: `5db98ff` (FCC diagnostic scripts), `95c4cf2` (concise mechanism figure + rate tooling), `259b92a` (cross-rate + pseudo-3D figure).
- **Not pushed:** GitHub `GI_GAN/codex/vqgan-multiseed-handoff` is at `3a5c866` — **4 commits behind** local. Result packages, caches, checkpoints are intentionally local-only (large).
- Still untracked: the 3 FCC `configs/compatibility/*.yaml` and other working files (manuscript drafts `main.tex`/`proposal_scheme*` etc. appear to be later, separate user work).

---

## D. Suggested next organizing actions (nothing done automatically)

1. **Decide GI_GAN sync:** push local `codex/vqgan-multiseed-handoff` (FCC + fusion code) to GitHub `GI_GAN` so the handoff branch matches local (currently 4 behind). Commit the 3 untracked FCC configs first.
2. **Decide GAN_FCC content:** per issue #1, copy the Rad-5 paper-only scope from the read-only `E:\ns_mc_gan_gi_code` into an authorized dir, review, and push to `GAN_FCC` on a `handoff/...` branch (never push logs/checkpoints to the public repo).
3. **FCC diagnostics conclusion (both lines):** `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`. Honest write-up option: report the structure-detail "+0.09–0.13 joint-beyond-naturalness" margin as a positive observation, with the open problem (balanced negatives that neutralise deployable baselines) as the stated limitation.
4. **Figure:** on hold.

_All paths above are under the authorized working dir `E:\ns_mc_gan_gi_code_fcc_phase1`._
