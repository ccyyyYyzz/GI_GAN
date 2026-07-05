# _unrelated_fresnel_zone_plate

This folder holds a **Fresnel zone-plate (菲涅耳波带片) student manuscript** that is unrelated to the ghost-imaging science in this repository. It was moved here from the repo root during the 2026-07 reorganisation to keep the root clean.

## Contents

| File | Description |
|---|---|
| `main.tex` / `main.pdf` | Primary manuscript: "菲涅耳波带片的目标光场编码设计：稳妥基准方案与目标光场编码创新方案" (Fresnel zone-plate target-light-field encoding design — baseline and innovation scheme) |
| `main_text.txt` / `paper_main_text.txt` | Plain-text extracts of the main manuscript |
| `proposal_scheme1_baseline.tex` / `.pdf` | Scheme 1 — baseline (open half-wave zone, contributions arrive in phase at focus) |
| `proposal_scheme2B_radial_binary.tex` / `.pdf` | Scheme 2B — radial binary encoding variant |
| `队友版_方案选择.tex` / `.pdf` | Team-member version of the scheme-selection writeup (Chinese) |

## Why it is here

The manuscript was committed to the ghost-imaging repo alongside other working files. It is **not** part of the ghost-imaging science (no connection to y = Ax + eps, null-space diagnostics, GAN/VQGAN priors, or FCC row-null analysis). It lives in this quarantine folder solely to preserve the file history; no content was deleted.

## Note: figures/assemble_mentor_proposals.py (deleted 2026-07-06)

The one-off generator `figures/assemble_mentor_proposals.py` — which hardcoded
`DEST = r"E:\ns_mc_gan_gi_code_fcc_phase1"` and would **re-emit**
`proposal_scheme1_baseline.tex` / `proposal_scheme2B_radial_binary.tex` to the
repo root if run — was **deleted from git on 2026-07-06** to remove that
run-hazard. It is recoverable from git history if ever needed.

## Status

Quarantined. No further edits expected. Ghost-imaging contributors can ignore this folder entirely.
