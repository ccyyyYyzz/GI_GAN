# paper/ — manuscripts and supporting material

This directory holds several manuscripts that share the ghost-imaging
measurement-accountability research program. They **target different venues and
make different core claims, and must not be merged** (see
[`HANDOFF/06_PAPERS_AND_CLAIMS.md`](../HANDOFF/06_PAPERS_AND_CLAIMS.md) for the
authoritative per-paper claims, numbers, and locked-source cautions).

## Manuscripts (newest first)

| Stem | Title (short) | Target venue | Date | Notes |
|---|---|---|---|---|
| **`OPTICS_DRAFT`** `.md`/`.tex` | *A Ground-Truth-Free Measurement-Consistency Audit for Learned Ghost and Single-Pixel Imaging at Low Sampling Ratios* | **Optics Express** (Chen-group style) | 2026-07-06 | **Current active submission target** and the **default build**. Content-complete and review-passed (GPT review Rounds 6–11); pending only author/affiliation/funding metadata and the official Optica LaTeX template. |
| `UNIFIED_PAPER_DRAFT` `.md`/`.tex` | *Certify What You Measure, Govern What You Cannot: A Range–Null Account of Accountability and Governed Detail Across Undersampled Imaging Operators* | broad / unified | 2026-07-05 | Wider two-ledger narrative spanning multiple operators (GI, MRI, CASSI). |
| `main` `.tex` | *Measurement Auditing for Learned Ghost Imaging: Certificates, Limits, and Prior-Supplied Content* | IEEE Trans. Computational Imaging | 2026-07-03 | The earlier conservative "main line" paper. See HANDOFF/06 §1. |

A fourth, independent **VQGAN positive-sibling draft** lives outside this
directory at
`outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md`
(HANDOFF/06 §2) — *Measurement-Consistent VQGAN Detail Fusion for Low-Rate Ghost
Imaging*. Do not auto-merge it into the conservative claim.

> Note: `HANDOFF/06_PAPERS_AND_CLAIMS.md` was written 2026-07-03 and documents
> `main.tex` + the VQGAN sibling; the `UNIFIED_PAPER_DRAFT` and `OPTICS_DRAFT`
> drafts postdate it. This README is the current manuscript map.

## Building

```
python paper/build_variant.py [STEM]     # STEM defaults to OPTICS_DRAFT
```

Builds run xelatex on `<STEM>.tex`. Preview/layout-check PNGs are regenerable and
are **not** tracked (see `paper/tmp/`, git-ignored).

## Supporting material

- `_draft/` — sectional decomposition of the unified draft (a drafting aid; the
  canonical source is the monolithic `.md` files, not these sections).
- `figures/` — figure assets and their `make_*.py` generation scripts.
- `notes/` — **archived drafting-process notes** (audits, brainstorms, exploration
  logs). Not part of any build; kept for provenance. See `notes/README.md`.
- `materials_inventory.md` — evidence ledger mapping paper claims to the external
  result tables under `E:\ns_mc_gan_gi\results\...` (read-only, outside this repo).
- Reference material kept at root: `RELATED_WORK.md`, `CHEN_GROUP_STYLE_GUIDE.md`,
  `GI99_reading_cards.md`, `INNOVATION_POINTS.md`, `HQ_GI_INNOVATION_POINTS.md`,
  `NOVELTY_THREAT_DIFFERENTIATION.md`.
- `innovation_attempts/` (IP-*), `hq_innovation_attempts/` (HQ-*) — logged
  innovation-attempt records cross-referenced by the reference material above.
