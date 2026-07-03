# 08 Non-GI Content Quarantine

This document records two distinct bodies of non-ghost-imaging content that ended
up in this repo and the actions taken to isolate or remove them.

---

## 1. Fresnel Zone Plate Student Manuscript (quarantined)

### What it is

A Chinese-language undergraduate physics-competition manuscript on Fresnel zone
plates, wholly unrelated to ghost imaging.  The LaTeX title is
"菲涅耳波带片的目标光场编码设计：稳妥基准方案与目标光场编码创新方案".

### Location after quarantine

All files were **git-renamed** (not deleted) from repo root into:

```
_unrelated_fresnel_zone_plate/
  main.tex                        # Chinese ctexart source (FZP)
  main.pdf                        # Compiled PDF (25 pp, A4, xdvipdfmx)
  main_text.txt                   # Plain-text extract of main.pdf (mixed: see note below)
  paper_main_text.txt             # Plain-text extract of paper/main.pdf (GI paper — misplaced)
  proposal_scheme1_baseline.tex   # Scheme-1 English proposal (auto-generated)
  proposal_scheme1_baseline.pdf
  proposal_scheme2B_radial_binary.tex  # Scheme-2B English proposal (auto-generated)
  proposal_scheme2B_radial_binary.pdf
  队友版_方案选择.tex              # Teammate variant, Chinese ctexart
  队友版_方案选择.pdf
```

Git status at time of writing shows these as staged renames (`R`), confirming
they were originally at repo root and moved inward.

**Note on `paper_main_text.txt`:** despite residing in `_unrelated_fresnel_zone_plate/`,
its content is a plain-text extract of `paper/main.pdf` (the GI measurement-auditing
paper, "Measurement Auditing for Learned Ghost Imaging…").  It was misplaced during
a text-extraction step.  It is not FZP content; treat it as a stale reference copy.

### The `figures/assemble_mentor_proposals.py` hazard

`figures/assemble_mentor_proposals.py` is the script that produced
`proposal_scheme1_baseline.tex` and `proposal_scheme2B_radial_binary.tex`.
It hardcodes two absolute paths:

| Variable | Hardcoded value |
|---|---|
| `OUT_JSON` | `D:\tmp\claude\E--ns-mc-gan-gi-code-fcc-phase1\649423c0-...\tasks\wof8tkr6b.output` |
| `DEST` | `E:\ns_mc_gan_gi_code_fcc_phase1` |

**If this script is run**, it reads the session-scoped JSON from `D:\tmp\...`
and writes `proposal_scheme1_baseline.tex` and `proposal_scheme2B_radial_binary.tex`
directly to repo root (`E:\ns_mc_gan_gi_code_fcc_phase1\`), polluting root again.
The `OUT_JSON` path references a now-stale session scratchpad; the script will
fail with a `FileNotFoundError` on that path unless the session file is restored.
Do not run this script.  It should not be executed as part of any GI workflow.

---

## 2. ZIFB Flow-Battery Content (deleted)

### What was here

During commit `9141784` ("Check in full working tree…"), the following
non-GI battery-paper artifacts were accidentally committed to this repo:

- `_manuscript_view/` — page-render PNGs and PDFs of the ZIFB manuscript
  (`main.pdf`, `main_scichina.pdf`, `main_scts.pdf`, `SI.pdf`,
  plus ~50 page-image PNGs: `SIpage-01` through `SIpage-10`,
  `scpage-*`, `sctpage-*`, `fig1check-02.png`, `Br_series_Vt.png`)
- `_inbox/REPLY_TO_GPT_REVISION_GUIDANCE_20260629.md` — a 45-line ZIFB
  revision-guidance memo

SHA-256 identity audit confirmed `_manuscript_view/main.pdf` and
`_manuscript_view/SI.pdf` were **byte-for-byte identical** to
`E:\zifb_final_9129_luck\manuscript\main.pdf` and `…\SI.pdf` respectively
(paper title: "Retained iodine is not blocking iodine: a dimensionless theory
of dissolution-limited passivation in the zinc–iodine flow battery positive
electrode"; 19 pp, Letter, LaTeX with hyperref / pdfTeX-1.40.26).

### Deletion

Commit `986bdfe` ("Remove leaked ZIFB manuscript files") deleted all 61 files
(60 binary assets + 1 markdown, totalling 2 087 line-deletions in the diff).
As of that commit `_manuscript_view/` no longer exists in the worktree.

The ZIFB project's canonical home is `E:\zifb_final_9129_luck`.
A full-tree backup including these renders is retained in
`GAN_FCC_WORK/data_warehouse` (13 GB fullcopy).

### Remaining reference in this repo

The audit that detected the misidentification is preserved at:

```
research_lines/00_program_overview/manuscript_audit/
  ROUND1_MANUSCRIPT_VIEW_IDENTITY_AUDIT.csv    # machine-readable hash/metadata table
  ROUND1_MANUSCRIPT_VIEW_IDENTITY_AUDIT_CN.md  # human-readable audit memo (Chinese)
```

The CSV records SHA-256, page count, PDF creator, and first-page text for six
files: the two `_manuscript_view/` renders (now deleted), their ZIFB originals,
the root FZP `main.pdf`, and `paper/main.pdf` (the actual GI paper).  These
audit records reference the incident and are kept as provenance; they do not
constitute ZIFB science in this repo.

### `_inbox/` remainder

After deletion, `_inbox/` retains only GI-related content:

```
_inbox/
  FCC_THEORY_AND_PROMPT_BUNDLE/
    FCC_THEORY_RANGE_NULL_COMPATIBILITY.md   # FCC range-null theory notes (GI)
    CLAUDE_CODE_FCC_DIAGNOSTIC_PROMPT.md     # FCC diagnostic prompt (GI)
  fcc_verify_workflow.js                     # FCC verification workflow (GI)
  sd_fcc_verify_workflow.js                  # SD FCC verification workflow (GI)
```

These files belong to the FCC/range-null ghost-imaging subline (research line 08)
and are legitimate repo content.

---

## 3. Summary

| Content | Action | Current state |
|---|---|---|
| FZP student manuscript (`main.tex`, proposals, 队友版) | git-renamed to `_unrelated_fresnel_zone_plate/` | Staged rename; files present in quarantine dir |
| `figures/assemble_mentor_proposals.py` | Left in place; flagged as DO-NOT-RUN | Writes proposal .tex to repo root if executed |
| ZIFB `_manuscript_view/` (60 binary files) | Deleted in commit `986bdfe` | Directory does not exist |
| ZIFB `_inbox/REPLY_TO_GPT_REVISION_GUIDANCE_20260629.md` | Deleted in commit `986bdfe` | File does not exist |
| ZIFB audit records in `research_lines/00_program_overview/manuscript_audit/` | Retained as provenance | Not ZIFB science; documents the identity-check incident |

None of this content is part of the ghost-imaging measurement-auditing science.
The authoritative GI manuscript is `paper/main.pdf` (12 pp, "Measurement Auditing
for Learned Ghost Imaging: Certificates, Limits, and Prior-Supplied Content").
