# Repo guide (read before reorganizing)

Ghost-imaging + VQGAN measurement-accountability research program (the range–null
"two-ledger" line). This file captures the load-bearing structural facts so the
repo isn't accidentally broken by a well-meaning cleanup.

## ⚠️ Architecture: flat-root imports — do NOT move/rename root `.py` files

~68 Python modules live **flat at the repo root by deliberate design**. Scripts
are always run with `cwd = repo root`, and they import each other and the package
by **bare module name** (e.g. `import gan_high_quality_gi as hq`,
`import phase1_3r_recovery_and_relock as p13r`, `from src.X import ...`).

- **34 root scripts depend on sibling root scripts via bare imports.** Moving even
  one into a subdirectory breaks ~50 import statements across the codebase with no
  `from parent.module` fallback.
- The phase pipeline (`phase1_1 → phase1_2 → phase1_3 → phase1_3r → phase1_4a →
  phase1_4ir → phase1_4v4a`) and the `*_v2` / `*_locked` score variants are
  **intentional** locked-evaluation checkpoints, not clutter — do not delete.
- If you must improve discoverability, add docs — **do not relocate code.**

## Where to start

- `HANDOFF/00_START_HERE.md` — authoritative entry point, stage map (0–8), repro.
- `HANDOFF/03_CODE_MAP.md` — what each root module is.
- `HANDOFF/06_PAPERS_AND_CLAIMS.md` + `paper/README.md` — the manuscripts (distinct
  venues, **must not be merged**).
- `HANDOFF/07_RED_LINES_AND_WORKING_RULES.md` — negative results not to revisit.
- `research_lines/NN/INDEX.md` — per-stage pointers (they index code, don't hold it).

## Intentional non-code / quarantine dirs (leave as-is)

`_unrelated_fresnel_zone_plate/` (quarantined non-GI manuscript),
`_inbox/` (FCC theory + verify workflows, read by `src/fcc_canary.py`),
`locked_bundle/`, `artifacts/p0_manifest.json` (projector provenance),
`outputs/` & `results/` (committed evidence bundles referenced by HANDOFF),
`paper/notes/` (archived drafting notes). Generated LaTeX previews under
`paper/tmp/` are git-ignored and regenerable.

## Data / env

External data + the canonical `py311` env live under
`E:\GAN_FCC_WORK\data_warehouse\` (after a mid-2026 E-drive consolidation);
`E:\ns_mc_gan_gi` no longer exists on disk. See
`HANDOFF/04_REPRODUCIBILITY_GUIDE.md`.
