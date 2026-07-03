# Stage 4 — Manuscript and Mechanism Construction (Phases 18-45)

**Research question:** Convert the accumulated null-space/audit evidence into publication-ready figures, tables, and a LaTeX manuscript.

**Theory:** The manuscript develops the range-null view of undersampled ghost imaging (y = Ax + eps, m << n). P_R = A^dagger A locks only the row-space component; P_0 = I - A^dagger A is invisible to A. A plug-in test-time audit B_lambda(v) = v - A^T(AA^T + lambda I)^{-1}(Av - y) drives RelMeasErr down by the singular-mode factor lambda/(lambda + sigma_i^2) without retraining. Mechanism figures visualize the geometry (the affine feasible plane, the null-space shift, the fusion dial). Figure scripts load real certificate data from cert_package_20260612 and do NOT invent numbers.

---

## Manuscript

| File | Role |
|---|---|
| `paper/main.tex` | Main IEEE-TCI LaTeX source. Title: "Measurement Auditing for Learned Ghost Imaging: Certificates, Limits, and Prior-Supplied Content". Sections: Abstract, Intro, Problem Setup/Range-Null Decomposition, Test-Time Audit, Boundary of the Certificate, GAN Case Study, Conclusions. |
| `paper/materials_inventory.md` | Exhaustive inventory of every numerical claim in main.tex, with the exact CSV/NPZ source file for each number. Generated 2026-06-15. Do NOT draft prose from memory; use this file to locate the source. |

**Compilation** (from repo root, requires LaTeX + IEEEtran):

```
cd paper
pdflatex main.tex
pdflatex main.tex   # second pass for cross-refs
```

Figures must be pre-generated (see below) before compilation; `paper/main.tex` references `figures/figure1_feasible_geometry.pdf` and other PDFs via `\includegraphics`.

---

## Mechanism Figures

### Core data-flow flowchart (`core_mechanism_figure.py`)

**Script:** `core_mechanism_figure.py` (repo root)  
**Outputs:** `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/CORE_MECHANISM_FIGURE.{png,pdf,svg}` (all three confirmed present)  
**What it draws:** A concise horizontal data-flow for the measurement-consistent VQGAN detail-fusion path. Boxes show: GI bucket data -> LMMSE anchor x0 (row-space fixed) -> null-space injection via dial B -> x_hat_B with A x_hat_B = y exactly for every B. No real image data is needed at render time; purely schematic with matplotlib.

**Reproduce:**

```bash
# From repo root, py311 env
python core_mechanism_figure.py
```

### Pseudo-3D geometry diagram (`method_diagram_3d.py`)

**Script:** `method_diagram_3d.py` (repo root)  
**Outputs:** `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/METHOD_DIAGRAM_3D.{png,pdf,svg}` (all three confirmed present)  
**What it draws:** Oblique pseudo-3D rendering of the affine feasible plane {x : Ax = y} = x0 + null(A). Key points on the plane: LMMSE anchor x0, VQAE structure point x_A, VQGAN detail point x_G, true scene x* (unknown null-space location). The fusion dial x_hat_B = x0 + P0(d_A + B(d_G - d_A)) slides along the in-plane x_A -> x_G segment. A perception-distortion gradient is shown under the dial. Stylized chips represent each image type. Oblique projection parameters: KY=0.52, JY=0.34, KZ=-0.14.

**Reproduce:**

```bash
python method_diagram_3d.py
```

---

## Paper Figures (`paper/figures/`)

All figure scripts write output into `paper/figures/` or subdirectories thereof. Run from repo root.

| Script | Output(s) | What it draws | External data needed |
|---|---|---|---|
| `paper/figures/make_figure1_feasible_geometry.py` | `paper/figures/figure1_feasible_geometry.{pdf,png}` | Figure 1 in main.tex: real bucket readout y[1789] (m=205 measurements), three-layer image cards showing the shared row-space skeleton (blue) and differing null-space components (orange) for the Rad-5 pair i=1789 (car truth) and j=935 (horse donor). Reported numbers: RelMeasErr(x_i, y_i) = 5.36e-3, RelMeasErr(u_ij, y_i) = 2.93e-15. | `E:\ns_mc_gan_gi\results\cert_package_20260612\cache\main_rad5.npz`, `A_rad5.npy`, `split_eval_indices_stl10_test.npy` |
| `paper/figures/make_paper_figures.py` | `paper/figures/pipeline_range_null_schematic.{pdf,png}`, `theorem2_psnr_bound.{pdf,png}`, others | Pipeline range-null schematic; theorem 2 PSNR bound panel; additional paper figures drawing from T4/T5 tables and the main Rad-5 cache. | `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T4_pairs.csv`, `T5_rho.csv`, cache NPZ/NPY files |
| `paper/figures/make_feasible_wrong_candidate_pool.py` | `paper/figures/feasible_wrong_candidate_pool/` (120 candidates, `candidates.csv`, 4 sheets of 30) | Generates the full candidate pool of feasible-wrong images for i=1789 (car target): 120 donors across STL-10 classes, scored by RelMeasErr_u_vs_yi. | cert_package cache |
| `paper/figures/make_feasible_wrong_gallery.py` | `paper/figures/feasible_wrong_gallery_quality_matched/` | Quality-matched gallery with reference donor j=935 (horse) and quality donors [1902, 53, 602, 1050], showing measurement-consistent but semantically wrong images. | cert_package cache |
| `paper/figures/select_feasible_wrong_images.py` | `paper/figures/feasible_wrong_selected/` | Copies 5 selected wrong images (ranks 011/118/057/106/031, covering cat/dog/bird/airplane/ship) and the car truth + horse reference into a clean selection directory for paper inclusion. | `feasible_wrong_candidate_pool/` must exist first |

**Confirmed rendered outputs in `paper/figures/`:**
- `figure1_feasible_geometry.pdf`, `figure1_feasible_geometry.png`
- `pipeline_range_null_schematic.pdf`, `pipeline_range_null_schematic.png`
- `feasible_hallucination_pair.pdf`, `feasible_hallucination_pair.png`
- `theorem2_psnr_bound.pdf`, `theorem2_psnr_bound.png`
- Subdirs: `feasible_wrong_candidate_pool/`, `feasible_wrong_gallery/`, `feasible_wrong_gallery_quality_matched/`, `feasible_wrong_selected/`, `feasible_wrong_final_selection/`, `figure1_assets/`

---

## Phase 12 Report Generator (`src/make_phase12_report.py`)

**Script:** `src/make_phase12_report.py`  
**Role:** Generates the phase-12 experiment registry report (Markdown) that tracks all runs by method_id, dataset, sampling_ratio, PSNR, SSIM, status, and preferred_for_paper flag. Reads from `outputs/phase12/final_result_registry.csv` (via `src/phase12_common.py`). Produces section tables for: preferred rows, main STL-10 tables at 10%/5%, simple domains, reproducibility, claims, DC row control, minimal baselines, and threshold status. Best preferred PSNR/SSIM are computed dynamically from the registry.

**Reproduce** (from repo root):

```bash
python -m src.make_phase12_report
```

This is a reporting utility, not a training step. It reads existing CSV results and emits Markdown summaries to `outputs/phase12/`.

---

## Key Numbers (from `paper/materials_inventory.md`)

All numbers below are sourced from `paper/materials_inventory.md` (generated 2026-06-15) which itself is sourced from cert_package_20260612 and the phase-30 submission package.

**Main reconstruction table (STL-10):**

| Regime | PSNR | SSIM | BP PSNR | Delta PSNR | RelMeasErr pre | RelMeasErr post |
|---|---:|---:|---:|---:|---:|---:|
| Rad-5 | 22.316 | 0.635 | 7.297 | 15.019 | 3.775e-05 | 2.149e-09 |
| Scr-5 | 22.271 | 0.632 | 14.310 | 7.961 | 5.509e-03 | 5.504e-06 |
| Rad-10 | 24.781 | 0.747 | 7.756 | 17.025 | 5.872e-05 | 7.612e-09 |
| Scr-10 | 24.730 | 0.746 | 14.533 | 10.197 | 5.712e-03 | 5.707e-06 |

**Feasible-wrong boundary (T4_pairs.csv):** 16/16 cross-class pairs satisfy the wrong measurement record to ~2e-15 (Rad-5) or exactly 0 (Scr-5), while truth's own RelMeasErr is 3e-3 to 8e-3. The pair i=1789 (car), j=935 (horse) is the canonical Figure 1 example.

**Modal contraction (T3):** float64 k=1 audit follows lambda/(lambda+sigma^2) contraction; pipeline float32 hits solver floor for repeated k.

**Range-share formula:** DeltaPSNR_max = -10 log10(1 - s), where s = rho (fraction of energy in range of A). Rad-5 rho ~ 0.050 -> ceiling 6.67 dB; Scr-5 rho ~ 0.804 -> ceiling 14.31 dB.

---

## What This Stage Does NOT Do

- Stage 4 does not establish the null-space barrier (consistency != correctness). That is Stage 5 (`../05_range_null_barrier/`).
- Stage 4 does not run GAN training. The GAN case study is Stage 6 (`../06_gauge_gan_rad5/`).
- Stage 4 does not include VQGAN/FCC work. That is Stage 8 (`../08_vqgan_fcc/`).
- `paper/materials_inventory.md` explicitly notes it is an inventory only and does not draft manuscript prose.

---

## Reproduction Order

```
# 1. Pre-generate feasible-wrong candidate pool (needs cert_package cache on E:\ns_mc_gan_gi)
python paper/figures/make_feasible_wrong_candidate_pool.py

# 2. Select from pool
python paper/figures/select_feasible_wrong_images.py

# 3. Generate Figure 1 and other paper figures
python paper/figures/make_figure1_feasible_geometry.py
python paper/figures/make_feasible_wrong_gallery.py
python paper/figures/make_paper_figures.py

# 4. Generate mechanism figures (VQGAN detail-fusion paper assets)
python core_mechanism_figure.py
python method_diagram_3d.py

# 5. Compile LaTeX
cd paper && pdflatex main.tex && pdflatex main.tex
```

Steps 1-3 require `E:\ns_mc_gan_gi\results\cert_package_20260612\` to be accessible. Steps 4-5 are self-contained. The py311 env (`E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311`) is required throughout.

---

## Cross-References

- Stage 3 (baselines + audit): `../03_baselines_audit/` — produces cert_package_20260612 tables consumed here
- Stage 5 (range-null barrier): `../05_range_null_barrier/` — extends the feasible-wrong gallery to the boundary theorem
- Stage 6 (GAN case study): `../06_gauge_gan_rad5/` — the GAN branch referenced in main.tex Section V
- Stage 8 (VQGAN/FCC): `../08_vqgan_fcc/` — the separate VQGAN detail-fusion draft; its mechanism figures (`core_mechanism_figure.py`, `method_diagram_3d.py`) live at repo root and write into `outputs/compatibility/...`; do NOT auto-merge into the conservative IEEE-TCI main claim
