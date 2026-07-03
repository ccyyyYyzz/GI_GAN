from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path(r"E:\ns_mc_gan_gi")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("/mnt/e/ns_mc_gan_gi")
CACHE = DATA_ROOT / "results" / "cert_package_20260612" / "cache"
OUT = Path(__file__).resolve().parent / "feasible_wrong_candidate_pool"

I_TARGET = 1789
REFERENCE_DONOR = 935
N_CANDIDATES = 120
PER_SHEET = 30
LABEL_NAMES = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",
    8: "ship",
    9: "truck",
}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def psnr_one(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((np.clip(pred, 0.0, 1.0) - target) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def save_gray(path: Path, flat: np.ndarray) -> None:
    arr = np.clip(flat.reshape(64, 64), 0.0, 1.0)
    plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def add_sheet(path: Path, rows: list[dict[str, object]], images: list[np.ndarray]) -> None:
    cols = 6
    total = len(rows)
    rows_n = int(np.ceil(total / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(12.0, 2.28 * rows_n), dpi=190)
    axes_arr = np.array(axes, dtype=object).reshape(rows_n, cols)
    for ax in axes_arr.flat:
        ax.axis("off")
    for ax, row, image in zip(axes_arr.flat, rows, images):
        ax.imshow(np.clip(image.reshape(64, 64), 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
        title = (
            f"rank {int(row['rank']):03d} | {row['donor_label']} j={row['donor_j']}\n"
            f"Rel={float(row['relmeaserr_uij_vs_yi']):.1e} "
            f"PSNR={float(row['psnr_uij_vs_xi_clipped']):.1f} "
            f"clip={float(row['clip_penalty_mean']):.1e}"
        )
        ax.set_title(title, fontsize=6.3)
        ax.axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates_dir = OUT / "candidates"
    sheets_dir = OUT / "contact_sheets"
    candidates_dir.mkdir(exist_ok=True)
    sheets_dir.mkdir(exist_ok=True)

    with np.load(CACHE / "main_rad5.npz") as data:
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)
        labels = data["labels"].astype(int)
    split_eval = np.load(CACHE / "split_eval_indices_stl10_test.npy")
    a = np.load(CACHE / "A_rad5.npy").astype(np.float64)
    cho = np.linalg.cholesky(a @ a.T)

    def adag(w: np.ndarray) -> np.ndarray:
        z = np.linalg.solve(cho, w.T)
        z = np.linalg.solve(cho.T, z)
        return z.T @ a

    xi = x[I_TARGET]
    yi = y[I_TARGET]
    ref_u = x[REFERENCE_DONOR] - adag((x[REFERENCE_DONOR] @ a.T - yi)[None, :])[0]
    ref_psnr = psnr_one(ref_u, xi)
    ref_clip = float(np.mean(np.maximum(-ref_u, 0.0) + np.maximum(ref_u - 1.0, 0.0)))
    ref_range = float(max(0.0, -ref_u.min()) + max(0.0, ref_u.max() - 1.0))

    candidate_indices = np.where(labels != labels[I_TARGET])[0]
    xj_all = x[candidate_indices]
    u_all = xj_all - adag(xj_all @ a.T - yi[None, :])
    clipped = np.clip(u_all, 0.0, 1.0)
    mse = np.mean((clipped - xi[None, :]) ** 2, axis=1)
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))
    clip_penalty = np.mean(np.maximum(-u_all, 0.0) + np.maximum(u_all - 1.0, 0.0), axis=1)
    range_penalty = np.maximum(-u_all.min(axis=1), 0.0) + np.maximum(u_all.max(axis=1) - 1.0, 0.0)
    rel_xj = np.linalg.norm(xj_all @ a.T - yi[None, :], axis=1) / max(np.linalg.norm(yi), 1e-12)
    score = (
        120.0 * clip_penalty
        + 12.0 * range_penalty
        + 0.28 * np.abs(psnr - ref_psnr)
        + 0.10 * np.abs(rel_xj - 0.90)
    )
    order = np.argsort(score)[:N_CANDIDATES]

    correct_path = OUT / f"correct_x_i_{I_TARGET}_car.png"
    ref_path = OUT / f"reference_fig1_horse_j{REFERENCE_DONOR}_clipped.png"
    save_gray(correct_path, xi)
    save_gray(ref_path, ref_u)

    metadata: dict[str, object] = {
        "construction": "u_ij = x_j - A_dagger(A x_j - y_i), fixed target measurement y_i from x_i",
        "target_i": int(I_TARGET),
        "target_label": LABEL_NAMES[int(labels[I_TARGET])],
        "target_original_stl10_test_index": int(split_eval[I_TARGET]),
        "reference_fig1_donor_j": int(REFERENCE_DONOR),
        "reference_fig1_donor_label": LABEL_NAMES[int(labels[REFERENCE_DONOR])],
        "reference_fig1_psnr_uij_vs_xi_clipped": ref_psnr,
        "reference_fig1_clip_penalty_mean": ref_clip,
        "reference_fig1_range_penalty": ref_range,
        "correct_image": str(correct_path),
        "reference_fig1_image": str(ref_path),
        "n_candidates": int(len(order)),
        "candidates": [],
    }

    rows: list[dict[str, object]] = []
    imgs: list[np.ndarray] = []
    for rank, pos in enumerate(order, start=1):
        pos = int(pos)
        j = int(candidate_indices[pos])
        uij = u_all[pos]
        out_path = candidates_dir / (
            f"rank{rank:03d}_u_i{I_TARGET}_j{j}_{LABEL_NAMES[int(labels[j])]}_clipped.png"
        )
        save_gray(out_path, uij)
        row = {
            "rank": int(rank),
            "donor_j": int(j),
            "donor_label_index": int(labels[j]),
            "donor_label": LABEL_NAMES[int(labels[j])],
            "donor_original_stl10_test_index": int(split_eval[j]),
            "relmeaserr_uij_vs_yi": rel(uij @ a.T, yi),
            "relmeaserr_xi_vs_yi": rel(xi @ a.T, yi),
            "relmeaserr_xj_vs_yi": float(rel_xj[pos]),
            "psnr_uij_vs_xi_clipped": float(psnr[pos]),
            "uij_min": float(uij.min()),
            "uij_max": float(uij.max()),
            "clip_penalty_mean": float(clip_penalty[pos]),
            "range_penalty": float(range_penalty[pos]),
            "quality_match_score": float(score[pos]),
            "image": str(out_path),
        }
        rows.append(row)
        imgs.append(uij)
        metadata["candidates"].append(row)

    sheet_paths: list[str] = []
    for sheet_idx, start in enumerate(range(0, len(rows), PER_SHEET), start=1):
        sheet_path = sheets_dir / f"candidate_sheet_{sheet_idx:02d}_ranks_{start + 1:03d}_{min(start + PER_SHEET, len(rows)):03d}.png"
        add_sheet(sheet_path, rows[start : start + PER_SHEET], imgs[start : start + PER_SHEET])
        sheet_paths.append(str(sheet_path))
    metadata["contact_sheets"] = sheet_paths

    csv_path = OUT / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata["csv"] = str(csv_path)
    metadata_path = OUT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "metadata": str(metadata_path), "n_candidates": len(rows), "sheets": sheet_paths}, indent=2))


if __name__ == "__main__":
    main()
