from __future__ import annotations

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
OUT = Path(__file__).resolve().parent / "feasible_wrong_gallery_quality_matched"

I_TARGET = 1789
REFERENCE_DONOR = 935
QUALITY_DONORS = [1902, 53, 602, 1050]
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


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((np.clip(pred, 0.0, 1.0) - target) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def save_gray(path: Path, flat: np.ndarray) -> None:
    arr = np.clip(flat.reshape(64, 64), 0.0, 1.0)
    plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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
    ref_clip_penalty = float(np.mean(np.maximum(-ref_u, 0.0) + np.maximum(ref_u - 1.0, 0.0)))
    ref_psnr = psnr(ref_u, xi)
    ref_range_penalty = float(max(0.0, -ref_u.min()) + max(0.0, ref_u.max() - 1.0))

    selected: list[tuple[dict[str, object], np.ndarray]] = []
    for j in QUALITY_DONORS:
        if int(labels[j]) == int(labels[I_TARGET]):
            raise RuntimeError(f"Selected donor {j} has the same label as target.")
        xj = x[j]
        uij = xj - adag((xj @ a.T - yi)[None, :])[0]
        clip_penalty = float(np.mean(np.maximum(-uij, 0.0) + np.maximum(uij - 1.0, 0.0)))
        range_penalty = float(max(0.0, -uij.min()) + max(0.0, uij.max() - 1.0))
        row = {
            "target_i": int(I_TARGET),
            "target_label_index": int(labels[I_TARGET]),
            "target_label": LABEL_NAMES[int(labels[I_TARGET])],
            "target_original_stl10_test_index": int(split_eval[I_TARGET]),
            "donor_j": j,
            "donor_label_index": int(labels[j]),
            "donor_label": LABEL_NAMES[int(labels[j])],
            "donor_original_stl10_test_index": int(split_eval[j]),
            "relmeaserr_uij_vs_yi": rel(uij @ a.T, yi),
            "relmeaserr_xi_vs_yi": rel(xi @ a.T, yi),
            "relmeaserr_xj_vs_yi": rel(xj @ a.T, yi),
            "psnr_uij_vs_xi_clipped": psnr(uij, xi),
            "psnr_xj_vs_xi": psnr(xj, xi),
            "uij_min": float(uij.min()),
            "uij_max": float(uij.max()),
            "clip_penalty_mean": clip_penalty,
            "range_penalty": range_penalty,
        }
        selected.append((row, uij))

    correct_path = OUT / f"correct_x_i_{I_TARGET}_car.png"
    save_gray(correct_path, xi)
    reference_path = OUT / f"reference_fig1_horse_j{REFERENCE_DONOR}_clipped.png"
    save_gray(reference_path, ref_u)
    metadata: dict[str, object] = {
        "construction": "u_ij = x_j - A_dagger(A x_j - y_i), fixed target measurement y_i from x_i",
        "target_i": int(I_TARGET),
        "target_label": LABEL_NAMES[int(labels[I_TARGET])],
        "target_original_stl10_test_index": int(split_eval[I_TARGET]),
        "reference_fig1_donor_j": int(REFERENCE_DONOR),
        "reference_fig1_donor_label": LABEL_NAMES[int(labels[REFERENCE_DONOR])],
        "reference_fig1_psnr_uij_vs_xi_clipped": ref_psnr,
        "reference_fig1_clip_penalty_mean": ref_clip_penalty,
        "reference_fig1_range_penalty": ref_range_penalty,
        "reference_fig1_image": str(reference_path),
        "correct_image": str(correct_path),
        "wrong_images": [],
    }

    for k, (row, uij) in enumerate(selected, start=1):
        wrong_path = OUT / (
            f"wrong_{k}_u_i{I_TARGET}_j{row['donor_j']}_{row['donor_label']}_clipped.png"
        )
        save_gray(wrong_path, uij)
        row["wrong_image"] = str(wrong_path)
        metadata["wrong_images"].append(row)

    fig, axes = plt.subplots(1, 5, figsize=(10.0, 2.25), dpi=220)
    items = [("correct\ncar i=1789", xi)] + [
        (
            f"wrong {k}\n{row['donor_label']} j={row['donor_j']}\nRelErr={row['relmeaserr_uij_vs_yi']:.1e}",
            uij,
        )
        for k, (row, uij) in enumerate(selected, start=1)
    ]
    for ax, (title, img) in zip(axes, items):
        ax.imshow(np.clip(img.reshape(64, 64), 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    contact_path = OUT / "contact_sheet_correct_plus_4_wrong.png"
    fig.savefig(contact_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    metadata["contact_sheet"] = str(contact_path)
    metadata_path = OUT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
