from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="E:/ns_mc_gan_gi/outputs_phase69B_controlled_gauge_cgan_pilot")
    parser.add_argument("--cache", default="E:/ns_mc_gan_gi/results/cert_package_20260612/cache/main_scr5.npz")
    args = parser.parse_args()
    out = Path(args.out_dir)
    eval_dir = out / "evaluation"
    try:
        import lpips
    except Exception as exc:
        write_csv(
            out / "lpips_or_dists_results.csv",
            [{"metric_package": "LPIPS", "module": "lpips", "available": False, "note": str(exc)}],
            ["metric_package", "module", "available", "note"],
        )
        return 0

    n = np.load(eval_dir / "per_sample_outputs_A.npz")["x_hat_unclipped"].shape[0]
    z = np.load(Path(args.cache), allow_pickle=False)
    x_true = z["x"][:n].reshape(n, 64, 64).astype(np.float32)

    def prep(arr: np.ndarray) -> torch.Tensor:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        tensor = torch.from_numpy(arr[:, None, :, :])
        return tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    true_t = prep(x_true)
    rows: list[dict] = []
    per_rows: list[dict] = []
    for arm in ["A", "B", "C"]:
        pred = np.load(eval_dir / f"per_sample_outputs_{arm}.npz")["x_hat_unclipped"].astype(np.float32)
        pred_t = prep(pred)
        vals: list[float] = []
        with torch.no_grad():
            for i in range(0, n, 16):
                dist = loss_fn(pred_t[i : i + 16].to(device), true_t[i : i + 16].to(device))
                vals.extend(dist.reshape(-1).detach().cpu().numpy().astype(float).tolist())
        vals_np = np.asarray(vals, dtype=np.float64)
        rows.append(
            {
                "metric_package": "LPIPS",
                "module": "lpips",
                "available": True,
                "arm": arm,
                "n": n,
                "lpips_mean": float(vals_np.mean()),
                "lpips_median": float(np.median(vals_np)),
                "lpips_std": float(vals_np.std()),
                "note": "",
            }
        )
        for idx, value in enumerate(vals):
            per_rows.append({"arm": arm, "sample_ordinal": idx, "lpips": float(value)})
    rows.extend(
        [
            {"metric_package": "DISTS", "module": "DISTS_pytorch", "available": False, "arm": "", "n": "", "lpips_mean": "", "lpips_median": "", "lpips_std": "", "note": "module unavailable"},
            {"metric_package": "KID", "module": "torchmetrics/cleanfid", "available": "not_run", "arm": "", "n": n, "lpips_mean": "", "lpips_median": "", "lpips_std": "", "note": "small-sample KID not used for decision"},
        ]
    )
    write_csv(out / "lpips_or_dists_results.csv", rows, ["metric_package", "module", "available", "arm", "n", "lpips_mean", "lpips_median", "lpips_std", "note"])
    write_csv(out / "lpips_per_sample.csv", per_rows, ["arm", "sample_ordinal", "lpips"])

    b = np.asarray([row["lpips"] for row in per_rows if row["arm"] == "B"], dtype=np.float64)
    c = np.asarray([row["lpips"] for row in per_rows if row["arm"] == "C"], dtype=np.float64)
    improvement = b - c
    rng = np.random.default_rng(69070)
    idx = np.arange(n)
    boot = []
    for _ in range(1000):
        sample = rng.choice(idx, size=n, replace=True)
        boot.append(float(improvement[sample].mean()))
    comp = {
        "metric": "lpips",
        "direction": "lower",
        "mean_B": float(b.mean()),
        "mean_C": float(c.mean()),
        "mean_C_minus_B": float((c - b).mean()),
        "improvement_positive_means_C_better": float(improvement.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "ci_excludes_zero_in_favor_of_C": bool(np.percentile(boot, 2.5) > 0),
    }
    paired_path = out / "paired_comparison_C_vs_B.csv"
    with paired_path.open("r", encoding="utf-8", newline="") as f:
        paired = [row for row in csv.DictReader(f) if row.get("metric") != "lpips"]
    paired.append(comp)
    write_csv(
        paired_path,
        paired,
        ["metric", "direction", "mean_B", "mean_C", "mean_C_minus_B", "improvement_positive_means_C_better", "ci_low", "ci_high", "ci_excludes_zero_in_favor_of_C"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
