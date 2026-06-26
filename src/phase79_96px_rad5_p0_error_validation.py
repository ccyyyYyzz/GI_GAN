from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from . import phase78_96px_rad5_one_seed_probe as p78


DATA_ROOT = Path("E:/ns_mc_gan_gi")
SOURCE_OUT = DATA_ROOT / "outputs_phase78_96px_rad5_one_seed_probe"
OUT = DATA_ROOT / "outputs_phase79_96px_rad5_p0_error_validation"
SEED_DIR = SOURCE_OUT / "seed01"
ARMS = ("A", "B", "C")
TEST_BATCH_SIZE = 8
N_BINS = 10


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(p69a.json_safe(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_log(message: str) -> None:
    ensure_dir(OUT)
    with (OUT / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {now()} {message}\n")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    sx = float(x.std())
    sy = float(y.std())
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(stats.spearmanr(x[mask], y[mask]).statistic)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_test_cache(measurement, device: torch.device) -> tuple[p78.SplitCache, dict[str, Any]]:
    eval_indices_full = np.load(p78.SPLIT_EVAL).astype(np.int64)
    test_indices = eval_indices_full[: p78.TEST_COUNT]
    base_test = p78.stl10_dataset_96("test")
    test = p78.build_split_cache("test", base_test, test_indices, measurement, device, seed=78023)
    split = {
        "test_count": int(test_indices.shape[0]),
        "test_source": "STL10 official test partition resized to 96x96 grayscale",
        "eval_full_sorted_sha256": p69a.sha256_np(eval_indices_full, sort_int64=True),
        "test_indices_sha256": p69a.sha256_np(test_indices),
        "split_eval_path": str(p78.SPLIT_EVAL),
    }
    return test, split


def make_loader(cache: p78.SplitCache) -> DataLoader:
    return DataLoader(
        cache.dataset(),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def checkpoint_for_arm(arm: str) -> Path | None:
    if arm == "A":
        return None
    return SEED_DIR / arm / "checkpoints" / "best_by_val.pt"


def load_arm_generator(arm: str, config: dict[str, Any], measurement, device: torch.device):
    if arm == "A":
        return p78.load_generator_96(config, measurement, device, train=False)
    ckpt = checkpoint_for_arm(arm)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint for arm {arm}: {ckpt}")
    return p78.load_probe_checkpoint_for_eval(ckpt, config, measurement, device)


@torch.no_grad()
def collect_arm_arrays(
    arm: str,
    generator,
    measurement,
    test: p78.SplitCache,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    generator.eval()
    x_true: list[np.ndarray] = []
    x_hat: list[np.ndarray] = []
    p0_xhat: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    relmeas: list[np.ndarray] = []
    for x, y, lab, idx in make_loader(test):
        x = x.to(device)
        y = y.to(device)
        out = p78.forward_candidate(generator, measurement, x, y, config)
        xhat_flat = out["x_hat_flat"].detach()
        p0_flat = measurement.null_project(xhat_flat)
        rel = p78.relmeas_batch(xhat_flat, y, measurement)
        x_true.append(x.detach().cpu().numpy()[:, 0].astype(np.float32))
        x_hat.append(out["x_hat"].detach().cpu().numpy()[:, 0].astype(np.float32))
        p0_xhat.append(measurement.unflatten_img(p0_flat).detach().cpu().numpy()[:, 0].astype(np.float32))
        labels.append(lab.detach().cpu().numpy().astype(np.int64))
        indices.append(idx.detach().cpu().numpy().astype(np.int64))
        relmeas.append(rel.astype(np.float64))
    x_true_arr = np.concatenate(x_true, axis=0)
    x_hat_arr = np.concatenate(x_hat, axis=0)
    p0_arr = np.concatenate(p0_xhat, axis=0)
    return {
        "sample_index": np.concatenate(indices, axis=0),
        "label": np.concatenate(labels, axis=0),
        "x_true": x_true_arr,
        "x_hat_unclipped": x_hat_arr,
        "x_hat_clipped": np.clip(x_hat_arr, 0.0, 1.0).astype(np.float32),
        "p0_xhat": p0_arr,
        "abs_p0_xhat": np.abs(p0_arr).astype(np.float32),
        "abs_error_unclipped": np.abs(x_hat_arr - x_true_arr).astype(np.float32),
        "abs_error_clipped": np.abs(np.clip(x_hat_arr, 0.0, 1.0) - x_true_arr).astype(np.float32),
        "relmeaserr_unclipped_float64": np.concatenate(relmeas, axis=0),
    }


def pooled_bin_rows(arm: str, h: np.ndarray, err: np.ndarray, err_clip: np.ndarray) -> list[dict[str, Any]]:
    h_flat = h.reshape(-1).astype(np.float64)
    e_flat = err.reshape(-1).astype(np.float64)
    ec_flat = err_clip.reshape(-1).astype(np.float64)
    order = np.argsort(h_flat, kind="mergesort")
    bins = np.array_split(order, N_BINS)
    rows: list[dict[str, Any]] = []
    for b, idx in enumerate(bins):
        rows.append(
            {
                "arm": arm,
                "bin": b,
                "scope": "pooled_pixels",
                "n_pixels": int(idx.size),
                "abs_p0_xhat_mean": float(h_flat[idx].mean()),
                "abs_p0_xhat_min": float(h_flat[idx].min()),
                "abs_p0_xhat_max": float(h_flat[idx].max()),
                "abs_error_unclipped_mean": float(e_flat[idx].mean()),
                "abs_error_clipped_mean": float(ec_flat[idx].mean()),
            }
        )
    return rows


def sample_bin_rows(arm: str, h: np.ndarray, err: np.ndarray, err_clip: np.ndarray) -> list[dict[str, Any]]:
    per_bin_e: list[list[float]] = [[] for _ in range(N_BINS)]
    per_bin_ec: list[list[float]] = [[] for _ in range(N_BINS)]
    per_bin_h: list[list[float]] = [[] for _ in range(N_BINS)]
    for i in range(h.shape[0]):
        hi = h[i].reshape(-1).astype(np.float64)
        ei = err[i].reshape(-1).astype(np.float64)
        eci = err_clip[i].reshape(-1).astype(np.float64)
        order = np.argsort(hi, kind="mergesort")
        for b, idx in enumerate(np.array_split(order, N_BINS)):
            per_bin_h[b].append(float(hi[idx].mean()))
            per_bin_e[b].append(float(ei[idx].mean()))
            per_bin_ec[b].append(float(eci[idx].mean()))
    rows: list[dict[str, Any]] = []
    for b in range(N_BINS):
        vals = np.asarray(per_bin_e[b], dtype=np.float64)
        vals_clip = np.asarray(per_bin_ec[b], dtype=np.float64)
        rows.append(
            {
                "arm": arm,
                "bin": b,
                "scope": "within_sample_deciles_mean_over_samples",
                "n_samples": int(h.shape[0]),
                "abs_p0_xhat_mean": float(np.mean(per_bin_h[b])),
                "abs_error_unclipped_mean": float(vals.mean()),
                "abs_error_unclipped_std_over_samples": float(vals.std(ddof=1)),
                "abs_error_unclipped_se": float(vals.std(ddof=1) / math.sqrt(vals.size)),
                "abs_error_clipped_mean": float(vals_clip.mean()),
                "abs_error_clipped_std_over_samples": float(vals_clip.std(ddof=1)),
                "abs_error_clipped_se": float(vals_clip.std(ddof=1) / math.sqrt(vals_clip.size)),
            }
        )
    return rows


def per_sample_rows(arm: str, arrays: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    h = arrays["abs_p0_xhat"]
    err = arrays["abs_error_unclipped"]
    err_clip = arrays["abs_error_clipped"]
    rows: list[dict[str, Any]] = []
    for i in range(h.shape[0]):
        hi = h[i].reshape(-1)
        ei = err[i].reshape(-1)
        eci = err_clip[i].reshape(-1)
        p = pearson(hi, ei)
        s = spearman(hi, ei)
        pc = pearson(hi, eci)
        sc = spearman(hi, eci)
        order = np.argsort(hi, kind="mergesort")
        top = order[int(0.9 * order.size) :]
        rest = order[: int(0.9 * order.size)]
        rows.append(
            {
                "arm": arm,
                "sample_row": i,
                "sample_index": int(arrays["sample_index"][i]),
                "label": int(arrays["label"][i]),
                "pixel_pearson_abs_p0_vs_abs_error_unclipped": p,
                "pixel_spearman_abs_p0_vs_abs_error_unclipped": s,
                "pixel_pearson_abs_p0_vs_abs_error_clipped": pc,
                "pixel_spearman_abs_p0_vs_abs_error_clipped": sc,
                "abs_p0_xhat_mean": float(hi.mean()),
                "abs_p0_xhat_top10_mean": float(hi[top].mean()),
                "abs_error_unclipped_mean": float(ei.mean()),
                "abs_error_unclipped_top10_p0_mean": float(ei[top].mean()),
                "abs_error_unclipped_rest90_p0_mean": float(ei[rest].mean()),
                "abs_error_clipped_mean": float(eci.mean()),
                "abs_error_clipped_top10_p0_mean": float(eci[top].mean()),
                "abs_error_clipped_rest90_p0_mean": float(eci[rest].mean()),
                "relmeaserr_unclipped_float64": float(arrays["relmeaserr_unclipped_float64"][i]),
            }
        )
    rng = np.random.default_rng(79001 + ARMS.index(arm))
    spears = np.asarray([r["pixel_spearman_abs_p0_vs_abs_error_unclipped"] for r in rows], dtype=np.float64)
    pears = np.asarray([r["pixel_pearson_abs_p0_vs_abs_error_unclipped"] for r in rows], dtype=np.float64)
    ci_s = bootstrap_ci(spears, rng)
    ci_p = bootstrap_ci(pears, rng)
    summary = {
        "arm": arm,
        "n_samples": int(h.shape[0]),
        "per_sample_pearson_mean": float(np.nanmean(pears)),
        "per_sample_pearson_median": float(np.nanmedian(pears)),
        "per_sample_pearson_mean_ci_low": ci_p[0],
        "per_sample_pearson_mean_ci_high": ci_p[1],
        "per_sample_spearman_mean": float(np.nanmean(spears)),
        "per_sample_spearman_median": float(np.nanmedian(spears)),
        "per_sample_spearman_mean_ci_low": ci_s[0],
        "per_sample_spearman_mean_ci_high": ci_s[1],
    }
    return rows, summary


def summarize_arm(arm: str, arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    h = arrays["abs_p0_xhat"]
    err = arrays["abs_error_unclipped"]
    err_clip = arrays["abs_error_clipped"]
    per_rows, per_summary = per_sample_rows(arm, arrays)
    pooled_rows = pooled_bin_rows(arm, h, err, err_clip)
    within_rows = sample_bin_rows(arm, h, err, err_clip)
    top_idx = np.argsort(h.reshape(-1), kind="mergesort")[int(0.9 * h.size) :]
    rest_idx = np.argsort(h.reshape(-1), kind="mergesort")[: int(0.9 * h.size)]
    e_flat = err.reshape(-1)
    ec_flat = err_clip.reshape(-1)
    summary = {
        "arm": arm,
        "n_samples": int(h.shape[0]),
        "n_pixels": int(h.size),
        "pooled_pixel_pearson_abs_p0_vs_abs_error_unclipped": pearson(h, err),
        "pooled_pixel_spearman_abs_p0_vs_abs_error_unclipped": spearman(h, err),
        "pooled_pixel_pearson_abs_p0_vs_abs_error_clipped": pearson(h, err_clip),
        "pooled_pixel_spearman_abs_p0_vs_abs_error_clipped": spearman(h, err_clip),
        "abs_p0_xhat_mean": float(h.mean()),
        "abs_p0_xhat_median": float(np.median(h)),
        "abs_error_unclipped_mean": float(err.mean()),
        "abs_error_clipped_mean": float(err_clip.mean()),
        "top10_abs_p0_abs_error_unclipped_mean": float(e_flat[top_idx].mean()),
        "rest90_abs_p0_abs_error_unclipped_mean": float(e_flat[rest_idx].mean()),
        "top10_abs_p0_abs_error_clipped_mean": float(ec_flat[top_idx].mean()),
        "rest90_abs_p0_abs_error_clipped_mean": float(ec_flat[rest_idx].mean()),
    }
    summary.update(per_summary)
    return summary, per_rows, pooled_rows, within_rows


def save_npz(arm: str, arrays: dict[str, np.ndarray]) -> Path:
    path = OUT / "per_sample_pixel_outputs" / f"per_sample_p0_error_{arm}.npz"
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        sample_index=arrays["sample_index"],
        label=arrays["label"],
        x_true=arrays["x_true"].astype(np.float32),
        x_hat_unclipped=arrays["x_hat_unclipped"].astype(np.float32),
        x_hat_clipped=arrays["x_hat_clipped"].astype(np.float32),
        p0_xhat=arrays["p0_xhat"].astype(np.float32),
        abs_p0_xhat=arrays["abs_p0_xhat"].astype(np.float32),
        abs_error_unclipped=arrays["abs_error_unclipped"].astype(np.float32),
        abs_error_clipped=arrays["abs_error_clipped"].astype(np.float32),
        relmeaserr_unclipped_float64=arrays["relmeaserr_unclipped_float64"].astype(np.float64),
    )
    return path


def plot_results(
    summaries: list[dict[str, Any]],
    binned: list[dict[str, Any]],
    sampled_pixels: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    ensure_dir(OUT / "figs")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for col, arm in enumerate(ARMS):
        ax = axes[0, col]
        h, e = sampled_pixels[arm]
        ax.hexbin(h, e, gridsize=55, bins="log", mincnt=1, cmap="viridis")
        summ = next(r for r in summaries if r["arm"] == arm)
        ax.set_title(
            f"{arm}: r={summ['pooled_pixel_pearson_abs_p0_vs_abs_error_unclipped']:.3f}, "
            f"rho={summ['pooled_pixel_spearman_abs_p0_vs_abs_error_unclipped']:.3f}"
        )
        ax.set_xlabel("|P0 xhat|")
        ax.set_ylabel("|xhat - x|")
        ax2 = axes[1, col]
        rows = [r for r in binned if r["arm"] == arm and r["scope"] == "within_sample_deciles_mean_over_samples"]
        rows = sorted(rows, key=lambda r: int(r["bin"]))
        xs = [int(r["bin"]) for r in rows]
        ys = [float(r["abs_error_unclipped_mean"]) for r in rows]
        se = [float(r["abs_error_unclipped_se"]) for r in rows]
        ax2.errorbar(xs, ys, yerr=se, marker="o", linewidth=1.8)
        ax2.set_xlabel("|P0 xhat| within-sample decile")
        ax2.set_ylabel("mean |xhat - x|")
        ax2.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig_p0_error_validation_A_B_C.png", dpi=200)
    fig.savefig(OUT / "figs" / "fig_p0_error_validation_A_B_C.pdf")
    plt.close(fig)


def write_report(summaries: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase79 96px Rad-5 P0 Error Validation",
        "",
        "Post-hoc only: no training, no test-split tuning. Existing Phase78 A/B/C checkpoints were evaluated on the locked 256-image STL10 official test split.",
        "",
        "Primary test: pixelwise association between `|P0 xhat|` and actual absolute reconstruction error `|xhat - x|`.",
        "",
        "## Summary",
        "",
        p69a.format_table(
            summaries,
            [
                "arm",
                "n_samples",
                "n_pixels",
                "pooled_pixel_pearson_abs_p0_vs_abs_error_unclipped",
                "pooled_pixel_spearman_abs_p0_vs_abs_error_unclipped",
                "per_sample_spearman_mean",
                "top10_abs_p0_abs_error_unclipped_mean",
                "rest90_abs_p0_abs_error_unclipped_mean",
            ],
        ),
        "",
        "## Interpretation Guardrail",
        "",
        "A positive correlation means high-magnitude unmeasured-content pixels are also higher-error pixels on this locked test set. A weak or near-zero correlation would be a negative result for the quantitative diagnostic claim.",
        "",
        "## Reproducibility",
        "",
        f"- output_dir: `{OUT}`",
        f"- source_output_dir: `{SOURCE_OUT}`",
        f"- measurement A sha256 float32 bytes: `{manifest['measurement']['A_sha256_float32_bytes']}`",
        f"- test split sha256: `{manifest['split']['test_indices_sha256']}`",
        "",
    ]
    (OUT / "PHASE79_P0_ERROR_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir(OUT)
    append_log("phase79_start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p78.set_seed(790000)
    config = p78.make_config(device)
    measurement = p78.make_measurement(config, device)
    test, split = load_test_cache(measurement, device)
    measurement_manifest = {
        "img_size": int(measurement.img_size),
        "n": int(measurement.n),
        "m": int(measurement.m),
        "sampling_ratio_effective": float(measurement.m / measurement.n),
        "pattern_type": measurement.pattern_type,
        "matrix_normalization": measurement.matrix_normalization,
        "A_sha256_float32_bytes": p69a.sha256_np(measurement.A.detach().cpu().numpy().astype(np.float32)),
    }
    summaries: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    binned: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    sampled_pixels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(790002)
    for arm in ARMS:
        append_log(f"collect_arm_start arm={arm}")
        gen = load_arm_generator(arm, config, measurement, device)
        arrays = collect_arm_arrays(arm, gen, measurement, test, config, device)
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        npz_path = save_npz(arm, arrays)
        summary, per_rows, pooled_rows, within_rows = summarize_arm(arm, arrays)
        summaries.append(summary)
        per_sample.extend(per_rows)
        binned.extend(pooled_rows)
        binned.extend(within_rows)
        h_flat = arrays["abs_p0_xhat"].reshape(-1)
        e_flat = arrays["abs_error_unclipped"].reshape(-1)
        n_scatter = min(200000, h_flat.size)
        idx = rng.choice(h_flat.size, size=n_scatter, replace=False)
        sampled_pixels[arm] = (h_flat[idx], e_flat[idx])
        output_rows.append(
            {
                "arm": arm,
                "npz_path": str(npz_path),
                "npz_sha256": p69a.sha256_file(npz_path),
                "source_phase78_npz": str(SEED_DIR / "evaluation" / f"per_sample_outputs_{arm}.npz"),
                "source_phase78_npz_sha256": p69a.sha256_file(SEED_DIR / "evaluation" / f"per_sample_outputs_{arm}.npz"),
                "checkpoint_path": str(checkpoint_for_arm(arm) or p78.RAD5_CHECKPOINT),
                "checkpoint_sha256": p69a.sha256_file(checkpoint_for_arm(arm) or p78.RAD5_CHECKPOINT),
            }
        )
        append_log(f"collect_arm_complete arm={arm} npz={npz_path}")
    write_csv(OUT / "p0_error_correlation_summary.csv", summaries)
    write_csv(OUT / "p0_error_per_sample_stats.csv", per_sample)
    write_csv(OUT / "p0_error_binned_curve.csv", binned)
    write_csv(OUT / "per_sample_output_manifest.csv", output_rows)
    plot_results(summaries, binned, sampled_pixels)
    manifest = {
        "phase": "Phase79_96px_rad5_p0_error_validation",
        "created_at": now(),
        "device": str(device),
        "source_output_dir": str(SOURCE_OUT),
        "output_dir": str(OUT),
        "post_hoc_only_no_training": True,
        "primary_metric": "pixelwise correlation between |P0 xhat| and |xhat - x|",
        "measurement": measurement_manifest,
        "split": split,
        "outputs": output_rows,
    }
    save_json(OUT / "PHASE79_MANIFEST.json", manifest)
    write_report(summaries, manifest)
    append_log("phase79_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
