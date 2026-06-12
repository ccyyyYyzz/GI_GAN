from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from .phase53C_exact_projector import build_rowspace_basis, project_null
from .phase53D_common import (
    add_phase53d_args,
    configure_light_task,
    resolve_device,
    save_bar,
    write_rows,
)
from .utils import ensure_dir, save_json, set_seed


LAMBDA_GRID = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53D exact-null projector and soft leakage checks.")
    add_phase53d_args(parser)
    parser.add_argument("--lambda_grid", nargs="*", type=float, default=LAMBDA_GRID)
    return parser.parse_args()


def projector_checks(A: torch.Tensor, Q: torch.Tensor) -> dict[str, Any]:
    A = A.detach().float()
    Q = Q.detach().float()
    n = A.shape[1]
    eye_q = torch.eye(Q.shape[1], device=Q.device, dtype=Q.dtype)
    gen = torch.Generator(device=A.device)
    gen.manual_seed(5301)
    probe = torch.randn(min(96, n), n, device=A.device, dtype=A.dtype, generator=gen)
    p0_probe = project_null(probe, Q)
    pr_probe = probe - p0_probe
    S = torch.linalg.svdvals(A)
    return {
        "m": int(A.shape[0]),
        "n": int(A.shape[1]),
        "row_rank": int(torch.linalg.matrix_rank(A).item()),
        "basis_cols": int(Q.shape[1]),
        "qtq_minus_I_fro": float(torch.linalg.norm(Q.T @ Q - eye_q).item()),
        "A_P0_relative_norm": float((torch.linalg.norm(A @ p0_probe.T) / torch.linalg.norm(probe).clamp_min(1e-12)).item()),
        "P0_idempotence_relative_norm": float((torch.linalg.norm(project_null(p0_probe, Q) - p0_probe) / torch.linalg.norm(probe).clamp_min(1e-12)).item()),
        "PR_idempotence_relative_norm": float((torch.linalg.norm(((pr_probe @ Q) @ Q.T) - pr_probe) / torch.linalg.norm(probe).clamp_min(1e-12)).item()),
        "singular_min": float(S.min().item()),
        "singular_max": float(S.max().item()),
        "singular_mean": float(S.mean().item()),
        "singular_std": float(S.std().item()),
        "condition_number": float((S.max() / S.min().clamp_min(1e-12)).item()),
        "row_gram_minus_I_fro": float(torch.linalg.norm(A @ A.T - torch.eye(A.shape[0], device=A.device, dtype=A.dtype)).item()),
    }


def soft_leakage_rows(task: str, family: str, A: torch.Tensor, lambdas: list[float]) -> list[dict[str, Any]]:
    A = A.detach().float()
    S = torch.linalg.svdvals(A)
    rows = []
    for lam in lambdas:
        lam_t = torch.tensor(float(lam), device=S.device, dtype=S.dtype)
        factors = lam_t / (S.square() + lam_t)
        rows.append(
            {
                "task": task,
                "family": family,
                "lambda": lam,
                "mean_theory_leakage_factor": float(factors.mean().item()),
                "max_theory_leakage_factor": float(factors.max().item()),
                "min_theory_leakage_factor": float(factors.min().item()),
                "expected_growth": "lambda/(lambda+sigma_i^2)",
            }
        )
    return rows


def save_singular_plot(path: Path, rows_by_task: dict[str, torch.Tensor]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for task, s in rows_by_task.items():
        vals = s.detach().cpu().float().numpy()
        plt.plot(range(1, len(vals) + 1), vals, label=task)
    plt.xlabel("singular value index")
    plt.ylabel("singular value")
    plt.title("Measurement operator singular spectra")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    set_seed(args.seed)
    exact_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    spectra: dict[str, torch.Tensor] = {}
    for task in args.tasks:
        task_out = ensure_dir(out / "exact_projector" / task)
        info, _config, measurement, exact_info = configure_light_task(args, task, task_out, device)
        A = measurement.get_current_A().detach().float().to(device)
        Q = build_rowspace_basis(A)
        checks = projector_checks(A, Q)
        row = {
            "task": task,
            "family": info["metadata"]["display"],
            "sampling_pct": info["metadata"]["sampling_pct"],
            "requires_exact_A": info["metadata"]["requires_exact_A"],
            "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
            "exact_A_path": exact_info.get("exact_A_path", ""),
            **checks,
            "interpretation": "exact P0 is numerically row-blind" if checks["A_P0_relative_norm"] < 1e-4 else "check A/P0 leakage",
        }
        exact_rows.append(row)
        task_leak = soft_leakage_rows(task, info["metadata"]["display"], A, args.lambda_grid)
        leakage_rows.extend(task_leak)
        spectra[task] = torch.linalg.svdvals(A).detach().cpu()
        torch.save({"Q": Q.detach().cpu(), "checks": checks}, task_out / "Q_exact_null.pt")
        write_rows(task_out, "exact_projector_checks", [row], "Exact Projector Checks")
        write_rows(task_out, "soft_leakage_by_lambda", task_leak, "Soft Projector Leakage by Lambda")
        save_json({"exact_A_info": exact_info, "projector_checks": checks}, task_out / "projector_check_manifest.json")
    write_rows(out, "exact_projector_checks", exact_rows, "Phase53D Exact Projector Checks")
    write_rows(out, "soft_leakage_by_lambda", leakage_rows, "Phase53D Soft Projector Leakage by Lambda")
    save_bar(out / "soft_leakage_by_lambda.png", leakage_rows, "lambda", "mean_theory_leakage_factor", "Soft P_N leakage grows with lambda", "mean leakage factor")
    save_singular_plot(out / "singular_value_spectrum.png", spectra)
    report = [
        "# Phase53D Exact Projector Report",
        "",
        "This local preflight constructs exact row-space bases from `A^T = QR` and checks `P0 = I - QQ^T`.",
        "",
        "Key rule: exact-null critic inputs must use exact `P0`; soft `P_N^lambda` is only a diagnostic because its row-space leakage scales as `lambda/(lambda + sigma_i^2)`.",
        "",
        "## Summary",
    ]
    for row in exact_rows:
        report.append(
            f"- {row['task']}: A_P0_relative_norm={row['A_P0_relative_norm']:.3e}, "
            f"rank={row['row_rank']}/{row['m']}, exact_A_loaded={row['exact_A_loaded']}."
        )
    (out / "EXACT_PROJECTOR_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "EXACT_PROJECTOR_REPORT.md")


if __name__ == "__main__":
    main()

