from __future__ import annotations

import json

from .phase16_common import ALL_MAIN_METHODS, PHASE16, evaluate_backprojection, evaluate_tv_pgd, save_bar_plot, write_all


OUT = PHASE16 / "traditional_baselines"
FIELDS = ["method_id", "baseline", "dataset", "sampling_ratio", "measurement_family", "num_samples", "iterations", "lambda_tv", "psnr", "ssim", "mse", "rel_meas_err", "runtime_sec", "status", "notes"]


def main() -> None:
    rows = []
    for method_id in ALL_MAIN_METHODS:
        linear_limit = 100 if "stl" in method_id or "rademacher" in method_id or "scrambled" in method_id else 300
        rows.append(evaluate_backprojection(method_id, limit=linear_limit, mode="ridge_pinv"))
        rows.append(evaluate_backprojection(method_id, limit=linear_limit, mode="adjoint"))
        tv_limit = 24 if "stl" in method_id or "rademacher" in method_id or "scrambled" in method_id else 48
        for lam in [0.001, 0.003, 0.01]:
            try:
                rows.append(evaluate_tv_pgd(method_id, limit=tv_limit, iterations=50, lambda_tv=lam))
            except Exception as exc:
                rows.append({"method_id": method_id, "baseline": "tv_pgd", "lambda_tv": lam, "status": "failed", "notes": f"{type(exc).__name__}: {exc}"})
    write_all(OUT / "tv_pgd_baseline_results", rows, FIELDS)
    best = []
    for method_id in ALL_MAIN_METHODS:
        sub = [r for r in rows if r.get("method_id") == method_id and r.get("status") == "completed"]
        if sub:
            best.append(max(sub, key=lambda r: float(r.get("psnr", -1e9))))
    save_bar_plot(best, OUT / "traditional_baseline_psnr.png", "psnr", title="Best traditional baseline PSNR", ylabel="PSNR")
    save_bar_plot(best, OUT / "traditional_baseline_ssim.png", "ssim", title="Best traditional baseline SSIM", ylabel="SSIM")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
