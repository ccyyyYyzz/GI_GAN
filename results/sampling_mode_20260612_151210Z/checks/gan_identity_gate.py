from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def orthonormal_row_operator(m: int, n: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(n, m, dtype=torch.float64, generator=generator)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.contiguous()


def verify_g_i1(tol: float, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    m, n, batch = 32, 96, 7
    A = orthonormal_row_operator(m, n, gen)
    A_pinv = A.T
    P0 = torch.eye(n, dtype=torch.float64) - A_pinv @ A
    v = torch.randn(batch, n, dtype=torch.float64, generator=gen)
    y = torch.randn(batch, m, dtype=torch.float64, generator=gen)
    pi_v = v - (A @ v.T - y.T).T @ A
    identity_rhs = v @ P0.T + y @ A_pinv.T
    p0_err = torch.linalg.norm(pi_v @ P0.T - v @ P0.T) / torch.linalg.norm(v @ P0.T).clamp_min(1e-30)
    y_err = torch.linalg.norm(A @ pi_v.T - y.T) / torch.linalg.norm(y.T).clamp_min(1e-30)
    closed_form_err = torch.linalg.norm(pi_v - identity_rhs) / torch.linalg.norm(identity_rhs).clamp_min(1e-30)
    max_rel = max(float(p0_err), float(y_err), float(closed_form_err))
    return {
        "name": "G-I1 exact audit/null-gauge identities",
        "tol": tol,
        "m": m,
        "n": n,
        "relative_error_P0_Pi_equals_P0": float(p0_err),
        "relative_error_A_Pi_equals_y": float(y_err),
        "relative_error_closed_form": float(closed_form_err),
        "max_relative_error": max_rel,
        "pass": max_rel <= tol,
    }


def verify_g_i2(tol_abs: float, seed: int, trials: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    m, n = 6, 16
    noise_std = 0.35
    A = orthonormal_row_operator(m, n, gen)
    eye_m = torch.eye(m, dtype=torch.float64)
    posterior_cov = torch.eye(n, dtype=torch.float64) - A.T @ torch.linalg.solve(A @ A.T + noise_std**2 * eye_m, A)
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    eigvals, eigvecs = torch.linalg.eigh(posterior_cov)
    eigvals = eigvals.clamp_min(0)
    sqrt_cov = eigvecs @ torch.diag(torch.sqrt(eigvals))
    mmse = float(torch.trace(posterior_cov))
    z1 = torch.randn(trials, n, dtype=torch.float64, generator=gen)
    z2 = torch.randn(trials, n, dtype=torch.float64, generator=gen)
    diff = (z1 - z2) @ sqrt_cov.T
    sampler_mse = float((diff.square().sum(dim=1)).mean())
    ratio = sampler_mse / mmse
    return {
        "name": "G-I2 Gaussian posterior 3 dB law",
        "tol_abs": tol_abs,
        "m": m,
        "n": n,
        "trials": trials,
        "noise_std": noise_std,
        "mmse_trace": mmse,
        "sampler_mse": sampler_mse,
        "sampler_mse_over_mmse": ratio,
        "pass": abs(ratio - 2.0) <= tol_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Identity gate for null-gauge GAN sampling mode.")
    parser.add_argument("--output", default=str(Path(__file__).resolve().with_name("identity_gate_results.json")))
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--tol_gi1", type=float, default=1e-12)
    parser.add_argument("--tol_gi2_abs", type=float, default=0.01)
    args = parser.parse_args()
    result = {
        "seed": args.seed,
        "G_I1": verify_g_i1(args.tol_gi1, args.seed),
        "G_I2": verify_g_i2(args.tol_gi2_abs, args.seed + 1, args.trials),
    }
    result["pass"] = bool(result["G_I1"]["pass"] and result["G_I2"]["pass"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
