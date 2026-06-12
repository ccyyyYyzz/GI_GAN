"""Build and verify exact null-space projectors P0 = I - A_pinv A per config.

For each (ensemble, rate) task in {rad5, scr5, rad10, scr10}:
  * load the canonical A (rademacher: measurement_operator_exact.pt from the
    phase15 no-leak bundle; scrambled hadamard: rebuilt deterministically from
    the bundle's resolved_config.yaml),
  * compute the SVD in float64 and form P0 = I - V_r V_r^T  (== I - A^+ A),
  * verify on random float64 probes:
        max_v ||A @ P0 @ v|| / ||A @ v||  <= 1e-12
        ||P0 @ P0 - P0||_F                <= 1e-10
  * persist a float32 artifact (P0 is 4096x4096 dense, ~64 MB) plus a JSON
    manifest with source/artifact SHA-256 hashes and the verification numbers.

Usage:
    python scripts/g2r/build_p0.py --bundle_root E:/ns_mc_gan_gi/outputs_phase15/imported_noleak \
        --out_root E:/ns_mc_gan_gi/results/g2r_protocol/p0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import make_measurement  # noqa: E402
from src.exact_measurement import tensor_from_exact_payload, torch_load  # noqa: E402
from src.phase48_49_common import TASKS, file_sha256, load_bundle_task  # noqa: E402
from src.utils import apply_experiment_defaults  # noqa: E402

RANGE_LEAK_TOL = 1e-12
IDEMPOTENCE_TOL = 1e-10
N_PROBES = 32
PROBE_SEED = 20260612


def load_canonical_A(bundle_root: str, task_key: str) -> tuple[torch.Tensor, dict]:
    info = load_bundle_task(bundle_root, task_key)
    meta = {
        "task": task_key,
        "ensemble": TASKS[task_key]["sampling_family"],
        "sampling_pct": TASKS[task_key]["sampling_pct"],
        "config_path": str(info["config_path"]),
        "config_sha256": file_sha256(info["config_path"]),
    }
    if info["exact_A_path"] is not None:
        payload = torch_load(info["exact_A_path"], map_location="cpu")
        A = tensor_from_exact_payload(payload).to(torch.float32)
        meta["A_source"] = str(info["exact_A_path"])
        meta["A_source_sha256"] = file_sha256(info["exact_A_path"])
        meta["A_source_kind"] = "measurement_operator_exact.pt"
    else:
        config = apply_experiment_defaults(dict(info["config"]))
        measurement = make_measurement(config, torch.device("cpu"))
        A = measurement.get_current_A().detach().to(torch.float32)
        meta["A_source"] = str(info["config_path"])
        meta["A_source_sha256"] = meta["config_sha256"]
        meta["A_source_kind"] = "rebuilt_from_resolved_config(seed-deterministic)"
    return A, meta


def build_p0_float64(A: torch.Tensor) -> tuple[torch.Tensor, int]:
    """P0 = I - V_r V_r^T from the SVD of A, all in float64."""
    A64 = A.detach().to(torch.float64)
    _U, S, Vh = torch.linalg.svd(A64, full_matrices=False)
    rtol = max(A64.shape) * torch.finfo(S.dtype).eps
    rank = int((S > rtol * S.max().clamp_min(1e-300)).sum().item())
    if rank <= 0:
        raise ValueError("A appears rank deficient with rank 0.")
    Vr = Vh[:rank].T.contiguous()
    n = A64.shape[1]
    P0 = torch.eye(n, dtype=torch.float64) - Vr @ Vr.T
    return P0, rank


def verify_p0_float64(
    A: torch.Tensor,
    P0: torch.Tensor,
    n_probes: int = N_PROBES,
    seed: int = PROBE_SEED,
) -> dict:
    """Float64 verification of range-leakage and idempotence; returns numbers."""
    A64 = A.detach().to(torch.float64)
    n = A64.shape[1]
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    V = torch.randn(n, int(n_probes), dtype=torch.float64, generator=gen)
    Av = A64 @ V
    APv = A64 @ (P0 @ V)
    ratios = torch.linalg.norm(APv, dim=0) / torch.linalg.norm(Av, dim=0).clamp_min(1e-300)
    idem = torch.linalg.norm(P0 @ P0 - P0, ord="fro").item()
    results = {
        "n_probes": int(n_probes),
        "range_leak_max": float(ratios.max().item()),
        "range_leak_mean": float(ratios.mean().item()),
        "idempotence_fro": float(idem),
        "range_leak_tol": RANGE_LEAK_TOL,
        "idempotence_tol": IDEMPOTENCE_TOL,
    }
    results["range_leak_pass"] = results["range_leak_max"] <= RANGE_LEAK_TOL
    results["idempotence_pass"] = results["idempotence_fro"] <= IDEMPOTENCE_TOL
    results["pass"] = results["range_leak_pass"] and results["idempotence_pass"]
    return results


def build_task(bundle_root: str, task_key: str, out_root: Path) -> dict:
    A, meta = load_canonical_A(bundle_root, task_key)
    m, n = int(A.shape[0]), int(A.shape[1])
    P0, rank = build_p0_float64(A)
    checks = verify_p0_float64(A, P0)
    print(
        f"[{task_key}] m={m} n={n} rank={rank} | "
        f"max ||A P0 v||/||A v|| = {checks['range_leak_max']:.3e} (tol {RANGE_LEAK_TOL:.0e}) | "
        f"||P0 P0 - P0||_F = {checks['idempotence_fro']:.3e} (tol {IDEMPOTENCE_TOL:.0e}) | "
        f"{'PASS' if checks['pass'] else 'FAIL'}"
    )
    entry = {
        **meta,
        "m": m,
        "n": n,
        "rank": rank,
        "verification_float64": checks,
        "torch_version": torch.__version__,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if not checks["pass"]:
        entry["artifact"] = None
        return entry
    out_root.mkdir(parents=True, exist_ok=True)
    artifact_path = out_root / f"p0_{task_key}.pt"
    torch.save(
        {
            "P0": P0.to(torch.float32),
            "task": task_key,
            "ensemble": meta["ensemble"],
            "sampling_pct": meta["sampling_pct"],
            "rank": rank,
            "A_source": meta["A_source"],
            "A_source_sha256": meta["A_source_sha256"],
            "verification_float64": checks,
        },
        artifact_path,
    )
    entry["artifact"] = {
        "path": str(artifact_path),
        "dtype": "float32",
        "sha256": file_sha256(artifact_path),
        "bytes": artifact_path.stat().st_size,
    }
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact P0 projectors per (ensemble, rate).")
    parser.add_argument("--bundle_root", default=r"E:\ns_mc_gan_gi\outputs_phase15\imported_noleak")
    parser.add_argument("--out_root", default=r"E:\ns_mc_gan_gi\results\g2r_protocol\p0")
    parser.add_argument("--tasks", nargs="*", default=["rad5", "scr5", "rad10", "scr10"])
    parser.add_argument(
        "--manifest_copy",
        default=str(REPO_ROOT / "artifacts" / "p0_manifest.json"),
        help="Repo-side manifest copy (committed); pass '' to skip.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    entries = [build_task(args.bundle_root, task, out_root) for task in args.tasks]
    manifest = {
        "kind": "g2r_p0_manifest",
        "bundle_root": str(args.bundle_root),
        "range_leak_tol": RANGE_LEAK_TOL,
        "idempotence_tol": IDEMPOTENCE_TOL,
        "tasks": entries,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "p0_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")
    if args.manifest_copy:
        copy_path = Path(args.manifest_copy)
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest copy written: {copy_path}")
    failed = [e["task"] for e in entries if not e["verification_float64"]["pass"]]
    if failed:
        print(f"FAILED verification: {failed}")
        return 1
    print("All P0 verifications PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
