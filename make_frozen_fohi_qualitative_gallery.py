"""Render a pre-specified, metric-blind FOHI held-out qualitative gallery.

This is a *post-decision visualisation* program.  It does not train, tune, or
select by reconstruction quality.  The four scenes are selected exclusively
from lane 0's rate-05 cache manifest: the smallest STL-10 official-test
``source_index`` in each of the car (2), cat (3), dog (5), and horse (6)
classes.  Those same source records are then rendered at 5% and 10% sampling.

The program reconstructs only those four already-held-out cache records using
the frozen lane-0 weights and the frozen FOHI parameters.  Its final
structural, fixed, and FOHI projections target the intrinsic record computed
from the *raw cached bucket vector* ``y``.  It writes raw arrays, faithful
image plates, projection certificates, and SHA-256 provenance.  The existing
held-out metric vectors are deliberately not reported as metrics of these
raw-fiber outputs: they belong to the earlier clipped-anchor terminal
projection.  Round59 raw-y vectors are instead hash-checked and emitted for
the pre-specified records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import gan_high_quality_gi as hq
from diagnose_fiber_residual_frequency_fusion import (
    load_generator,
    smooth_radial_high_pass,
)
from src.fiber_orthogonal_innovation import fiber_orthogonal_innovation
from src.gauge_geometry import GaugeGeometry, project_box_fiber_q
from train_fiber_residual_phase_gan import predict_all
from train_vqae_centered_residual_adapter import prepare_split


LABELS: dict[int, str] = {2: "car", 3: "cat", 5: "dog", 6: "horse"}
RATES = ("05", "10")
METRICS = ("psnr", "ssim", "lpips")
REQUIRED_PARAMETERS = {
    "filter_mode": "highpass",
    "cutoff": 0.12,
    "transition": 0.03,
    "alpha": 0.5,
    "exact_projection_iterations": 4096,
}

# Round59 deliberately changes exactly these three frozen implementation files
# to make the final raw-bucket target explicit.  They cannot be compared to the
# Round52 code hashes, but they must be clean files at the checked-out current
# Git revision and their current byte hashes are recorded in provenance.
POST_FREEZE_CODE_EXCEPTIONS = {
    "diagnose_fiber_orthogonal_highpass_innovation.py": "Round59 adds the explicit raw_y final-target option and certificate.",
    "src/gauge_geometry.py": "Round59 exposes the raw-bucket residual certificate used by raw_y projection.",
    "train_vqae_centered_residual_adapter.py": "Round59 carries cached raw_y through the prepared split.",
    "run_frozen_fohi_heldout_once.py": "Post-freeze legacy caller explicitly pins legacy_clipped_anchor; it is not invoked by the Round60 gallery.",
}
RAW_Y_INTENTIONALLY_CHANGED_PATHS = frozenset(POST_FREEZE_CODE_EXCEPTIONS)

# These files execute in ``reconstruct_selected`` (directly or through its
# imports).  Round60 must use exactly the code bytes attested by the completed
# Round59 raw-y lane, not merely a source tree with a matching Git label.
ROUND59_RECONSTRUCTION_CORE = (
    "diagnose_fiber_orthogonal_highpass_innovation.py",
    "diagnose_fiber_residual_frequency_fusion.py",
    "gan_high_quality_gi.py",
    "src/fiber_orthogonal_innovation.py",
    "src/gauge_geometry.py",
    "train_fiber_residual_phase_gan.py",
    "train_vqae_centered_residual_adapter.py",
)


@dataclass(frozen=True)
class SelectedRecord:
    label: int
    class_name: str
    source_index: int
    raw_sha256: str
    local_index_rate05: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}_MISMATCH:{actual!r}:{expected!r}")


def require_frozen_parameters(manifest: dict[str, Any]) -> None:
    require_equal(
        manifest.get("status"),
        "VQGAN_GUIDED_FOHI_HELDOUT_FROZEN",
        "FREEZE_STATUS",
    )
    parameters = manifest.get("method_parameters", {})
    for key, expected in REQUIRED_PARAMETERS.items():
        actual = parameters.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 1e-12:
                raise RuntimeError(f"FROZEN_PARAMETER_MISMATCH:{key}:{actual!r}:{expected!r}")
        else:
            require_equal(actual, expected, f"FROZEN_PARAMETER_{key.upper()}")


def git_head(repo_root: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def git_file_is_clean(repo_root: Path, relative: Path) -> bool:
    import subprocess

    return subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=repo_root, check=False
    ).returncode == 0


def verify_frozen_code(repo_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify unchanged frozen code and explicitly attest raw-y changes.

    A Round52 hash mismatch is never silently accepted.  The explicit Round59
    raw-y changes and the post-freeze legacy caller guard
    are the only exceptions; each is required to be clean at current ``HEAD``
    and is recorded with both frozen and current hashes.
    """

    unchanged_verified: dict[str, str] = {}
    raw_y_changes: dict[str, dict[str, str]] = {}
    marker = Path("/content/GI_GAN")
    head = git_head(repo_root)
    for raw_path, expected in sorted(manifest["code_sha256"].items()):
        frozen_path = Path(raw_path)
        try:
            relative = frozen_path.relative_to(marker)
        except ValueError as exc:
            raise RuntimeError(f"UNMAPPABLE_FROZEN_CODE_PATH:{frozen_path}") from exc
        local = repo_root / relative
        if not local.is_file():
            raise FileNotFoundError(f"FROZEN_CODE_MISSING:{local}")
        actual = sha256(local)
        relative_text = str(relative).replace("\\", "/")
        if relative_text in RAW_Y_INTENTIONALLY_CHANGED_PATHS:
            if not git_file_is_clean(repo_root, relative):
                raise RuntimeError(f"RAW_Y_CODE_NOT_CLEAN_AT_HEAD:{local}:{head}")
            raw_y_changes[relative_text] = {
                "frozen_round52_sha256": expected,
                "current_sha256": actual,
                "git_head": head,
                "reason": POST_FREEZE_CODE_EXCEPTIONS[relative_text],
            }
        elif actual != expected:
            raise RuntimeError(f"FROZEN_CODE_HASH_MISMATCH:{local}:{actual}:{expected}")
        else:
            unchanged_verified[str(local)] = actual
    observed = set(raw_y_changes)
    if observed != RAW_Y_INTENTIONALLY_CHANGED_PATHS:
        raise RuntimeError(f"RAW_Y_CHANGE_SET_MISMATCH:{sorted(observed)}")
    return {
        "unchanged_round52_code_sha256": unchanged_verified,
        "intentional_round59_raw_y_code": raw_y_changes,
        "current_git_head": head,
    }


def verify_round59_reconstruction_core(
    repo_root: Path, round59_complete: dict[str, Any]
) -> dict[str, str]:
    """Require the actual Round60 reconstruction imports to match Round59.

    The checked lane receipt is the execution identity.  A matching Git branch
    is insufficient because local byte changes could otherwise alter a gallery
    after the held-out raw-y evaluation.
    """

    expected_hashes = round59_complete.get("code_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("ROUND59_CORE_CODE_RECEIPT_MISSING")
    verified: dict[str, str] = {}
    for relative_text in ROUND59_RECONSTRUCTION_CORE:
        expected = expected_hashes.get(relative_text)
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"ROUND59_CORE_CODE_HASH_MISSING:{relative_text}")
        path = repo_root / relative_text
        if not path.is_file():
            raise FileNotFoundError(f"ROUND59_CORE_CODE_MISSING:{path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"ROUND59_CORE_CODE_HASH_MISMATCH:{relative_text}:{actual}:{expected}"
            )
        verified[relative_text] = actual
    return verified


def verify_artifacts(manifest: dict[str, Any], lane_index: int) -> dict[str, str]:
    lane = manifest["lanes"].get(str(lane_index))
    if lane is None:
        raise RuntimeError(f"UNFROZEN_LANE:{lane_index}")
    verified: dict[str, str] = {}
    for raw_path, expected in sorted(lane["artifact_sha256"].items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"FROZEN_ARTIFACT_MISSING:{path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"FROZEN_ARTIFACT_HASH_MISMATCH:{path}:{actual}:{expected}")
        verified[str(path)] = actual
    return verified


def select_records_rate05(test_samples: Iterable[dict[str, Any]]) -> list[SelectedRecord]:
    """Apply the declared class/minimum-index rule without accepting metrics."""

    rows = list(test_samples)
    selected: list[SelectedRecord] = []
    for label, class_name in LABELS.items():
        candidates = [row for row in rows if int(row["label"]) == label]
        if not candidates:
            raise RuntimeError(f"SELECTION_LABEL_ABSENT:{label}")
        chosen = min(candidates, key=lambda row: int(row["source_index"]))
        matching_indices = [
            index
            for index, row in enumerate(rows)
            if int(row["source_index"]) == int(chosen["source_index"])
            and str(row["raw_sha256"]) == str(chosen["raw_sha256"])
            and int(row["label"]) == label
        ]
        if len(matching_indices) != 1:
            raise RuntimeError(f"SELECTION_NOT_UNIQUE:{label}:{matching_indices}")
        selected.append(
            SelectedRecord(
                label=label,
                class_name=class_name,
                source_index=int(chosen["source_index"]),
                raw_sha256=str(chosen["raw_sha256"]),
                local_index_rate05=matching_indices[0],
            )
        )
    return selected


def local_indices_for_rate(
    selected: list[SelectedRecord], test_samples: Iterable[dict[str, Any]]
) -> list[int]:
    rows = list(test_samples)
    positions: list[int] = []
    for record in selected:
        matched = [
            index
            for index, row in enumerate(rows)
            if int(row["source_index"]) == record.source_index
            and int(row["label"]) == record.label
            and str(row["raw_sha256"]) == record.raw_sha256
        ]
        if len(matched) != 1:
            raise RuntimeError(f"RATE_SAMPLE_IDENTITY_MISMATCH:{record.source_index}:{matched}")
        positions.append(matched[0])
    return positions


def load_metric_rows(
    vector_path: Path,
    selected: list[SelectedRecord],
    local_indices: list[int],
    rate: str,
) -> list[dict[str, Any]]:
    with np.load(vector_path) as vectors:
        required = {f"{arm}_{metric}" for arm in ("structural", "fixed", "fohi") for metric in METRICS}
        absent = required.difference(vectors.files)
        if absent:
            raise RuntimeError(f"METRIC_VECTOR_KEYS_MISSING:{sorted(absent)}")
        length = len(vectors["structural_psnr"])
        rows: list[dict[str, Any]] = []
        for record, local_index in zip(selected, local_indices, strict=True):
            if not 0 <= local_index < length:
                raise RuntimeError(f"METRIC_LOCAL_INDEX_OUT_OF_RANGE:{local_index}:{length}")
            row: dict[str, Any] = {
                "rate": str(rate),
                "label": record.label,
                "class_name": record.class_name,
                "source_index": record.source_index,
                "raw_sha256": record.raw_sha256,
                "local_index": int(local_index),
            }
            for arm in ("structural", "fixed", "fohi"):
                for metric in METRICS:
                    value = float(vectors[f"{arm}_{metric}"][local_index])
                    if not np.isfinite(value):
                        raise RuntimeError(f"METRIC_NOT_FINITE:{rate}:{record.source_index}:{arm}:{metric}")
                    row[f"{arm}_{metric}"] = value
            rows.append(row)
    return rows


def subset_cache(pack: dict[str, Any], local_indices: list[int]) -> dict[str, Any]:
    required = {"truth", "x0", "x_A", "x_G", "y", "source_index", "label"}
    absent = required.difference(pack)
    if absent:
        raise RuntimeError(f"CACHE_KEYS_MISSING:{sorted(absent)}")
    index = torch.tensor(local_indices, dtype=torch.long)
    return {
        key: value.index_select(0, index) if isinstance(value, torch.Tensor) and value.ndim >= 1 else value
        for key, value in pack.items()
    }


def raw_fiber_intrinsic(pack: dict[str, Any], geometry: GaugeGeometry) -> torch.Tensor:
    """Return the final projection target from raw cached bucket measurements.

    ``x0`` is retained as a model input only.  It is never used as the terminal
    measurement target in this corrected qualitative reconstruction.
    """

    if "y" not in pack or not isinstance(pack["y"], torch.Tensor):
        raise RuntimeError("RAW_CACHED_BUCKET_VECTOR_MISSING")
    y = pack["y"].to(device=geometry.Q.device)
    if y.ndim != 2 or y.shape[1] != geometry.m:
        raise RuntimeError(f"RAW_CACHED_BUCKET_SHAPE_MISMATCH:{tuple(y.shape)}:{geometry.m}")
    intrinsic = geometry.intrinsic_record(y)
    if not bool(torch.isfinite(intrinsic).all()):
        raise RuntimeError("RAW_CACHED_BUCKET_INTRINSIC_NOT_FINITE")
    return intrinsic


def projection_with_certificates(
    proposal: torch.Tensor,
    intrinsic: torch.Tensor,
    raw_y: torch.Tensor,
    geometry: GaugeGeometry,
    selected: list[SelectedRecord],
    *,
    exact_iterations: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images: list[torch.Tensor] = []
    certificates: list[dict[str, Any]] = []
    for batch_index, record in enumerate(selected):
        result = project_box_fiber_q(
            proposal[batch_index : batch_index + 1].flatten(1),
            intrinsic[batch_index : batch_index + 1],
            geometry,
            exact=True,
            exact_iterations=int(exact_iterations),
            record_tolerance=1.0e-7,
            step_tolerance=1.0e-8,
        )
        raw_measurement_residual = geometry.raw_measurement_residual_certificate(
            result.image_flat,
            raw_y[batch_index : batch_index + 1].to(geometry.Q.device),
        )
        certificate = {
            "source_index": record.source_index,
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "max_relative_record_error": float(result.max_relative_record_error),
            "max_box_violation": float(result.max_box_violation),
            "max_step_change": float(result.max_step_change),
            "raw_measurement_residual": raw_measurement_residual,
            "passed": bool(
                result.converged
                and result.max_relative_record_error <= 1.0e-7
                and result.max_box_violation <= 1.0e-12
                and raw_measurement_residual["passed"]
            ),
        }
        if not certificate["passed"]:
            raise RuntimeError(f"PROJECTION_CERTIFICATE_FAILED:{certificate}")
        images.append(result.image_flat.reshape_as(proposal[batch_index : batch_index + 1]).float())
        certificates.append(certificate)
    return torch.cat(images, dim=0), certificates


def reconstruct_selected(
    *,
    cache_path: Path,
    config_path: Path,
    control_checkpoint: Path,
    proposal_checkpoint: Path,
    selected: list[SelectedRecord],
    local_indices: list[int],
    params: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows_np, operator_manifest = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    geometry = GaugeGeometry.from_rows_qr(torch.from_numpy(rows_np).to(torch.float64)).to(device)
    if geometry.info.rows_sha256 != operator_manifest["rows_sha256"]:
        raise RuntimeError("OPERATOR_IDENTITY_MISMATCH")

    full_pack = torch.load(cache_path, map_location="cpu", weights_only=False)
    pack = subset_cache(full_pack, local_indices)
    del full_pack
    for position, record in enumerate(selected):
        if int(pack["source_index"][position]) != record.source_index or int(pack["label"][position]) != record.label:
            raise RuntimeError(f"CACHE_RECORD_IDENTITY_MISMATCH:{record.source_index}")

    gan_split = prepare_split(pack, pack, geometry, arm="gan", batch_size=len(selected), device=device)
    control_split = prepare_split(pack, pack, geometry, arm="vqae_control", batch_size=len(selected), device=device)
    for key in ("truth", "anchor", "base", "intrinsic", "source_index"):
        if not torch.equal(gan_split[key], control_split[key]):
            raise RuntimeError(f"PREPARED_SPLIT_MISMATCH:{key}")
    indices = torch.arange(len(selected))
    control_model, control_manifest = load_generator(control_checkpoint, device)
    proposal_model, proposal_manifest = load_generator(proposal_checkpoint, device)
    control_prediction, control_correction, control_model_audit = predict_all(
        control_model, control_split, geometry, indices=indices, batch_size=len(selected), device=device
    )
    proposal_split = gan_split if proposal_manifest["source_arm"] == "gan" else control_split
    _, proposal_correction, proposal_model_audit = predict_all(
        proposal_model, proposal_split, geometry, indices=indices, batch_size=len(selected), device=device
    )
    raw_intrinsic = raw_fiber_intrinsic(pack, geometry)
    structural, structural_certificates = projection_with_certificates(
        control_prediction.to(device),
        raw_intrinsic,
        pack["y"],
        geometry,
        selected,
        exact_iterations=int(params["exact_projection_iterations"]),
    )
    base = gan_split["base"].to(device)
    truth = gan_split["truth"].to(device)
    structural_direction = geometry.null_project_flat(control_correction.to(device).flatten(1))
    difference = geometry.null_project_flat(
        (proposal_correction.to(device) - control_correction.to(device)).flatten(1)
    ).reshape_as(base)
    high_difference = smooth_radial_high_pass(
        difference, cutoff=float(params["cutoff"]), transition=float(params["transition"])
    )
    innovation = geometry.null_project_flat(high_difference.flatten(1))
    orthogonal, beta, orthogonal_audit = fiber_orthogonal_innovation(structural_direction, innovation)
    fixed_proposal = base.flatten(1) + structural_direction + float(params["alpha"]) * innovation
    fohi_proposal = base.flatten(1) + structural_direction + float(params["alpha"]) * orthogonal
    fixed, fixed_certificates = projection_with_certificates(
        fixed_proposal.reshape_as(base),
        raw_intrinsic,
        pack["y"],
        geometry,
        selected,
        exact_iterations=int(params["exact_projection_iterations"]),
    )
    fohi, fohi_certificates = projection_with_certificates(
        fohi_proposal.reshape_as(base),
        raw_intrinsic,
        pack["y"],
        geometry,
        selected,
        exact_iterations=int(params["exact_projection_iterations"]),
    )
    arrays = {
        "truth": truth.detach().cpu().numpy(),
        "structural": structural.detach().cpu().numpy(),
        "fixed": fixed.detach().cpu().numpy(),
        "fohi": fohi.detach().cpu().numpy(),
        # These are the literal intermediate tensors used below to form the
        # FOHI proposal.  They are retained so the mechanism plate never
        # relies on a redrawn, synthetic, or post-hoc surrogate image.
        "vqae_structural_direction": structural_direction.reshape_as(base).detach().cpu().numpy(),
        "vqgan_conditional_detail": difference.detach().cpu().numpy(),
        "highpass_conditional_detail": high_difference.detach().cpu().numpy(),
        "nullspace_highpass_detail": innovation.reshape_as(base).detach().cpu().numpy(),
        "orthogonal_detail": orthogonal.reshape_as(base).detach().cpu().numpy(),
        "fohi_proposal_before_projection": fohi_proposal.reshape_as(base).detach().cpu().numpy(),
    }
    arrays["abs_fohi_truth"] = np.abs(arrays["fohi"] - arrays["truth"])
    certificates = {"structural": structural_certificates, "fixed": fixed_certificates, "fohi": fohi_certificates}
    audit = {
        "operator_sha256": geometry.info.rows_sha256,
        "control_manifest": control_manifest,
        "proposal_manifest": proposal_manifest,
        "control_model_audit": control_model_audit,
        "proposal_model_audit": proposal_model_audit,
        "beta": beta.detach().cpu().flatten().tolist(),
        "parallel_energy_fraction": orthogonal_audit["parallel_energy_fraction"].detach().cpu().tolist(),
        "relative_orthogonality_residual": orthogonal_audit["relative_orthogonality_residual"].detach().cpu().tolist(),
        "final_projection_target": "geometry.intrinsic_record(raw cached y)",
        "raw_cached_y_sha256": tensor_sha256(pack["y"]),
        "raw_cached_y_shape": list(pack["y"].shape),
    }
    torch.cuda.empty_cache()
    return arrays, certificates, audit


def configure_matplotlib() -> str:
    candidates = ["Times New Roman", "Times", "Nimbus Roman No9 L"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((font for font in candidates if font in available), None)
    if chosen is None:
        raise RuntimeError("TIMES_FONT_REQUIRED_FOR_PUBLICATION_FIGURE")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [chosen],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    )
    return chosen


def image2d(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"EXPECTED_SINGLE_CHANNEL_IMAGE:{image.shape}")
    return image


def write_arrays(
    output_dir: Path, rate: str, selected: list[SelectedRecord], arrays: dict[str, np.ndarray]
) -> list[Path]:
    destination = output_dir / f"rate{rate}" / "arrays"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for local_position, record in enumerate(selected):
        prefix = f"source{record.source_index:04d}_{record.class_name}"
        for name, values in arrays.items():
            path = destination / f"{prefix}_{name}.npy"
            np.save(path, values[local_position])
            written.append(path)
    return written


def render_single_images(
    output_dir: Path, rate: str, selected: list[SelectedRecord], arrays: dict[str, np.ndarray]
) -> list[Path]:
    destination = output_dir / f"rate{rate}" / "single_images"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    error_vmax = max(float(np.max(arrays["abs_fohi_truth"])), 1.0e-8)
    for local_position, record in enumerate(selected):
        prefix = f"source{record.source_index:04d}_{record.class_name}"
        for name, values in arrays.items():
            fig, axis = plt.subplots(figsize=(2.0, 2.0))
            if name == "abs_fohi_truth":
                axis.imshow(image2d(values[local_position]), cmap="magma", vmin=0.0, vmax=error_vmax)
            else:
                axis.imshow(image2d(values[local_position]), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_axis_off()
            path = destination / f"{prefix}_{name}.png"
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            written.append(path)
    return written


def render_gallery(
    output_dir: Path,
    rate: str,
    selected: list[SelectedRecord],
    arrays: dict[str, np.ndarray],
) -> list[Path]:
    panels = (
        ("truth", "Ground truth"),
        ("structural", "Structural (raw-y fiber)"),
        ("fixed", "Fixed high-pass (raw-y fiber)"),
        ("fohi", "FOHI (raw-y fiber)"),
        ("abs_fohi_truth", "|FOHI − truth|"),
    )
    fig, axes = plt.subplots(len(selected), len(panels), figsize=(7.16, 5.85))
    fig.subplots_adjust(left=0.095, right=0.995, top=0.92, bottom=0.05, hspace=0.18, wspace=0.08)
    error_vmax = max(float(np.max(arrays["abs_fohi_truth"])), 1.0e-8)
    for row, record in enumerate(selected):
        for column, (key, title) in enumerate(panels):
            axis = axes[row, column]
            if key == "abs_fohi_truth":
                axis.imshow(image2d(arrays[key][row]), cmap="magma", vmin=0.0, vmax=error_vmax)
            else:
                axis.imshow(image2d(arrays[key][row]), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(title, fontsize=7.0, pad=3.0)
        axes[row, 0].text(
            -0.08,
            0.5,
            f"{record.class_name}\n{record.source_index}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.0,
        )
    fig.text(0.01, 0.965, "a" if rate == "05" else "b", fontsize=10.0, fontweight="bold")
    fig.suptitle(
        f"Frozen held-out examples at {int(rate)}% sampling", x=0.095, y=0.975, ha="left", fontsize=8.5, fontweight="bold"
    )
    fig.text(0.095, 0.012, "Rows are selected solely by class and minimum official-test source index; all final images are exactly projected to the raw cached-y fiber.", fontsize=7.0)
    destination = output_dir / f"rate{rate}"
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / f"fohi_qualitative_gallery_rate{rate}"
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _detail_limits(*images: np.ndarray) -> tuple[float, float]:
    """Use a common, data-derived symmetric range for signed detail images."""

    maximum = max(float(np.max(np.abs(image2d(image)))) for image in images)
    maximum = max(maximum, 1.0e-12)
    return -maximum, maximum


def _mechanism_image(
    axis: plt.Axes,
    image: np.ndarray,
    *,
    signed: bool,
    title: str,
    signed_limits: tuple[float, float] | None = None,
) -> None:
    if signed:
        vmin, vmax = signed_limits if signed_limits is not None else _detail_limits(image)
        axis.imshow(image2d(image), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    else:
        axis.imshow(image2d(image), cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title(title, fontsize=7.0, pad=2.5)
    axis.set_axis_off()


def _straight_arrow(
    figure: plt.Figure, start: tuple[float, float], end: tuple[float, float], *, label: str | None = None
) -> None:
    """Add a single straight, annotation-free mechanism connector."""

    arrow = mpl.patches.FancyArrowPatch(
        start,
        end,
        transform=figure.transFigure,
        arrowstyle="-|>",
        mutation_scale=8.0,
        linewidth=0.75,
        color="0.20",
    )
    figure.add_artist(arrow)
    if label is not None:
        figure.text(
            (start[0] + end[0]) / 2.0,
            (start[1] + end[1]) / 2.0 + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=7.0,
        )


def render_mechanism_figures(
    output_dir: Path,
    rate: str,
    selected: list[SelectedRecord],
    arrays: dict[str, np.ndarray],
) -> list[Path]:
    """Render one real-data FOHI mechanism plate for every frozen record.

    The four image panels are deliberately limited to tensors from
    ``reconstruct_selected``: projected structure, the VQGAN conditional
    null-space candidate, its orthogonal residual, and the final projected
    FOHI image.  Connector labels name the actual operations between them.
    """

    required = {"structural", "vqgan_conditional_detail", "orthogonal_detail", "fohi"}
    absent = required.difference(arrays)
    if absent:
        raise RuntimeError(f"MECHANISM_ARRAYS_MISSING:{sorted(absent)}")
    destination = output_dir / f"rate{rate}" / "mechanism"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for position, record in enumerate(selected):
        fig = plt.figure(figsize=(7.16, 2.22), facecolor="white")
        structural_axis = fig.add_axes((0.035, 0.17, 0.205, 0.58))
        candidate_axis = fig.add_axes((0.305, 0.52, 0.165, 0.38))
        orthogonal_axis = fig.add_axes((0.545, 0.52, 0.165, 0.38))
        fohi_axis = fig.add_axes((0.775, 0.17, 0.205, 0.58))
        shared_detail_limits = _detail_limits(
            arrays["vqgan_conditional_detail"][position],
            arrays["orthogonal_detail"][position],
        )
        _mechanism_image(structural_axis, arrays["structural"][position], signed=False, title="VQAE structure")
        _mechanism_image(
            candidate_axis,
            arrays["vqgan_conditional_detail"][position],
            signed=True,
            title="VQGAN detail candidate",
            signed_limits=shared_detail_limits,
        )
        _mechanism_image(
            orthogonal_axis,
            arrays["orthogonal_detail"][position],
            signed=True,
            title="Orthogonal detail",
            signed_limits=shared_detail_limits,
        )
        _mechanism_image(fohi_axis, arrays["fohi"][position], signed=False, title="FOHI output")
        _straight_arrow(
            fig,
            (0.475, 0.71),
            (0.54, 0.71),
            label="high-pass  •  null-space  •  remove structure overlap",
        )
        _straight_arrow(fig, (0.24, 0.46), (0.735, 0.43), label="structure")
        _straight_arrow(fig, (0.71, 0.71), (0.735, 0.45))
        fig.text(0.74, 0.455, "+", ha="center", va="center", fontsize=10.0)
        _straight_arrow(fig, (0.75, 0.43), (0.77, 0.43))
        fig.text(0.76, 0.29, "raw-measurement projection", ha="center", va="top", fontsize=7.0)
        fig.text(0.035, 0.965, "a", fontsize=9.0, fontweight="bold", va="top")
        fig.text(
            0.075,
            0.965,
            f"STL-10 {record.class_name}  |  record {record.source_index}  |  {int(rate)}% sampling",
            fontsize=7.0,
            va="top",
        )
        prefix = f"source{record.source_index:04d}_{record.class_name}_fohi_mechanism"
        png = destination / f"{prefix}.png"
        pdf = destination / f"{prefix}.pdf"
        fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.015)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.015)
        plt.close(fig)
        written.extend((png, pdf))
    return written


def write_metric_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("METRIC_ROWS_REQUIRED")
    path = output_dir / "selected_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def hashes_for_files(paths: Iterable[Path], *, relative_to: Path) -> dict[str, str]:
    return {str(path.relative_to(relative_to)): sha256(path) for path in sorted(paths)}


def output_files(output_dir: Path) -> list[Path]:
    return [path for path in output_dir.rglob("*") if path.is_file() and path.name != "provenance.json"]


def verify_round59_raw_y_diagnostic(
    summary: dict[str, Any], *, rate: str, expected_images: int
) -> None:
    """Refuse anything other than the completed raw-y held-out diagnostic."""

    require_equal(
        summary.get("status"),
        "FIBER_ORTHOGONAL_HIGHPASS_INNOVATION_DIAGNOSTIC",
        f"ROUND59_DIAGNOSTIC_STATUS_RATE{rate}",
    )
    require_equal(summary.get("evaluation_scope"), "heldout", f"ROUND59_DIAGNOSTIC_SCOPE_RATE{rate}")
    require_equal(summary.get("validation_only"), False, f"ROUND59_DIAGNOSTIC_VALIDATION_RATE{rate}")
    require_equal(summary.get("test_split_opened"), True, f"ROUND59_DIAGNOSTIC_TEST_OPEN_RATE{rate}")
    require_equal(summary.get("final_target"), "raw_y", f"ROUND59_FINAL_TARGET_RATE{rate}")
    require_equal(summary.get("evaluation_images"), expected_images, f"ROUND59_DIAGNOSTIC_COUNT_RATE{rate}")
    for key, expected in REQUIRED_PARAMETERS.items():
        diagnostic_key = "exact_iterations" if key == "exact_projection_iterations" else key
        actual = summary.get(diagnostic_key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 1e-12:
                raise RuntimeError(f"ROUND59_DIAGNOSTIC_PARAMETER_MISMATCH:{rate}:{diagnostic_key}:{actual!r}:{expected!r}")
        else:
            require_equal(actual, expected, f"ROUND59_DIAGNOSTIC_PARAMETER_{rate}_{diagnostic_key}")
    for arm in ("structural", "fixed", "fohi"):
        audit = summary.get(f"{arm}_projection_audit")
        if not isinstance(audit, dict) or audit.get("all_converged") is not True:
            raise RuntimeError(f"ROUND59_DIAGNOSTIC_PROJECTION_NOT_CERTIFIED:{rate}:{arm}")
        raw_certificate = summary.get("raw_measurement_residual_certificate", {}).get(arm)
        if not isinstance(raw_certificate, dict) or raw_certificate.get("passed") is not True:
            raise RuntimeError(f"ROUND59_RAW_MEASUREMENT_NOT_CERTIFIED:{rate}:{arm}")


def checked_rate_inputs(
    *,
    source_round56_lane: Path,
    round59_lane: Path,
    rate: str,
    round59_complete: dict[str, Any],
    expected_images: int,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    cache_dir = source_round56_lane / f"rate{rate}" / "cache"
    evaluation_dir = round59_lane / f"rate{rate}" / "fohi"
    cache = cache_dir / "test_cache.pt"
    cache_manifest = cache_dir / "test_cache_manifest.json"
    vectors = evaluation_dir / "metric_vectors.npz"
    summary = evaluation_dir / "summary.json"
    for path in (cache, cache_manifest, vectors, summary):
        if not path.is_file():
            raise FileNotFoundError(f"ROUND59_GALLERY_INPUT_MISSING:{path}")
    rate_receipt = round59_complete["rates"].get(rate)
    if rate_receipt is None:
        raise RuntimeError(f"ROUND59_COMPLETE_RATE_MISSING:{rate}")
    for name, path in (("metric_vectors", vectors), ("summary", summary)):
        expected = rate_receipt.get(f"{name}_sha256")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"ROUND59_RECEIPT_HASH_MISMATCH:{rate}:{name}:{actual}:{expected}")
    reused_cache = rate_receipt.get("reused_cache")
    if not isinstance(reused_cache, dict):
        raise RuntimeError(f"ROUND59_REUSED_CACHE_RECEIPT_MISSING:{rate}")
    for name, path in (("cache", cache), ("cache_manifest", cache_manifest)):
        expected = reused_cache.get(f"{name}_sha256")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"ROUND59_REUSED_CACHE_HASH_MISMATCH:{rate}:{name}:{actual}:{expected}")
    manifest = read_json(cache_manifest)
    if manifest.get("status") != "FROZEN_FOHI_TEST_CACHE_COMPLETE":
        raise RuntimeError(f"ROUND56_CACHE_STATUS_MISMATCH:{rate}")
    if manifest.get("lane_index") != 0 or str(manifest.get("rate")) != rate:
        raise RuntimeError(f"ROUND56_CACHE_SCOPE_MISMATCH:{rate}")
    require_equal(manifest.get("test_images"), expected_images, f"CACHE_IMAGE_COUNT_RATE{rate}")
    if sha256(cache) != manifest.get("cache_sha256"):
        raise RuntimeError(f"ROUND56_CACHE_HASH_MISMATCH:{rate}")
    verify_round59_raw_y_diagnostic(
        read_json(summary), rate=rate, expected_images=expected_images
    )
    return cache, cache_manifest, vectors, summary, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/content/GI_GAN"))
    parser.add_argument(
        "--source-round56-lane",
        type=Path,
        default=Path("/content/gan_r56_heldout_recovery/lane0"),
        help="Read-only source of the hash-pinned raw cache.",
    )
    parser.add_argument(
        "--round59-lane",
        type=Path,
        default=Path("/content/gan_r59_raw_fiber/lane0"),
        help="Completed raw-y evaluation receipt, summaries, and metric vectors.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/content/gan_r60_qualitative_gallery/lane0"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"REPO_ROOT_MISSING:{repo_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"OUTPUT_DIR_NOT_EMPTY:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(args.freeze_manifest)
    require_frozen_parameters(manifest)
    frozen_code = verify_frozen_code(repo_root, manifest)
    frozen_artifacts = verify_artifacts(manifest, lane_index=0)
    round59_complete_path = args.round59_lane / "ROUND59_COMPLETE.json"
    round59_complete = read_json(round59_complete_path)
    require_equal(round59_complete.get("status"), "ROUND59_RAW_FIBER_LANE_COMPLETE", "ROUND59_COMPLETE_STATUS")
    require_equal(round59_complete.get("lane_index"), 0, "ROUND59_COMPLETE_LANE")
    require_equal(round59_complete.get("final_target"), "raw_y", "ROUND59_COMPLETE_FINAL_TARGET")
    round59_core_code = verify_round59_reconstruction_core(repo_root, round59_complete)

    per_rate: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {
        "freeze_manifest": sha256(args.freeze_manifest),
        "gallery_script": sha256(Path(__file__).resolve()),
        "round59_complete_receipt": sha256(round59_complete_path),
    }
    for rate in RATES:
        cache, cache_manifest_path, vectors, summary, cache_manifest = checked_rate_inputs(
            source_round56_lane=args.source_round56_lane,
            round59_lane=args.round59_lane,
            rate=rate,
            round59_complete=round59_complete,
            expected_images=int(manifest["expected_test_images"]),
        )
        per_rate[rate] = {
            "cache": cache,
            "cache_manifest": cache_manifest_path,
            "cache_manifest_payload": cache_manifest,
            "vectors": vectors,
            "summary": summary,
        }
        input_hashes.update(
            {
                f"rate{rate}_cache": sha256(cache),
                f"rate{rate}_cache_manifest": sha256(cache_manifest_path),
                f"rate{rate}_metric_vectors": sha256(vectors),
                f"rate{rate}_summary": sha256(summary),
            }
        )

    selected = select_records_rate05(per_rate["05"]["cache_manifest_payload"]["test_samples"])
    all_certificates: dict[str, Any] = {}
    mechanism_intermediate_hashes: dict[str, dict[str, str]] = {}
    selected_local_indices: dict[str, list[int]] = {}
    metric_rows: list[dict[str, Any]] = []
    figure_font = configure_matplotlib()
    written: list[Path] = []
    lane = manifest["lanes"]["0"]
    for rate in RATES:
        cache_manifest = per_rate[rate]["cache_manifest_payload"]
        local_indices = local_indices_for_rate(selected, cache_manifest["test_samples"])
        selected_local_indices[rate] = local_indices
        metric_rows.extend(load_metric_rows(per_rate[rate]["vectors"], selected, local_indices, rate))
        frozen_rate = lane["rates"][rate]
        arrays, certificates, diagnostic_audit = reconstruct_selected(
            cache_path=per_rate[rate]["cache"],
            config_path=Path(frozen_rate["config"]),
            control_checkpoint=Path(frozen_rate["structural_checkpoint"]),
            proposal_checkpoint=Path(frozen_rate["proposal_checkpoint"]),
            selected=selected,
            local_indices=local_indices,
            params=manifest["method_parameters"],
        )
        all_certificates[rate] = certificates
        written.extend(write_arrays(output_dir, rate, selected, arrays))
        written.extend(render_single_images(output_dir, rate, selected, arrays))
        written.extend(render_gallery(output_dir, rate, selected, arrays))
        written.extend(render_mechanism_figures(output_dir, rate, selected, arrays))
        mechanism_intermediate_hashes[rate] = {
            str(path.relative_to(output_dir)): sha256(path)
            for path in sorted((output_dir / f"rate{rate}" / "arrays").glob("*.npy"))
            if any(
                marker in path.name
                for marker in (
                    "vqae_structural_direction",
                    "vqgan_conditional_detail",
                    "highpass_conditional_detail",
                    "nullspace_highpass_detail",
                    "orthogonal_detail",
                    "fohi_proposal_before_projection",
                )
            )
        }
        write_json(output_dir / f"rate{rate}" / "diagnostic_audit.json", diagnostic_audit)
        written.append(output_dir / f"rate{rate}" / "diagnostic_audit.json")
    selection_payload = {
        "rule": "For each of STL-10 labels 2 (car), 3 (cat), 5 (dog), and 6 (horse), select the smallest source_index among lane0 rate05 cache-manifest test_samples. No image-quality metric, reconstruction, or truth comparison enters selection.",
        "selected_records": [asdict(record) for record in selected],
        "local_indices_by_rate": selected_local_indices,
    }
    selection_path = output_dir / "selection.json"
    write_json(selection_path, selection_payload)
    written.append(selection_path)
    certificates_path = output_dir / "projection_certificates.json"
    write_json(certificates_path, all_certificates)
    written.append(certificates_path)
    selected_metrics_path = write_metric_csv(output_dir, metric_rows)
    written.append(selected_metrics_path)

    provenance = {
        "status": "FROZEN_FOHI_HELDOUT_QUALITATIVE_GALLERY_COMPLETE",
        "purpose": "Metric-blind qualitative visualisation of four pre-specified held-out lane-0 records; no training, tuning, or selection by quality metric.",
        "selection": selection_payload,
        "frozen_parameters": manifest["method_parameters"],
        "font": {"family": figure_font, "minimum_point_size": 7.0},
        "terminal_projection": {
            "target": "geometry.intrinsic_record(raw cached y)",
            "clipped_anchor_x0_role": "model input only; never a final projection target",
            "metric_source": "Round59 raw-y metric vectors, hash-checked against ROUND59_COMPLETE.json. No Round56 clipped-anchor vectors are read or reported.",
        },
        "frozen_code_verification": frozen_code,
        "round59_reconstruction_core_sha256": round59_core_code,
        "frozen_artifact_sha256": frozen_artifacts,
        "input_sha256": input_hashes,
        "projection_certificates": "projection_certificates.json",
        "mechanism_figure": {
            "claim": "VQGAN conditional detail is null-space filtered, high-pass filtered, and stripped of its VQAE-structural parallel component before addition to the structural reconstruction; the result is projected to the raw cached-y fiber.",
            "intermediate_array_sha256": mechanism_intermediate_hashes,
            "rendered_records": "One PNG and PDF mechanism plate per pre-specified car, cat, dog, and horse record at each rate; records are never chosen by metrics or visual quality.",
        },
        "metrics": "selected_metrics.csv contains only the four pre-specified records at each rate, indexed from hash-checked Round59 raw-y metric vectors. It is descriptive and does not replace the full held-out inference.",
        "output_sha256_excluding_provenance": hashes_for_files(output_files(output_dir), relative_to=output_dir),
        "no_post_test_tuning": True,
    }
    provenance_path = output_dir / "provenance.json"
    write_json(provenance_path, provenance)
    sidecar = output_dir / "provenance.json.sha256"
    sidecar.write_text(f"{sha256(provenance_path)}  provenance.json\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "selected": selection_payload, "provenance": str(provenance_path)}, indent=2))


if __name__ == "__main__":
    main()
