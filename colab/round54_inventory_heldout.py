from pathlib import Path
import hashlib
import json
import subprocess
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


repo = Path("/content/GI_GAN")
with zipfile.ZipFile("/content/gan_rate_bundle.zip") as archive:
    lane_index = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
operator_seed = 772101 + lane_index
labels = {
    0: "seed1_primary_seed2_control",
    1: "seed2_primary_seed0_control",
    2: "seed0_primary_seed1_control",
}
label = labels[lane_index]
five_root = Path("/content/gan_operator_assets")
ten_root = Path(f"/content/gan_rate_bundle_seed{lane_index}")
five_structural = require(
    f"/content/gan_r51_results/operator_seed_{operator_seed}/control/checkpoint_vqae_control_rot0.5_adv0.pt"
)
five_proposal = require(
    f"/content/gan_r51_results/operator_seed_{operator_seed}/gan_adv0/checkpoint_gan_rot0.5_adv0.pt"
)
ten_structural = require(
    f"/content/gan_r46_results/seed{lane_index}/rate10/control/checkpoint_vqae_control_rot0.5_adv0.pt"
)
ten_proposal = require(
    f"/content/gan_r50_results/{label}/rate10_train_gan_adv0/checkpoint_gan_rot0.5_adv0.pt"
)
five_summary = json.loads(
    require(f"/content/gan_r51_results/operator_seed_{operator_seed}/fohi/summary.json").read_text(
        encoding="utf-8"
    )
)
ten_summary = json.loads(
    require(f"/content/gan_r50_results/{label}/rate10_adv0_highpass/summary.json").read_text(
        encoding="utf-8"
    )
)

code_relatives = [
    "run_frozen_fohi_heldout_once.py",
    "prepare_frozen_fohi_test_cache.py",
    "prepare_fiber_rate_caches.py",
    "diagnose_fiber_orthogonal_highpass_innovation.py",
    "anchor_initialized_vqgan_inversion.py",
    "gan_high_quality_gi.py",
    "diagnose_afrb_proposal_headroom.py",
    "diagnose_fiber_residual_frequency_fusion.py",
    "diagnose_vqgan_causal_disagreement_controls.py",
    "train_fiber_residual_phase_gan.py",
    "train_vqae_centered_residual_adapter.py",
    "src/fiber_orthogonal_innovation.py",
    "src/gauge_geometry.py",
    "src/projections.py",
    "src/metrics.py",
]
code_hashes = {str(require(repo / relative)): sha256(repo / relative) for relative in code_relatives}

artifact_paths = []
for bundle_root, rate in ((five_root, "05"), (ten_root, "10")):
    artifact_paths.extend(
        [
            require(bundle_root / f"config_rate{rate}.yaml"),
            require(bundle_root / "priors/vqae.pt"),
            require(bundle_root / "priors/vqgan.pt"),
            require(bundle_root / f"rate{rate}/vqae_refiner.pt"),
            require(bundle_root / f"rate{rate}/vqgan_refiner.pt"),
        ]
    )
artifact_paths.extend([five_structural, five_proposal, ten_structural, ten_proposal])
artifact_hashes = {str(path): sha256(path) for path in artifact_paths}

head = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
).stdout.strip()
status = subprocess.run(
    ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True
).stdout.strip()
payload = {
    "status": "FOHI_HELDOUT_LANE_INVENTORY_COMPLETE",
    "lane_index": lane_index,
    "label": label,
    "operator_seed_05": operator_seed,
    "repo_head": head,
    "repo_dirty": bool(status),
    "test_split_opened": False,
    "code_sha256": code_hashes,
    "artifact_sha256": artifact_hashes,
    "rates": {
        "05": {
            "bundle_root": str(five_root),
            "config": str(five_root / "config_rate05.yaml"),
            "structural_checkpoint": str(five_structural),
            "proposal_checkpoint": str(five_proposal),
            "operator_seed": operator_seed,
            "operator_sha256": five_summary["operator_sha256"],
        },
        "10": {
            "bundle_root": str(ten_root),
            "config": str(ten_root / "config_rate10.yaml"),
            "structural_checkpoint": str(ten_structural),
            "proposal_checkpoint": str(ten_proposal),
            "rate_seed": lane_index,
            "operator_sha256": ten_summary["operator_sha256"],
        },
    },
}
output = Path(f"/content/gan_r54_heldout_inventory_lane{lane_index}.json")
output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
print("WROTE", output)
