from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from src.compatibility_data import compute_train_normalization, load_rad5_96_components, normalize_images, save_json, write_csv
from src.compatibility_model import CompatibilityCritic
from src.metrics import batch_metrics
from src.projections import exact_data_anchor, exact_null_project, relative_measurement_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-generator candidate selection using a Phase-1 compatibility critic.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--e1_report", required=True, help="gate_report_e1.json from train_compatibility.py.")
    parser.add_argument("--critic_checkpoint", required=True, help="Compatibility critic checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Candidate-selection output directory.")
    parser.add_argument("--device", default=None, help="Override config device.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def load_e1_gate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_generator(rad_config: dict[str, Any], measurement, device: torch.device, checkpoint: str | None):
    from src import phase78_96px_rad5_one_seed_probe as p78

    if checkpoint:
        return p78.load_probe_checkpoint_for_eval(Path(checkpoint), rad_config, measurement, device)
    return p78.load_generator_96(rad_config, measurement, device, train=False)


@torch.no_grad()
def generate_candidates(generator, measurement, rad_config: dict[str, Any], y: torch.Tensor, *, k: int, seed: int, device: torch.device):
    from src.phase79_rad5_rowspace_diversity_diagnostic import forward_with_noise

    gen = torch.Generator(device=device).manual_seed(int(seed))
    y_rep = y.repeat(int(k), 1).to(device)
    x_data_flat = measurement.data_solution(y_rep.float(), rad_config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    noise = torch.randn(x_data.shape, device=device, dtype=x_data.dtype, generator=gen)
    out = forward_with_noise(generator, measurement, y_rep, noise, rad_config)
    return out["x_hat_flat"].detach()


@torch.no_grad()
def score_candidates(critic, measurement, y: torch.Tensor, cand_flat: torch.Tensor, normalization: dict[str, float], device: torch.device) -> torch.Tensor:
    r_y = exact_data_anchor(y.to(device), measurement, device=device, as_image=False).repeat(cand_flat.shape[0], 1)
    n = exact_null_project(cand_flat.to(device), measurement, dtype=torch.float64, device=device)
    r_img = normalize_images(r_y.detach().cpu(), img_size=measurement.img_size, key="r", normalization=normalization).to(device)
    n_img = normalize_images(n.detach().cpu(), img_size=measurement.img_size, key="n", normalization=normalization).to(device)
    return critic.score_pairs(r_img, n_img).detach().cpu()


def p0_rmse(pred_flat: torch.Tensor, x_flat: torch.Tensor, measurement, device: torch.device) -> torch.Tensor:
    p = exact_null_project(pred_flat.to(device), measurement, dtype=torch.float64, device=device)
    t = exact_null_project(x_flat.to(device), measurement, dtype=torch.float64, device=device)
    return torch.sqrt(torch.mean((p - t) ** 2, dim=1)).detach().cpu()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    methods = sorted({str(r["method"]) for r in rows})
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        for metric in ["p0_rmse", "psnr", "ssim", "relmeaserr"]:
            vals = torch.tensor([float(r[metric]) for r in sub], dtype=torch.float32)
            out[f"{method}_{metric}_mean"] = float(vals.mean().item())
    return out


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.device:
        cfg["device"] = args.device
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.txt").write_text("$ " + " ".join(sys.argv) + "\n", encoding="utf-8")
    e1 = load_e1_gate(Path(args.e1_report))
    if not bool(e1.get("allowed_to_run_e2_candidate_selection", False)):
        report = {
            "phase": "E2_candidate_selection_rad5_96",
            "status": "skipped",
            "reason": "E1 gate did not pass; frozen-generator selection pilot not run.",
            "e1_report": str(args.e1_report),
        }
        save_json(out / "gate_report_e2.json", report)
        print(json.dumps({"gate_report_e2": str(out / "gate_report_e2.json"), "status": "skipped"}, indent=2))
        return

    device = resolve_device(str(cfg.get("device", "cuda")))
    measurement, rad_config, splits, _split_info = load_rad5_96_components(cfg, output_dir=out, device=device)
    payload = torch.load(args.critic_checkpoint, map_location=device, weights_only=False)
    normalization = payload.get("normalization") or compute_train_normalization(splits["train"])
    critic = CompatibilityCritic(
        embed_dim=int(cfg.get("embed_dim", 128)),
        base_channels=int(cfg.get("base_channels", 24)),
        temperature=float(cfg.get("temperature", 0.07)),
        learn_temperature=bool(cfg.get("learn_temperature", False)),
        use_joint_mlp=bool(cfg.get("use_joint_mlp", False)),
    ).to(device)
    critic.load_state_dict(payload["model"], strict=True)
    critic.eval()
    generator = load_generator(rad_config, measurement, device, cfg.get("candidate_generator_checkpoint"))
    generator.eval()
    ks = [int(k) for k in cfg.get("candidate_k_values", [1, 4, 8, 16])]
    max_images = min(int(cfg.get("candidate_max_images", 32)), splits["test"].size)
    rows: list[dict[str, Any]] = []
    for i in range(max_images):
        y = splits["test"].y[i : i + 1].to(device)
        x = splits["test"].x[i : i + 1].to(device)
        x_flat = x.reshape(1, -1)
        for k in ks:
            cand = generate_candidates(generator, measurement, rad_config, y, k=k, seed=int(cfg.get("seed", 1)) + 1000 + i * 37 + k, device=device)
            scores = score_candidates(critic, measurement, y, cand, normalization, device)
            p0e = p0_rmse(cand, x_flat.repeat(k, 1), measurement, device)
            selected = int(torch.argmax(scores).item())
            oracle = int(torch.argmin(p0e).item())
            random_idx = 0
            mean_flat = cand.mean(dim=0, keepdim=True)
            methods = {
                "random_candidate": cand[random_idx : random_idx + 1],
                "posterior_mean": mean_flat,
                "critic_selected": cand[selected : selected + 1],
                "oracle_best_of_k": cand[oracle : oracle + 1],
            }
            if k == 1:
                methods["deterministic_or_single"] = cand[:1]
            for method, pred_flat in methods.items():
                pred_img = measurement.unflatten_img(pred_flat).clamp(0, 1)
                bm = batch_metrics(pred_img, x.clamp(0, 1), measurement, y)
                rel = relative_measurement_error(pred_flat, y, measurement)
                rmse = p0_rmse(pred_flat, x_flat, measurement, device)
                rows.append(
                    {
                        "sample_ordinal": i,
                        "source_index": int(splits["test"].source_indices[i].item()),
                        "K": k,
                        "method": method,
                        "selected_index": selected if method == "critic_selected" else "",
                        "oracle_index": oracle,
                        "critic_score_selected": float(scores[selected].item()),
                        "p0_rmse": float(rmse[0].item()),
                        "psnr": float(bm["psnr"]),
                        "ssim": float(bm["ssim"]),
                        "lpips": "[DATA MISSING]",
                        "rapsd": "[DATA MISSING]",
                        "relmeaserr": float(rel[0].detach().cpu().item()),
                    }
                )
    write_csv(out / "candidate_selection_per_image.csv", rows)
    report = {
        "phase": "E2_candidate_selection_rad5_96",
        "status": "complete",
        "e1_report": str(args.e1_report),
        "critic_checkpoint": str(args.critic_checkpoint),
        "generator_frozen": True,
        "generator_loss_modified": False,
        "measurement": {"img_size": measurement.img_size, "m": measurement.m, "n": measurement.n},
        "summary": summarize(rows),
        "notes": [
            "Critic selection used r_y=exact_data_anchor(y) and n_k=exact_null_project(candidate).",
            "LPIPS/RAPSD are marked DATA MISSING in this pilot unless a later metric package is wired.",
        ],
    }
    save_json(out / "gate_report_e2.json", report)
    print(json.dumps({"gate_report_e2": str(out / "gate_report_e2.json"), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
