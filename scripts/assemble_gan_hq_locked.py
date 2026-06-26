from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gan_high_quality_gi as hq


def load_model(checkpoint: Path, config: dict[str, Any], measurement: hq.GhostMeasurementOperator, device: torch.device) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model_cfg = dict(config["model"])
    model = hq.build_generator(
        {"model_type": model_cfg.get("model_type", "hq_unet"), "base_channels": int(model_cfg.get("base_channels", 32))},
        measurement=measurement,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("generator_ema") or payload.get("generator")
    if state is None:
        raise KeyError(f"Checkpoint has no generator_ema/generator state: {checkpoint}")
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble locked GAN-HQ-GI reports from frozen final checkpoints.")
    parser.add_argument("--config", default="configs/compatibility/gan_high_quality_gi_locked_64_5pct.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variants", default="no_gan,gan")
    parser.add_argument("--train-seeds", default="0,1,2")
    args = parser.parse_args()

    started = time.time()
    config_path = Path(args.config)
    config = hq.load_yaml(config_path)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    hq.set_seed(int(config.get("seed", 20260625)))
    out = hq.ensure_dir(hq.ROOT / str(args.output_dir or config["output_dir"]))
    reports = hq.ensure_dir(out / "reports")
    shutil.copyfile(config_path, out / "config_used.yaml")

    train_ds, _val_ds, dev_ds, split_manifest = hq.build_split_datasets(config)
    split_audit = hq.save_split_hash_audit(reports / "sample_hash_audit.csv", {"train": train_ds, "dev": dev_ds})
    train_x, _train_labels, _train_indices = hq.tensor_dataset_to_matrix(train_ds)
    rows, op_meta = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    measurement = hq.make_measurement_operator(
        rows,
        img_size=int(config["data"]["img_size"]),
        device=device,
        lambda_solver=float(config["operator"].get("lambda_solver", 1e-8)),
    )
    lmmse = hq.EmpiricalLMMSE.fit(train_x, rows, lambda_=float(config["operator"].get("lmmse_lambda", 1e-4)))
    dev_loader = hq.build_loader(
        dev_ds,
        batch_size=int(config["data"]["batch_size"]),
        workers=int(config["data"].get("num_workers", 0)),
        shuffle=False,
        seed=int(config["seed"]) + 1,
        device=device,
    )

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    train_seeds = [int(v.strip()) for v in str(args.train_seeds).split(",") if v.strip()]
    all_method: list[dict[str, Any]] = []
    all_per: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for seed in train_seeds:
        for variant in variants:
            run_dir = out / "runs" / f"{variant}_seed{seed}"
            ckpt = run_dir / "checkpoints" / f"{variant}_seed{seed}_final.pt"
            model = load_model(ckpt, config, measurement, device)
            method_rows, per_rows, eval_diag = hq.evaluate_methods(
                methods={variant: model},
                lmmse=lmmse,
                measurement=measurement,
                loader=dev_loader,
                device=device,
                config=config,
                output_dir=run_dir,
                epoch_tag=f"{variant}_seed{seed}_locked_final",
            )
            for row in method_rows:
                row["train_seed"] = int(seed)
            for row in per_rows:
                row["train_seed"] = int(seed)
            all_method.extend(method_rows)
            all_per.extend(per_rows)
            manifests.append(
                {
                    "variant": variant,
                    "train_seed": int(seed),
                    "checkpoint": str(ckpt),
                    "checkpoint_sha256": hq.sha256_file(ckpt),
                    "eval_diag": eval_diag,
                }
            )

    hq.write_csv(reports / "method_metrics.csv", all_method)
    hq.write_csv(reports / "per_image_metrics.csv", all_per)
    gate = hq.summarize_gate(all_per, all_method, config)
    gate["protocol"] = {
        "status": "locked_once_assembled_from_frozen_checkpoints",
        "config": str(config_path),
        "device": str(device),
        "variants": variants,
        "train_seeds": train_seeds,
    }
    hq.write_json(reports / "gate_report.json", gate)
    hq.write_json(reports / "operator_manifest.json", op_meta)
    hq.write_json(reports / "split_manifest.json", split_manifest)
    hq.write_json(reports / "duplicate_audit.json", split_audit)
    hq.write_json(reports / "lmmse_manifest.json", {"lambda": lmmse.lambda_, "rows_sha256": lmmse.rows_sha256, "train_count": int(train_x.shape[0])})
    hq.write_json(reports / "assembled_checkpoint_manifest.json", manifests)
    hq.write_math_and_ledger(reports, gate)

    comparisons = gate.get("comparisons", [])
    lines = [
        "# Locked 64x64 5% GAN High-Quality GI Report",
        "",
        f"Classification: `{gate['classification']}`",
        f"Locked-test authorized: `{gate['locked_test_authorized']}`",
        f"Output: `{out}`",
        "",
        "## Paired Comparisons",
        "",
        "| metric | delta GAN-noGAN | 95% CI | status | pairs |",
        "|---|---:|---:|---|---:|",
    ]
    for comp in comparisons:
        if comp.get("status") not in {"OK", "PASS"}:
            lines.append(f"| {comp.get('metric')} | [DATA MISSING] | [DATA MISSING] | {comp.get('status')} | 0 |")
            continue
        lines.append(
            "| {metric} | {delta:.6g} | [{lo:.6g}, {hi:.6g}] | {status} | {n} |".format(
                metric=comp["metric"],
                delta=float(comp["mean_delta"]),
                lo=float(comp.get("ci95_low", comp.get("ci_low"))),
                hi=float(comp.get("ci95_high", comp.get("ci_high"))),
                status=comp["status"],
                n=int(comp.get("n_pairs", comp.get("n", 0))),
            )
        )
    lines.extend(
        [
            "",
            "## Gate Conditions",
            "",
            "```json",
            json.dumps(gate.get("conditions", {}), indent=2),
            "```",
            "",
            "This report was assembled from frozen final EMA checkpoints. It does not retrain or tune any method.",
            "",
        ]
    )
    hq.write_text(reports / "LOCKED_REPORT.md", "\n".join(lines))

    runtime = {
        "status": "PASS",
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "started_utc": hq.now_utc(),
        "config_sha256": hq.sha256_file(out / "config_used.yaml"),
        "method_metrics_sha256": hq.sha256_file(reports / "method_metrics.csv"),
        "per_image_metrics_sha256": hq.sha256_file(reports / "per_image_metrics.csv"),
        "gate_report_sha256": hq.sha256_file(reports / "gate_report.json"),
    }
    hq.write_json(reports / "runtime_and_hashes.json", runtime)
    summary = {
        "status": "GAN_HIGH_QUALITY_GI_LOCKED_ASSEMBLED",
        "output_dir": str(out),
        "classification": gate["classification"],
        "locked_test_authorized": gate["locked_test_authorized"],
        "key_artifacts": {
            "locked_report": str(reports / "LOCKED_REPORT.md"),
            "method_metrics": str(reports / "method_metrics.csv"),
            "per_image_metrics": str(reports / "per_image_metrics.csv"),
            "gate_report": str(reports / "gate_report.json"),
            "claim_evidence_ledger": str(reports / "claim_evidence_ledger.md"),
            "checkpoint_manifest": str(reports / "assembled_checkpoint_manifest.json"),
        },
        "runtime": runtime,
    }
    hq.write_json(reports / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
