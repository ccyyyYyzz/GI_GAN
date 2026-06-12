from __future__ import annotations

import csv
from pathlib import Path

from .phase12_common import PHASE12, copy_if_exists, load_registry, read_csv, write_csv, write_md_table
from .utils import ensure_dir


OUT = PHASE12 / "reconstruction_examples"


def export_for(row: dict) -> list[dict]:
    src_dir = Path(row["eval_metrics_path"]).parent if row.get("eval_metrics_path") else Path(row.get("best_checkpoint_path", "")).parent
    method = row["method_id"]
    dst_dir = ensure_dir(OUT / method)
    manifest = []
    recon = src_dir / "eval_samples" / "recon_grid.png"
    copied = copy_if_exists(recon, dst_dir / "recon_grid.png")
    manifest.append({"method_id": method, "artifact": "recon_grid", "source": str(recon), "target": str(dst_dir / "recon_grid.png"), "exists": copied, "limited_examples": not copied})
    per = src_dir / "eval_samples_individual" / "per_sample_metrics.csv"
    if per.exists():
        rows = read_csv(per)
        scored = []
        for r in rows:
            try:
                scored.append((float(r.get("psnr", r.get("model_psnr", ""))), r))
            except Exception:
                pass
        if scored:
            scored.sort(key=lambda x: x[0])
            picks = {"worst_psnr": scored[0][1], "median_psnr": scored[len(scored) // 2][1], "best_psnr": scored[-1][1]}
            for label, sample in picks.items():
                sample_id = sample.get("sample", sample.get("sample_id", ""))
                sample_id = str(sample_id).replace("sample_", "").zfill(3) if sample_id else "000"
                for suffix in ["gt", "backproj", "recon", "abs_error"]:
                    src = src_dir / "eval_samples_individual" / f"sample_{sample_id}_{suffix}.png"
                    copied = copy_if_exists(src, dst_dir / f"{label}_{suffix}.png")
                    manifest.append({"method_id": method, "artifact": f"{label}_{suffix}", "source": str(src), "target": str(dst_dir / f"{label}_{suffix}.png"), "exists": copied, "limited_examples": False})
        else:
            manifest.append({"method_id": method, "artifact": "per_sample_metrics", "source": str(per), "target": "", "exists": True, "limited_examples": True})
    else:
        manifest.append({"method_id": method, "artifact": "per_sample_metrics", "source": str(per), "target": "", "exists": False, "limited_examples": True})
    return manifest


def main() -> None:
    ensure_dir(OUT)
    rows = [row for row in load_registry() if str(row.get("preferred_for_paper")).lower() == "true"]
    manifest = []
    for row in rows:
        manifest.extend(export_for(row))
    fields = ["method_id", "artifact", "source", "target", "exists", "limited_examples"]
    write_csv(OUT / "examples_manifest.csv", manifest, fields)
    write_md_table(OUT / "examples_manifest.md", manifest, fields)
    print(OUT)


if __name__ == "__main__":
    main()
