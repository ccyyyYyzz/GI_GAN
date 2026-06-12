from __future__ import annotations

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from .phase16_common import CORE_STL_METHODS, PHASE16, ensure_dir, evaluate_model, write_all


OUT = PHASE16 / "classwise"
CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
FIELDS = ["method_id", "class_id", "class_name", "num_samples", "mean_psnr", "mean_ssim", "status", "notes"]


def main() -> None:
    rows = []
    try:
        for method_id in CORE_STL_METHODS:
            _, samples = evaluate_model(method_id, limit=500, collect_per_sample=True, noise_map_mode="fixed")
            grouped = defaultdict(list)
            for item in samples:
                if "label" in item:
                    grouped[int(item["label"])].append(item)
            if not grouped:
                raise RuntimeError("labels not available in STL-10 loader")
            for label, items in sorted(grouped.items()):
                rows.append(
                    {
                        "method_id": method_id,
                        "class_id": label,
                        "class_name": CLASSES[label] if 0 <= label < len(CLASSES) else str(label),
                        "num_samples": len(items),
                        "mean_psnr": float(np.mean([x["psnr"] for x in items])),
                        "mean_ssim": float(np.mean([x["ssim"] for x in items])),
                        "status": "completed",
                        "notes": "limit_eval_samples=500",
                    }
                )
        write_all(OUT / "classwise_stl10_metrics", rows, FIELDS)
        for metric, path in [("mean_psnr", OUT / "classwise_psnr.png"), ("mean_ssim", OUT / "classwise_ssim.png")]:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            for method_id in CORE_STL_METHODS:
                sub = [r for r in rows if r["method_id"] == method_id]
                ax.plot([r["class_name"] for r in sub], [r[metric] for r in sub], marker="o", label=method_id)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, frameon=False)
            plt.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
    except Exception as exc:
        ensure_dir(OUT)
        (OUT / "skipped_no_labels.md").write_text(f"# Classwise skipped\n\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
        rows.append({"method_id": "", "class_id": "", "class_name": "", "num_samples": "", "mean_psnr": "", "mean_ssim": "", "status": "skipped_no_labels", "notes": str(exc)})
        write_all(OUT / "classwise_stl10_metrics", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
