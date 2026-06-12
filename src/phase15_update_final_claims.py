from __future__ import annotations

import json

from .phase15_common import PHASE15, ensure_dir, load_registry, numeric, read_csv, threshold_for, write_json


def row_by_id(rows: list[dict[str, str]], method_id: str) -> dict[str, str]:
    return next(row for row in rows if row.get("method_id") == method_id)


def metric(row: dict[str, str]) -> str:
    return f"PSNR {numeric(row.get('psnr')):.2f} dB, SSIM {numeric(row.get('ssim')):.3f}"


def threshold_text(row: dict[str, str]) -> str:
    _, psnr_thr, ssim_thr = threshold_for(row.get("dataset", ""), numeric(row.get("sampling_ratio")))
    reached = numeric(row.get("psnr")) >= psnr_thr and numeric(row.get("ssim")) >= ssim_thr
    return f"threshold PSNR >= {psnr_thr:g}, SSIM >= {ssim_thr:g}: {'PASS' if reached else 'FAIL'}"


def exact_status(method_id: str) -> str:
    rows = read_csv(PHASE15 / "exactA_reeval" / "exactA_reeval_results.csv")
    for row in rows:
        if row.get("method_id") == method_id:
            return row.get("status", "unknown")
    return "not_required"


def main() -> None:
    ensure_dir(PHASE15)
    rows = load_registry()
    mnist = row_by_id(rows, "mnist_hadamard5_full_colab")
    fashion = row_by_id(rows, "fashion_hadamard5_full_colab")
    scr5 = row_by_id(rows, "scrambled_hadamard5_hq_noise001_colab")
    rad5 = row_by_id(rows, "rademacher5_hq_noise001_colab")
    scr10 = row_by_id(rows, "scrambled_hadamard10_full_noise001_colab")
    rad10 = row_by_id(rows, "rademacher10_full_noise001_colab")
    rad5_status = exact_status("rademacher5_hq_noise001_colab")
    rad10_status = exact_status("rademacher10_full_noise001_colab")

    text = f"""# Final Claims Locked

## Locked Main Claim

NS-MC-GAN reconstructs natural and simple-domain images from fixed non-adaptive compressive measurements under a strict no-leak protocol. The final checkpoint is selected by endpoint training, test metrics are computed only after training, and the strict no-leak registry is stored in `E:/ns_mc_gan_gi/outputs_phase15/noleak_registry.csv`.

Important lock note: scrambled Hadamard, MNIST, and Fashion-MNIST rows are currently the cleanest paper-safe rows. Rademacher rows have strict no-leak Colab metrics and exported exact measurement operators, but local exact-A re-evaluation is `{rad5_status}` for 5% and `{rad10_status}` for 10%; do not claim local reproduction of Rademacher metrics until that mismatch is resolved or explicitly explained.

## Paper-Safe Supported Results

| Setting | Result | Status |
|---|---:|---|
| STL-10, scrambled Hadamard, 5% | {metric(scr5)} | {threshold_text(scr5)} |
| STL-10, Rademacher, 5% | {metric(rad5)} | {threshold_text(rad5)}; exact-A local status: {rad5_status} |
| STL-10, scrambled Hadamard, 10% | {metric(scr10)} | {threshold_text(scr10)} |
| STL-10, Rademacher, 10% | {metric(rad10)} | {threshold_text(rad10)}; exact-A local status: {rad10_status} |
| MNIST, low-frequency Hadamard, 5% | {metric(mnist)} | {threshold_text(mnist)} |
| Fashion-MNIST, low-frequency Hadamard, 5% | {metric(fashion)} | {threshold_text(fashion)} |

## Claims Allowed In The Manuscript

- Strict no-leak STL-10 5% high-quality reconstruction is fully supported for scrambled Hadamard. Rademacher 5% is supported by Colab no-leak metrics and exported exact A, but local exact-A reproduction is pending resolution.
- Strict no-leak STL-10 10% high-quality reconstruction is fully supported for scrambled Hadamard. Rademacher 10% is supported by Colab no-leak metrics and exported exact A, but local exact-A reproduction is pending resolution.
- Simple-domain sanity checks are supported on MNIST and Fashion-MNIST at 5% low-frequency Hadamard sampling.
- Rademacher results are usable only with the exported exact measurement operator files stored in the Phase 15 import tree, and should be marked conditional until the local exact-A mismatch is resolved.
- Backprojection is a weak but valid physics baseline, and NS-MC-GAN substantially improves PSNR and SSIM over it in every strict no-leak run.

## Claims Not Allowed

- Do not use old test-selected best checkpoints as main evidence.
- Do not use old Rademacher checkpoints that lack the exact exported measurement operator.
- Do not claim full measurement-family superiority from one seed.
- Do not claim clinical, production, or universal compressive-imaging validity.
- Do not treat local rescue runs with historical test monitoring as strict primary results.

## Suggested Title

No-Leak Neural Signal Manifold Constrained Reconstruction for Low-Rate Single-Pixel Imaging

## Suggested Abstract Sentence

Across strict no-leak STL-10 scrambled-Hadamard experiments at 5% and 10% sampling, the method improves over ridge backprojection by 7.96 to 10.20 dB PSNR, while MNIST and Fashion-MNIST sanity checks confirm strong recovery in simpler domains. Rademacher imports show larger gains but remain conditional until exact-A local reproduction is resolved.
"""
    out = PHASE15 / "FINAL_CLAIMS_LOCKED.md"
    out.write_text(text, encoding="utf-8")
    payload = {
        "claims_file": str(out),
        "main_rows": [row.get("method_id") for row in rows],
        "status": "locked",
    }
    write_json(PHASE15 / "FINAL_CLAIMS_LOCKED.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
