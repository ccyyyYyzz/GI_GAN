from __future__ import annotations

from .phase12_common import PHASE12, load_registry, round_float
from .utils import ensure_dir


def by_id(rows: list[dict], method_id: str) -> dict:
    return next((row for row in rows if row.get("method_id") == method_id), {})


def metric_line(row: dict) -> str:
    return f"PSNR {round_float(row.get('psnr'))}, SSIM {round_float(row.get('ssim'))}"


def main() -> None:
    ensure_dir(PHASE12)
    rows = load_registry()
    r10 = by_id(rows, "stl10_rademacher10_colab_full")
    s10 = by_id(rows, "stl10_scrambled10_colab_full")
    h10 = by_id(rows, "stl10_hadamard10_local_full")
    h5 = by_id(rows, "stl10_hadamard5_local_medium")
    mnist = by_id(rows, "mnist_hadamard5_colab_full")
    fashion = by_id(rows, "fashion_hadamard5_colab_full")
    lines = [
        "# Final Claims Draft",
        "",
        "## Title Candidates",
        "- Physics-consistent high-quality ghost imaging reconstruction with orthogonal and random single-pixel measurements",
        "- High-quality low-sampling ghost imaging via measurement-consistent null-space neural reconstruction",
        "- Low-sampling ghost imaging reconstruction with orthogonal illumination and physics-consistent refinement",
        "",
        "## Main Result Statement",
        f"STL-10 at 10% sampling reaches the internal high-quality threshold with Rademacher measurements ({metric_line(r10)}), scrambled Hadamard measurements ({metric_line(s10)}), and low-frequency Hadamard measurements ({metric_line(h10)}).",
        f"Simple-domain 5% results are strong: MNIST {metric_line(mnist)} and Fashion-MNIST {metric_line(fashion)}.",
        f"STL-10 5% remains below the stated high-quality threshold: {metric_line(h5)}.",
        "",
        "## Supported Claims",
        "- STL-10 10% high-quality reconstruction is supported under the internal PSNR/SSIM threshold.",
        "- MNIST and Fashion-MNIST 5% high-quality reconstruction is supported.",
        "- Rademacher and scrambled Hadamard measurements with the HQ reconstructor produce the strongest STL-10 10% final PSNR/SSIM.",
        "- Low-frequency Hadamard provides a strong and interpretable backprojection.",
        "- DC row retention is crucial for low-frequency Hadamard backprojection quality.",
        "- Local/Colab Hadamard 5% medium agreement supports reproducibility.",
        "",
        "## Partially Supported Claims",
        "- Continuous learned physical illumination has earlier evidence but is not the best final high-quality story.",
        "- Network refinement is a major contributor for weak random/scrambled backprojections; low-frequency Hadamard quality is partly measurement-driven.",
        "",
        "## Unsupported Claims",
        "- STL-10 5% high-quality reconstruction.",
        "- Binary learned illumination improves reconstruction.",
        "- Learned illumination is the main driver of all high-quality results.",
        "- Strict SOTA claims against unrelated papers with mismatched protocols.",
        "",
        "## Suggested Paper Structure",
        "1. Introduction",
        "2. Ghost imaging forward model",
        "3. Measurement-consistent null-space reconstruction",
        "4. Orthogonal/Hadamard and random measurements",
        "5. Training losses and HQ reconstructor",
        "6. Experiments: STL-10 10%, STL-10 5%, MNIST/Fashion 5%, measurement comparisons, DC row ablation, baselines, reproducibility",
        "7. Limitations",
        "8. Conclusion",
        "",
    ]
    path = PHASE12 / "FINAL_CLAIMS_DRAFT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
