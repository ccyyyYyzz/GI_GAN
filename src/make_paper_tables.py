from __future__ import annotations

from pathlib import Path

from .phase12_common import PHASE12, as_float, load_registry, round_float, write_csv, write_md_table, write_tex_table
from .utils import ensure_dir


OUT = PHASE12 / "paper_tables"


def row_by_id(rows: list[dict], method_id: str) -> dict:
    return next((row for row in rows if row.get("method_id") == method_id), {})


def paper_row(row: dict, method: str, measurement: str, sampling: str, source: str) -> dict:
    return {
        "Method": method,
        "Measurement": measurement,
        "Sampling": sampling,
        "Source": source,
        "Backproj PSNR": round_float(row.get("backproj_psnr")),
        "Backproj SSIM": round_float(row.get("backproj_ssim")),
        "Model PSNR": round_float(row.get("psnr")),
        "Model SSIM": round_float(row.get("ssim")),
        "MSE": round_float(row.get("mse"), 6),
        "Delta PSNR": round_float(row.get("delta_psnr")),
        "Delta SSIM": round_float(row.get("delta_ssim")),
        "HQ threshold reached": row.get("threshold_reached", ""),
    }


def save_table(name: str, rows: list[dict], fields: list[str]) -> None:
    write_csv(OUT / f"{name}.csv", rows, fields)
    write_md_table(OUT / f"{name}.md", rows, fields)
    write_tex_table(OUT / f"{name}.tex", rows, fields)


def main() -> None:
    ensure_dir(OUT)
    rows = load_registry()
    fields_main = [
        "Method",
        "Measurement",
        "Sampling",
        "Source",
        "Backproj PSNR",
        "Backproj SSIM",
        "Model PSNR",
        "Model SSIM",
        "MSE",
        "Delta PSNR",
        "Delta SSIM",
        "HQ threshold reached",
    ]
    stl10_10 = [
        paper_row(row_by_id(rows, "stl10_hadamard10_local_full"), "Lowfreq Hadamard", "Hadamard", "10%", "local"),
        paper_row(row_by_id(rows, "stl10_rademacher10_colab_full"), "Rademacher", "Random signed", "10%", "Colab L4"),
        paper_row(row_by_id(rows, "stl10_scrambled10_colab_full"), "Scrambled Hadamard", "Scrambled Hadamard", "10%", "Colab L4"),
    ]
    save_table("table_main_stl10_10pct", stl10_10, fields_main)
    stl10_5 = [
        paper_row(row_by_id(rows, "stl10_hadamard5_local_medium"), "Lowfreq Hadamard medium", "Hadamard", "5%", "local"),
        paper_row(row_by_id(rows, "stl10_hadamard5_colab_medium"), "Lowfreq Hadamard medium", "Hadamard", "5%", "Colab L4"),
    ]
    save_table("table_stl10_5pct", stl10_5, fields_main)
    simple = [
        paper_row(row_by_id(rows, "mnist_hadamard5_colab_full"), "MNIST Hadamard", "Hadamard", "5%", "Colab L4"),
        paper_row(row_by_id(rows, "fashion_hadamard5_colab_full"), "Fashion Hadamard", "Hadamard", "5%", "Colab L4"),
    ]
    fashion_local = row_by_id(rows, "fashion_hadamard5_local")
    if fashion_local and fashion_local.get("status") == "completed":
        simple.append(paper_row(fashion_local, "Fashion Hadamard local", "Hadamard", "5%", "local"))
    save_table("table_simple_domains_5pct", simple, fields_main)
    repro = []
    pairs = [
        ("STL-10 Hadamard 5% medium", row_by_id(rows, "stl10_hadamard5_local_medium"), row_by_id(rows, "stl10_hadamard5_colab_medium")),
        ("Fashion-MNIST Hadamard 5%", row_by_id(rows, "fashion_hadamard5_local"), row_by_id(rows, "fashion_hadamard5_colab_full")),
    ]
    for name, local, colab in pairs:
        if not local or not colab:
            continue
        repro.append(
            {
                "Experiment": name,
                "Local PSNR": round_float(local.get("psnr")),
                "Colab PSNR": round_float(colab.get("psnr")),
                "Absolute PSNR diff": round_float(abs((as_float(local.get("psnr")) or 0) - (as_float(colab.get("psnr")) or 0))),
                "Local SSIM": round_float(local.get("ssim")),
                "Colab SSIM": round_float(colab.get("ssim")),
                "Absolute SSIM diff": round_float(abs((as_float(local.get("ssim")) or 0) - (as_float(colab.get("ssim")) or 0))),
            }
        )
    save_table("table_reproducibility", repro, list(repro[0].keys()) if repro else ["Experiment"])
    claims = [
        {"Claim": "STL-10 10% HQ", "Status": "supported", "Evidence": "All preferred STL-10 10% rows meet PSNR/SSIM threshold.", "Caveat": "Internal threshold, not SOTA protocol."},
        {"Claim": "STL-10 5% HQ", "Status": "unsupported", "Evidence": "Hadamard 5% medium is below 20 PSNR / 0.60 SSIM.", "Caveat": "Near-threshold but not high-quality by stated rule."},
        {"Claim": "MNIST 5% HQ", "Status": "supported", "Evidence": "MNIST 5% exceeds simple-domain threshold.", "Caveat": "Simple dataset."},
        {"Claim": "Fashion 5% HQ", "Status": "supported", "Evidence": "Fashion 5% Colab exceeds simple-domain threshold.", "Caveat": "Use separated Colab path."},
        {"Claim": "Binary learned illumination", "Status": "unsupported", "Evidence": "No final evidence shows improvement.", "Caveat": "Do not claim."},
        {"Claim": "Continuous learned illumination evidence", "Status": "partially supported", "Evidence": "Earlier evidence exists but not primary final result.", "Caveat": "Frame as future work/limited."},
        {"Claim": "Network main contribution", "Status": "partially supported", "Evidence": "Large improvements for weak random/scrambled backprojections; Hadamard backprojection is already strong.", "Caveat": "Attribute carefully."},
        {"Claim": "DC row importance", "Status": "supported", "Evidence": "Phase 9 include-vs-skip DC backprojection gap.", "Caveat": "Backprojection control, no new training."},
    ]
    save_table("table_claims", claims, ["Claim", "Status", "Evidence", "Caveat"])
    print(OUT)


if __name__ == "__main__":
    main()
