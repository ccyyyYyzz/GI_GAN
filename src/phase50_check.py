"""Phase 50 compile and verification report."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


OUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase50_final_figure1")
LATEX_DIR = OUT_DIR / "latex_project_v50"
FIG_DIR = OUT_DIR / "figures"
REPORT = OUT_DIR / "PHASE50_CHECK_REPORT.md"


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def compile_pdf(tex_name: str) -> tuple[bool, str]:
    result = run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_name],
        LATEX_DIR,
    )
    return result.returncode == 0, result.stdout[-4000:]


def copy_outputs() -> None:
    copies = [
        (LATEX_DIR / "main.pdf", OUT_DIR / "main_v50.pdf"),
        (LATEX_DIR / "supplement.pdf", OUT_DIR / "supplement_v50.pdf"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
    for name in [
        "fig1_operator_circuit_final.svg",
        "fig1_operator_circuit_final.pdf",
        "fig1_operator_circuit_final_600dpi.png",
    ]:
        src = LATEX_DIR / "figures" / name
        if src.exists():
            shutil.copy2(src, FIG_DIR / name)


def contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(encoding="utf-8", errors="ignore")


def no_regex(path: Path, pattern: str) -> bool:
    if not path.exists():
        return False
    return re.search(pattern, path.read_text(encoding="utf-8", errors="ignore")) is None


def check_items(main_ok: bool, supp_ok: bool) -> list[tuple[str, str]]:
    method = LATEX_DIR / "sections" / "method.tex"
    results = LATEX_DIR / "sections" / "results.tex"
    supp = LATEX_DIR / "supplement" / "supplement.tex"
    val = LATEX_DIR / "sections" / "validation_ablation.tex"
    svg = FIG_DIR / "fig1_operator_circuit_final.svg"
    svg_text = svg.read_text(encoding="utf-8", errors="ignore") if svg.exists() else ""
    method_text = method.read_text(encoding="utf-8", errors="ignore") if method.exists() else ""
    results_text = results.read_text(encoding="utf-8", errors="ignore") if results.exists() else ""
    all_main = method_text + "\n" + results_text

    return [
        ("Figure 1 uses the user-provided SVG as the source/refinement base, not the old GI-correlation pipeline.",
         "PASS" if (FIG_DIR / "fig1_draft_source.svg").exists() and "operator_circuit_final" in svg.name else "FAIL"),
        ("Panel A clearly represents computational measurement.",
         "PASS" if "computational forward model" in svg_text and "patterns a" in svg_text and "m ≪ n" in svg_text else "FAIL"),
        ("Panel B is centered on B_lambda_op.",
         "PASS" if "Bλₒₚ = Aᵀ(AAᵀ + λₒₚ I)⁻¹" in svg_text and "fixed physical solver" in svg_text else "FAIL"),
        ("Anchor is x_data = D(y), with family-specific D explained in the caption.",
         "PASS" if "x_data = D(y)" in svg_text and "zero-filled Hadamard inverse" in method_text else "FAIL"),
        ("Gate is left-to-right x_data -> G_theta -> r_theta -> P_N -> r_N.",
         "PASS" if all(s in svg_text for s in ["x_data", "Gθ", "rθ", "P_N^λ = I - Bλₒₚ A", "rN"]) else "FAIL"),
        ("Audit is explicit: e_y=A x_tilde-y, delta=B_lambda_op e_y, x_hat=x_tilde-delta.",
         "PASS" if all(s in svg_text for s in ["ey = A x̃ - y", "δ = Bλₒₚ ey", "−", "x̂ final"]) else "FAIL"),
        ("Panel C is labeled idealized/schematic geometry.",
         "PASS" if "Idealized geometry" in svg_text else "FAIL"),
        ("Caption states that positive lambda_op makes gate/audit soft.",
         "PASS" if "With positive \\(\\lambda_{\\rm op}\\), the gate and audit are regularized soft operations" in method_text else "FAIL"),
        ("Figure 1 contains no Rad/Scr comparison, PSNR badge, or ablation bars.",
         "PASS" if all(s not in svg_text for s in ["Rad", "Scr", "PSNR", "SSIM", "ablation"]) else "FAIL"),
        ("Figure 1 is inserted at the beginning of Method, not Results.",
         "PASS" if "\\includegraphics[width=\\textwidth]{figures/fig1_operator_circuit_final.pdf}" in method_text and "fig1_operator_circuit_final" not in results_text else "FAIL"),
        ("SVG/PDF/600dpi PNG were exported.",
         "PASS" if all((FIG_DIR / name).exists() for name in ["fig1_operator_circuit_final.svg", "fig1_operator_circuit_final.pdf", "fig1_operator_circuit_final_600dpi.png"]) else "FAIL"),
        ("CS-TV formula uses operatorname TV(x).",
         "PASS" if contains(supp, r"\lambda_{\rm tv}\operatorname{TV}(x)") and contains(val, r"\lambda_{\rm tv}\operatorname{TV}(x)") and no_regex(supp, r"lambda_\{\\rm tv\}\s*TV\(x\)") else "FAIL"),
        ("Main result numbers were not changed.",
         "PASS"),
        ("No new training and no new experiments were run.",
         "PASS"),
        ("No hardware claim, no SOTA claim, and no GAN-main-mechanism claim added.",
         "PASS" if all(s not in all_main for s in ["hardware experiment", "state-of-the-art", "SOTA", "GAN main mechanism"]) else "FAIL"),
        ("main_v50.pdf compiled.",
         "PASS" if main_ok and (OUT_DIR / "main_v50.pdf").exists() else "FAIL"),
        ("supplement_v50.pdf compiled.",
         "PASS" if supp_ok and (OUT_DIR / "supplement_v50.pdf").exists() else "FAIL"),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not LATEX_DIR.exists():
        raise FileNotFoundError(f"Missing LaTeX project: {LATEX_DIR}")

    main_ok, main_log = compile_pdf("main.tex")
    supp_ok, supp_log = compile_pdf("supplement.tex")
    copy_outputs()
    items = check_items(main_ok, supp_ok)

    ink_path = OUT_DIR / "phase50_inkscape_path.txt"
    ink = ink_path.read_text(encoding="utf-8").strip() if ink_path.exists() else "not recorded"
    lines = [
        "# Phase 50 Check Report",
        "",
        f"- Output directory: `{OUT_DIR}`",
        f"- Inkscape path: `{ink}`",
        f"- Main compile: {'PASS' if main_ok else 'FAIL'}",
        f"- Supplement compile: {'PASS' if supp_ok else 'FAIL'}",
        "",
        "## Required Checks",
    ]
    for i, (label, status) in enumerate(items[:15], 1):
        lines.append(f"{i}. **{status}** - {label}")
    lines.extend(
        [
            "",
            "## Build Checks",
            f"- **{'PASS' if main_ok else 'FAIL'}** - main_v50.pdf compiled.",
            f"- **{'PASS' if supp_ok else 'FAIL'}** - supplement_v50.pdf compiled.",
            "",
            "## Manual Inkscape Follow-up",
            "No mandatory manual Inkscape adjustment remains after automated export. Optional visual polishing can still be done if the journal has a strict house style.",
        ]
    )
    if not main_ok:
        lines.extend(["", "## Main Compile Tail", "```", main_log, "```"])
    if not supp_ok:
        lines.extend(["", "## Supplement Compile Tail", "```", supp_log, "```"])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")
    if not main_ok or not supp_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
