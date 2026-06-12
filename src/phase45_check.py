from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase45_math_repro"
PROJECT = OUT / "latex_project_v45"
REPORT = OUT / "PHASE45_CHECK_REPORT.md"
MAIN_PDF = OUT / "main_v45.pdf"
SUPP_PDF = OUT / "supplement_v45.pdf"
MAIN_TXT = OUT / "main_v45.txt"
SUPP_TXT = OUT / "supplement_v45.txt"
AUDIT_MD = OUT / "training_code_audit.md"
AUDIT_JSON = OUT / "training_code_audit_records.json"
TABLE_TEX = OUT / "tableS9_training_config_phase45.tex"

RESULT_STRINGS = [
    "22.316",
    "0.635",
    "22.271",
    "0.632",
    "24.781",
    "0.747",
    "24.730",
    "0.746",
    "27.692",
    "0.956",
    "25.019",
    "0.837",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(pdf: Path, txt: Path) -> str:
    if pdf.exists() and not txt.exists():
        try:
            subprocess.run(["pdftotext", str(pdf), str(txt)], check=False)
        except FileNotFoundError:
            pass
    return read(txt)


def all_tex() -> str:
    return "\n".join(read(path) for path in PROJECT.rglob("*.tex"))


def absent(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is None


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    for path in [MAIN_PDF, SUPP_PDF, AUDIT_MD, AUDIT_JSON, TABLE_TEX]:
        if not path.exists():
            failures.append(f"Missing required output: {path}")

    source = all_tex()
    audit = read(AUDIT_MD)
    table = read(TABLE_TEX)
    main_text = pdf_text(MAIN_PDF, MAIN_TXT)
    supp_text = pdf_text(SUPP_PDF, SUPP_TXT)
    public_text = main_text + "\n" + supp_text
    records = []
    if AUDIT_JSON.exists():
        try:
            records = json.loads(read(AUDIT_JSON))
        except json.JSONDecodeError as exc:
            failures.append(f"Audit JSON is invalid: {exc}")

    checks = [
        (
            "Are all actual training loss terms extracted from code/config?",
            all(
                term in audit
                for term in [
                    "lambda_l1",
                    "lambda_dc_loss",
                    "lambda_tv",
                    "lambda_charbonnier",
                    "lambda_edge",
                    "lambda_ms_l1",
                    "lambda_ssim",
                    "lambda_ms_ssim",
                    "lambda_gradient",
                    "lambda_frequency",
                    "lambda_stage1_aux",
                ]
            )
            and "adversarial" in audit.lower()
            and len(records) == 6,
        ),
        (
            "Are loss weights listed?",
            "Loss weights:" in audit and "lambda_l1=30" in audit and "lambda_adv=0" in audit,
        ),
        (
            "Is stage1 supervision yes/no stated?",
            "Stage-1 supervision:" in audit
            and "lambda_stage1_aux=0" in audit
            and ("Stage-1 supervision is no" in source or "not weighted" in source.lower()),
        ),
        (
            "Is measurement loss exact formula stated?",
            "F.mse_loss" in audit
            and "not normalized" in audit
            and r"\mathcal{L}_{\rm meas}(u,y)" in source
            and "not divided by" in source,
        ),
        (
            "Are optimizer, LR, scheduler, epochs, batch size stated?",
            all(term in audit for term in ["Optimizer:", "lr_g=", "lr_d=", "LR schedule:", "Epochs/batch:"])
            and "No learning-rate scheduler" in source,
        ),
        (
            r"Is \(\lambda_{\rm op}\) separated from \(\lambda_{\rm tv}\)?",
            r"\lambda_{\rm op}" in source
            and r"\lambda_{\rm tv}" in source
            and r"B_{\lambda_{\rm op}}" in source,
        ),
        (
            "Are soft-gate and audit residual formulas included?",
            (r"AP_N^{\lambda_{\rm op}}" in source or r"AP_N^{\lambda}" in source)
            and r"\lambda_{\rm op}K_{\lambda_{\rm op}}^{-1}Av" in source
            and r"A\Pi_y(v)-y" in source
            and "regularized soft audit" in source,
        ),
        (
            "Are Algorithm 1 and Algorithm 2 included?",
            "Algorithm 1: Inference with measurement-audited neural completion" in source
            and "Algorithm 2: Training one mini-batch" in source,
        ),
        (
            "Are metrics PSNR/SSIM/RelMeasErr defined?",
            r"\operatorname{MSE}" in source
            and r"\operatorname{PSNR}" in source
            and ("structural_similarity" in source or r"structural\_similarity" in source)
            and r"\operatorname{RelMeasErr}" in source,
        ),
        (
            "Is clipping convention stated?",
            "clamp_eval_only" in audit
            and "PSNR/SSIM use clipped" in audit
            and "unclipped" in source,
        ),
        (
            "Is exact-A handling stated as solver-cache rebuild?",
            "safe cache" in source.lower()
            or "rebuilds the Cholesky cache" in source
            and "Exact A exported" in table,
        ),
        (
            r"Is CS-TV formula fixed to \(\operatorname{TV}(x)\)?",
            r"\lambda_{\rm tv}\operatorname{TV}(x)" in source
            and r"\lambda\operatorname{TV}(x)" not in source
            and re.search(r"(?<!\\operatorname\{)TV\(x\)", source) is None,
        ),
        (
            "Are all reported numbers unchanged?",
            all(value in public_text for value in RESULT_STRINGS),
        ),
        (
            "Are missing config fields listed rather than guessed?",
            "## Missing or non-claimed fields" in audit
            and "Exact GPU model" in audit
            and "not claimed" in source,
        ),
        (
            "No new training and no new claims?",
            absent(public_text, r"new training experiment|we trained additional|state-of-the-art|SOTA|hardware experiment")
            and "do not introduce additional training" in public_text,
        ),
        (
            "No GAN main mechanism if adversarial disabled?",
            "Adversarial loss: disabled" in audit
            and ("use_adversarial=false" in source or r"use\_adversarial=false" in source)
            and "Adversarial loss is disabled" in source
            and absent(public_text, r"GAN main mechanism|GAN-based method"),
        ),
    ]

    for label, ok in checks:
        if not ok:
            failures.append(label)

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")
    if "Hadamard zero-fill anchor option" not in source:
        warnings.append("Hadamard zero-fill implementation note not found verbatim; verify Algorithm 1 text.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Phase 45 Check Report",
        "",
        f"Status: {status}",
        "",
        "## Required Questions",
        "",
    ]
    for i, (label, ok) in enumerate(checks, 1):
        lines.append(f"{i}. {label}: {'yes' if ok else 'no'}")
    lines.extend(
        [
            "",
            "## Missing Fields That Remain Non-Claimed",
            "",
            "- Exact GPU model is not available for every imported Colab run in local metadata; configs state `device: cuda`.",
            "- Not every imported final folder preserves the full original Colab training log or checkpoint payload locally.",
            "- Dataset subset indices are not listed in the paper; they are reproducible from the code, split name, limits, and seeds.",
            "",
            "## Failures",
        ]
    )
    lines.extend([f"- {item}" for item in failures] or ["- None."])
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines.extend(
        [
            "",
            "## Output Paths",
            f"- Main PDF: {MAIN_PDF}",
            f"- Supplement PDF: {SUPP_PDF}",
            f"- Training audit: {AUDIT_MD}",
            f"- Audit records JSON: {AUDIT_JSON}",
            f"- Repro config table: {TABLE_TEX}",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
