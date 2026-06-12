from __future__ import annotations

import argparse
from pathlib import Path


NOTEBOOKS = [
    "session_20_blind_critic_pretest.ipynb",
    "session_21_shortcut_audit.ipynb",
    "session_22_feasible_hallucination_dataset.ipynb",
    "session_23_blind_critic_gan_pilot.ipynb",
    "session_24_posterior_sampling_pilot.ipynb",
]


def status(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def bullet(label: str, path: Path) -> str:
    return f"- {label}: `{path}` [{status(path)}]"


def theory_notes() -> str:
    return r"""# Phase53B Theory Notes: Certified-Blind Null-Space Critic

Phase53B replaces the old full-input MCAC idea with the rule:

**Certify the measured, criticize the unmeasured.**

The measured/row-space part is handled by the analytic proximal certificate

\[
\Pi_y^\lambda(v)=v-B_\lambda(Av-y).
\]

The learned critic is restricted to the unmeasured/null-space component and the measured anchor:

\[
D_\psi(P_Nu,\ x_{\rm data}).
\]

The discriminator must not receive \(u\) directly, \(Au-y\), RelMeasErr, \(\delta_\lambda=B_\lambda(Au-y)\), or \(\Pi_y(u)-u\).

## Proposition 1: Known-channel Pair Critic Degeneracy

If \(p(y|u)=\mathcal N(Au,\sigma^2I)\), then the optimal pair critic for matched versus independent pairs is

\[
D^*(u,y)= -\|Au-y\|^2/(2\sigma^2)+c(y).
\]

Therefore a full measurement-conditioned discriminator can reduce to a measurement residual classifier. This is why the old full-input MCAC discriminator is not used as the main method.

## Proposition 2: Null-space No-certificate

If \(Au=Au'=y\), then \(p(y|u)=p(y|u')\). Any statistic depending only on \(A\), \(y\), or \(Au-y\) cannot distinguish feasible hallucinations. Row-space consistency is necessary, but it cannot certify null-space semantic plausibility.

## Proposition 3: Certified-blind Critic

A critic \(D_\psi(P_Nu,x_{\rm data})\) cannot directly compute \(Au-y\) from its null-space input under ideal \(P_N\). It tests prior or semantic compatibility of a null-space completion with the measured anchor, not row-space consistency.

## Vocabulary

- Analytic certificate: \(\Pi_y\).
- Learned critic: \(D_\psi\).
- Allowed phrase: certified-blind null-space critic.
- Forbidden phrase: adversarial certificate.
- Do not call \(D_\psi\) a certificate.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Phase53B theory notes and Colab readiness report.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="E:/ns_mc_gan_gi/outputs_phase53B_blind_null_critic_ready")
    parser.add_argument("--upload_dir", default="E:/ns_mc_gan_gi/colab_upload")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out = Path(args.output_dir)
    upload = Path(args.upload_dir)
    out.mkdir(parents=True, exist_ok=True)
    theory_path = out / "PHASE53B_THEORY_NOTES.md"
    theory_path.write_text(theory_notes(), encoding="utf-8")

    project_zip = upload / "ns_mc_gan_gi_project_phase53B.zip"
    bundle_53b = upload / "noleak_bundle_phase53B.zip"
    bundle_4849 = upload / "noleak_bundle_phase48_49.zip"
    lines = [
        "# Phase53B Colab Ready Report",
        "",
        "No local training is started by this readiness check.",
        "",
        "## Theory",
        bullet("theory notes", theory_path),
        "",
        "## Colab notebooks",
    ]
    for nb in NOTEBOOKS:
        lines.append(bullet(nb, repo / "colab" / "phase53B" / nb))
    lines.extend(
        [
            "",
            "## Session runners",
            bullet("Session20 pretest", repo / "src" / "phase53B_blind_critic_pretest.py"),
            bullet("Session21 shortcut audit", repo / "src" / "phase53B_shortcut_audit.py"),
            bullet("Session22 feasible hallucination", repo / "src" / "phase53B_feasible_hallucination.py"),
            bullet("Session23 blind GAN pilot", repo / "src" / "phase53B_blind_gan_pilot.py"),
            bullet("Session24 posterior sampling", repo / "src" / "phase53B_posterior_sampling.py"),
            bullet("aggregate", repo / "src" / "phase53B_aggregate.py"),
            "",
            "## Local scripts",
            bullet("prepare upload bundle", repo / "scripts" / "phase53B" / "phase53B_prepare_upload_bundle.ps1"),
            bullet("merge Colab parts", repo / "scripts" / "phase53B" / "phase53B_merge_colab_parts.ps1"),
            bullet("import Colab outputs", repo / "scripts" / "phase53B" / "phase53B_import_colab_outputs.ps1"),
            "",
            "## Colab upload files",
            bullet("Phase53B project zip", project_zip),
            f"- Phase53B no-leak bundle alias: `{bundle_53b}` [{'OK' if bundle_53b.exists() else 'OPTIONAL; fallback is allowed' if bundle_4849.exists() else 'MISSING'}]",
            bullet("Phase48/49 no-leak bundle fallback", bundle_4849),
            "",
            "## Colab output root",
            "- `/content/outputs_phase53B_blind_null_critic`",
            "",
            "## Local import root",
            "- `E:/ns_mc_gan_gi/outputs_phase53B_blind_null_critic_import`",
            "",
            "## Strict rules",
            "- Do not run old full-input MCAC as the main method.",
            "- The blind critic must not see residuals, RelMeasErr, delta, or audit displacement.",
            "- D is a critic, not a certificate; Pi_y is the certificate.",
            "- All Phase53B outputs are exploratory / innovation screening.",
        ]
    )
    path = out / "PHASE53B_COLAB_READY_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
