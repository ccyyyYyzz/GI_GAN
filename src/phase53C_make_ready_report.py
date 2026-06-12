from __future__ import annotations

import argparse
from pathlib import Path


NOTEBOOKS = [
    "session_20_exact_null_mi_pretest.ipynb",
    "session_21_soft_leakage_and_shortcut_audit.ipynb",
    "session_22_feasible_hallucination_figure.ipynb",
    "session_23_exact_null_critic_evaluator.ipynb",
    "session_24_optional_gan_and_posterior_sampling.ipynb",
]


def status(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def bullet(label: str, path: Path) -> str:
    return f"- {label}: `{path}` [{status(path)}]"


def theory_notes() -> str:
    return r"""# Phase53C Theory Notes: Anchor-Conditioned Exact-Null Critic

Core rule: **Certify the measured, criticize the unmeasured.**

The analytic measurement certificate is

\[
\Pi_y^\lambda(v)=v-B_\lambda(Av-y).
\]

The learned critic is only an anchor-conditioned null-space plausibility test:

\[
D_\psi(P_0u,\ x_{\rm data}).
\]

It is not a certificate.

## Theorem 1: Known-channel Pair Critic Degeneracy

For binary classification between matched \((u,y)\sim p(u,y)\) and independent \((u,y)\sim p(u)p(y)\), if

\[
p(y|u)=\mathcal N(Au,\sigma^2I),
\]

then the optimal BCE/logistic critic logit satisfies

\[
D^*(u,y)=\log\frac{p(u,y)}{p(u)p(y)}
=-\frac{1}{2\sigma^2}\|Au-y\|_2^2+c(y).
\]

Therefore a full measurement-conditioned discriminator that sees residual information degenerates to a measurement residual classifier.

## Proposition 2: Null-space No-certificate

If \(Au=Au'=y\), then \(p(y|u)=p(y|u')\). Any physical statistic depending only on \(A,y,Au-y\) cannot distinguish feasible alternatives inside the same affine measurement set.

## Proposition 3: Exact-null Algebraic Blindness and Soft Leakage

The exact null projector

\[
P_0=I-A^{\mathsf T}(AA^{\mathsf T})^{-1}A
\]

satisfies

\[
AP_0=0.
\]

Therefore \(P_0u\) contains no row-space component of \(u\), and \(Au-y\) is not algebraically computable from \(P_0u\) and \(x_{\rm data}\) alone.

Do not claim statistical independence. Use **algebraically blind** or **non-identifiable**.

For the soft projector

\[
P_N^\lambda=I-A^{\mathsf T}(AA^{\mathsf T}+\lambda I)^{-1}A,
\]

the leakage channel is

\[
AP_N^\lambda=\lambda(AA^{\mathsf T}+\lambda I)^{-1}A.
\]

In singular directions, the leakage factor is

\[
\lambda/(\lambda+\sigma_i^2).
\]

Therefore soft \(P_N^\lambda\) must not be used as the critic input if we claim exact blindness.

## Proposition 4: Anchor Information Law

For balanced classification between

\[
P=p(P_0x,x_{\rm data})
\]

and

\[
P_{\rm prod}=p(P_0x)p(x_{\rm data}),
\]

the optimal accuracy is

\[
\mathrm{acc}^*=\frac12(1+\mathrm{TV}(P,P_{\rm prod})).
\]

By Pinsker,

\[
\mathrm{acc}^*\le \frac12+\sqrt{I(P_0x;x_{\rm data})/8}.
\]

Since \(x_{\rm data}=f(y)\),

\[
I(P_0x;x_{\rm data})\le I(P_0x;y).
\]

Interpretation: weak anchor information implies any anchor-conditioned null-space critic must fail. This predicts Scr > Rad and 10% > 5% as a hypothesis, not a guarantee.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Phase53C theory notes and Colab readiness report.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_ready")
    parser.add_argument("--upload_dir", default="E:/ns_mc_gan_gi/colab_upload")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    out = Path(args.output_dir)
    upload = Path(args.upload_dir)
    out.mkdir(parents=True, exist_ok=True)
    theory = out / "PHASE53C_THEORY_NOTES.md"
    theory.write_text(theory_notes(), encoding="utf-8")
    project_zip = upload / "ns_mc_gan_gi_project_phase53C.zip"
    bundle_53c = upload / "noleak_bundle_phase53C.zip"
    bundle_4849 = upload / "noleak_bundle_phase48_49.zip"
    lines = [
        "# Phase53C Colab Ready Report",
        "",
        "No local training is started by this readiness check.",
        "",
        "## Theory",
        bullet("theory notes", theory),
        "",
        "## Colab notebooks",
    ]
    for nb in NOTEBOOKS:
        lines.append(bullet(nb, repo / "colab" / "phase53C" / nb))
    lines.extend(
        [
            "",
            "## Colab upload files",
            bullet("Phase53C project zip", project_zip),
            f"- Phase53C no-leak bundle alias: `{bundle_53c}` [{'OK' if bundle_53c.exists() else 'OPTIONAL; fallback is allowed' if bundle_4849.exists() else 'MISSING'}]",
            bullet("Phase48/49 no-leak bundle fallback", bundle_4849),
            "",
            "## Local scripts",
            bullet("prepare upload bundle", repo / "scripts" / "phase53C" / "phase53C_prepare_upload_bundle.ps1"),
            bullet("merge Colab parts", repo / "scripts" / "phase53C" / "phase53C_merge_colab_parts.ps1"),
            bullet("import Colab outputs", repo / "scripts" / "phase53C" / "phase53C_import_colab_outputs.ps1"),
            "",
            "## Output roots",
            "- Colab: `/content/outputs_phase53C_exact_null_critic`",
            "- Local import: `E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import`",
            "",
            "## Strict rules",
            "- D sees only exact `P0u` and `x_data` for the proposed critic.",
            "- Soft `P_N^lambda` may remain in the main reconstructor, but not in exact-blind critic input.",
            "- Analytic Pi_y is the measurement certificate; D is not a certificate.",
            "- All Phase53C outputs are exploratory / innovation screening.",
        ]
    )
    path = out / "PHASE53C_COLAB_READY_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
