from __future__ import annotations

from .phase14_common import PHASE14, ensure_dir, read_csv


def main() -> None:
    out = ensure_dir(PHASE14)
    rows = read_csv(out / "phase14_final_results.csv")
    p14 = [r for r in rows if r.get("method_id") in {"stl10_rademacher5_colab_full", "stl10_scrambled5_colab_full"}]
    completed = all(r.get("status") == "completed" for r in p14) and len(p14) == 2
    reached = all(r.get("threshold_reached") == "True" for r in p14) and len(p14) == 2
    lines = [
        "# Updated Claims After Phase 14",
        "",
        f"- Phase 14 5% Colab results present: {completed}",
        f"- Phase 14 5% HQ threshold fully reached: {reached}",
        "",
        "## Suggested claim wording",
        "",
    ]
    if reached:
        lines.append("STL-10 5% high-quality reconstruction is supported for both Rademacher and scrambled Hadamard measurements under the Phase 14 Colab runs.")
    elif completed:
        lines.append("STL-10 5% results are available, but at least one run does not meet the pre-set HQ threshold; describe them as stress-test results.")
    else:
        lines.append("STL-10 5% Rademacher/scrambled Hadamard remains pending; keep final claims anchored on completed 10% STL-10 and simple-domain 5% results.")
    target = out / "UPDATED_CLAIMS_FOR_WRITING.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
