from __future__ import annotations

from pathlib import Path

from .phase11_common import (
    CONFIG10,
    CONFIG11,
    ROOT11,
    as_float,
    copy_config_with_updates,
    ensure_dir,
    last_epoch,
    method_output_dir,
    phase10_row,
    read_convergence,
    read_phase10_results,
    write_json,
    write_md_table,
)


def row_close(row: dict, *, psnr: float, ssim: float) -> bool:
    return (as_float(row.get("model_psnr")) or -1.0) >= psnr or (as_float(row.get("model_ssim")) or -1.0) >= ssim


def row_passes(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"true", "1", "yes"}


def rising(output_dir: Path) -> bool:
    conv = read_convergence(output_dir)
    return bool(conv.get("continue_training_recommended", False))


def checkpoint_exists(output_dir: Path) -> bool:
    return (output_dir / "last.pt").exists()


def add_plan(plans: list[dict], method: str, should_run: bool, reason: str, config_path: str = "", command_note: str = "") -> None:
    plans.append(
        {
            "method": method,
            "should_run": bool(should_run),
            "reason": reason,
            "config_path": config_path,
            "command_note": command_note,
        }
    )


def make_continue_config(method: str, dest_name: str, extra_epochs: int) -> tuple[str, str]:
    src = CONFIG10 / f"{method}.yaml"
    out_dir = method_output_dir(method, phase="phase10")
    prev = last_epoch(out_dir, fallback=0)
    dest = CONFIG11 / f"{dest_name}.yaml"
    copy_config_with_updates(
        src,
        dest,
        {
            "output_dir": str(out_dir),
            "resume_checkpoint": str(out_dir / "last.pt"),
            "resume_mode": "full",
            "epochs": max(prev + extra_epochs, extra_epochs),
            "phase11_run_scale": "adaptive_continue",
        },
    )
    return str(dest), f"resume from {out_dir / 'last.pt'} to epoch {max(prev + extra_epochs, extra_epochs)}"


def make_confirm_config(source_method: str, dest_name: str, output_dir: str, min_epochs: int) -> tuple[str, str]:
    src = CONFIG10 / f"{source_method}.yaml"
    dest = CONFIG11 / f"{dest_name}.yaml"
    copy_config_with_updates(
        src,
        dest,
        {
            "output_dir": output_dir,
            "epochs": max(min_epochs, 60),
            "phase11_run_scale": "confirm_full",
        },
    )
    return str(dest), f"confirm run at {output_dir}"


def main() -> None:
    ensure_dir(ROOT11)
    ensure_dir(CONFIG11)
    rows = read_phase10_results()
    plans: list[dict] = []

    had10 = phase10_row("hadamard10_full_noise001")
    had10_dir = method_output_dir("hadamard10_full_noise001")
    if not had10 or had10.get("status") != "completed":
        add_plan(plans, "hadamard10_full_noise001", False, "missing_or_incomplete_phase10_result")
    elif row_passes(had10, "reaches_stl10_10pct_hq"):
        if rising(had10_dir) and checkpoint_exists(had10_dir):
            cfg, note = make_continue_config("hadamard10_full_noise001", "hadamard10_continue_noise001", 40)
            add_plan(plans, "hadamard10_full_noise001", True, "threshold_passed_but_convergence_still_rising", cfg, note)
        else:
            add_plan(plans, "hadamard10_full_noise001", False, "threshold_passed_no_forced_continue")
    elif row_close(had10, psnr=21.5, ssim=0.62) and rising(had10_dir) and checkpoint_exists(had10_dir):
        cfg, note = make_continue_config("hadamard10_full_noise001", "hadamard10_continue_noise001", 40)
        add_plan(plans, "hadamard10_full_noise001", True, "near_10pct_threshold_and_still_rising", cfg, note)
    else:
        add_plan(plans, "hadamard10_full_noise001", False, "not_close_or_not_rising_or_no_last_checkpoint")

    had5 = phase10_row("hadamard5_medium_noise001")
    had5_dir = method_output_dir("hadamard5_medium_noise001")
    if not had5 or had5.get("status") != "completed":
        add_plan(plans, "hadamard5_medium_noise001", False, "missing_or_incomplete_phase10_result")
    elif row_passes(had5, "reaches_stl10_5pct_hq"):
        cfg, note = make_confirm_config(
            "hadamard5_full_noise001",
            "hadamard5_confirm_full_noise001",
            "E:/ns_mc_gan_gi/outputs_phase11/hadamard5_confirm_full_noise001",
            60,
        )
        add_plan(plans, "hadamard5_confirm_full_noise001", True, "5pct_threshold_passed_confirm_full", cfg, note)
    elif (row_close(had5, psnr=19.0, ssim=0.55) or rising(had5_dir)) and checkpoint_exists(had5_dir):
        cfg, note = make_continue_config("hadamard5_medium_noise001", "hadamard5_continue_noise001", 60)
        add_plan(plans, "hadamard5_medium_noise001", True, "near_5pct_threshold_or_still_rising", cfg, note)
    else:
        add_plan(plans, "hadamard5_medium_noise001", False, "not_close_or_no_last_checkpoint")

    for method in ["rademacher10_full_noise001", "scrambled_hadamard10_full_noise001", "mnist_hadamard5_full", "fashion_hadamard5_full"]:
        row = phase10_row(method)
        out_dir = method_output_dir(method)
        if not row or row.get("status") != "completed":
            add_plan(plans, method, False, "missing_or_incomplete_phase10_result")
        elif method.startswith("scrambled") and row_close(row, psnr=21.5, ssim=0.62) and checkpoint_exists(out_dir):
            cfg, note = make_continue_config(method, f"{method}_continue", 40)
            add_plan(plans, method, True, "scrambled_close_to_lowfreq_optional_continue", cfg, note)
        elif method in {"mnist_hadamard5_full", "fashion_hadamard5_full"} and not row_passes(row, "reaches_simple_domain_hq") and row_close(row, psnr=23.5, ssim=0.75) and checkpoint_exists(out_dir):
            cfg, note = make_continue_config(method, f"{method}_continue", 30)
            add_plan(plans, method, True, "simple_domain_close_to_threshold", cfg, note)
        else:
            add_plan(plans, method, False, "control_or_not_close_enough")

    payload = {
        "source": str(Path("E:/ns_mc_gan_gi/outputs_phase10/phase10_results.csv")),
        "plan": plans,
        "should_run_count": sum(1 for item in plans if item["should_run"]),
    }
    write_json(payload, ROOT11 / "adaptive_continue_plan.json")
    write_md_table(plans, ROOT11 / "adaptive_continue_plan.md", ["method", "should_run", "reason", "config_path", "command_note"])
    print(f"Adaptive continuation plan written to: {ROOT11 / 'adaptive_continue_plan.json'}")


if __name__ == "__main__":
    main()
