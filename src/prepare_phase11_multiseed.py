from __future__ import annotations

from .phase11_common import (
    CONFIG10,
    CONFIG11,
    ROOT11,
    as_bool,
    copy_config_with_updates,
    ensure_dir,
    phase10_row,
    write_json,
    write_md_table,
)


def main() -> None:
    ensure_dir(CONFIG11)
    ensure_dir(ROOT11)
    had10 = phase10_row("hadamard10_full_noise001")
    rows = []
    if not had10 or had10.get("status") != "completed":
        reason = "hadamard10_full_noise001_missing_or_incomplete"
    elif not as_bool(had10.get("reaches_stl10_10pct_hq")):
        reason = "hadamard10_full_noise001_did_not_reach_threshold"
    else:
        reason = ""
    if reason:
        rows.append({"seed": "", "config_path": "", "should_run": False, "skipped_reason": reason})
    else:
        for seed in [43, 44]:
            dest = CONFIG11 / f"hadamard10_seed{seed}.yaml"
            copy_config_with_updates(
                CONFIG10 / "hadamard10_full_noise001.yaml",
                dest,
                {
                    "seed": seed,
                    "output_dir": f"E:/ns_mc_gan_gi/outputs_phase11/hadamard10_seed{seed}",
                    "epochs": 40,
                    "limit_train_samples": 30000,
                    "limit_val_samples": 1500,
                    "phase11_run_scale": "multiseed",
                },
            )
            rows.append({"seed": seed, "config_path": str(dest), "should_run": True, "skipped_reason": ""})
    write_json({"runs": rows}, ROOT11 / "multiseed_plan.json")
    write_md_table(rows, ROOT11 / "multiseed_plan.md", ["seed", "config_path", "should_run", "skipped_reason"])
    print(f"Multiseed plan written to: {ROOT11 / 'multiseed_plan.json'}")


if __name__ == "__main__":
    main()
