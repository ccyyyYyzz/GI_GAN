from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .phase26_common import PILOT_CONFIGS, drive_root, ensure_dir, output_root, read_json, write_csv, write_json
from .utils import load_config


FIELDS = ["config_name", "config_path", "output_dir", "status", "started_at", "ended_at", "seconds", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 26 medium architecture pilots.")
    parser.add_argument("--drive_root", default=None)
    parser.add_argument("--configs", default=",".join(PILOT_CONFIGS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_configs", type=int, default=None)
    return parser.parse_args()


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def existing_records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("records") if isinstance(payload, dict) else None
    return rows if isinstance(rows, list) else []


def main() -> None:
    args = parse_args()
    root = drive_root(args.drive_root)
    out = output_root(root)
    status_path = out / "arch_pilot_run_status.json"
    records = existing_records(status_path)
    requested = csv_strings(args.configs)
    if args.max_configs is not None:
        requested = requested[: int(args.max_configs)]
    ensure_dir(out)
    for config_name in requested:
        config_path = Path("configs") / "phase26_arch_pilot" / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}. Run src.phase26_prepare_arch_pilot first.")
        config = load_config(config_path)
        output_dir = Path(config["output_dir"])
        if args.skip_existing and (output_dir / "eval_metrics.json").exists():
            records.append(
                {
                    "config_name": config_name,
                    "config_path": str(config_path),
                    "output_dir": str(output_dir),
                    "status": "skipped_existing",
                    "started_at": "",
                    "ended_at": "",
                    "seconds": "",
                    "notes": "eval_metrics.json already exists",
                }
            )
            write_json(status_path, {"records": records})
            continue
        start = time.time()
        row = {
            "config_name": config_name,
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "status": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": "",
            "seconds": "",
            "notes": "",
        }
        records.append(row)
        write_json(status_path, {"records": records})
        try:
            run_cmd([sys.executable, "-m", "src.train", "--config", str(config_path), "--device", args.device])
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "src.eval_auto",
                    "--output_dir",
                    str(output_dir),
                    "--config",
                    str(config_path),
                    "--device",
                    args.device,
                    "--limit_val_samples",
                    str(config.get("limit_val_samples", 1000)),
                    "--batch_size",
                    str(config.get("batch_size", 8)),
                ]
            )
            run_cmd([sys.executable, "-m", "src.analyze_convergence", "--output_dir", str(output_dir)])
            row["status"] = "complete"
        except Exception as exc:
            row["status"] = "failed"
            row["notes"] = repr(exc)
            write_json(status_path, {"records": records})
            write_csv(out / "arch_pilot_run_status.csv", records, FIELDS)
            raise
        finally:
            row["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            row["seconds"] = time.time() - start
            write_json(status_path, {"records": records})
            write_csv(out / "arch_pilot_run_status.csv", records, FIELDS)
    print({"status_path": str(status_path), "records": len(records)})


if __name__ == "__main__":
    main()
