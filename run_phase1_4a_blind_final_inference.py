from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import phase1_4a_freeze_and_blind as p14a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1.4A blind final inference only.")
    parser.add_argument("--output-dir", default=str(p14a.OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    p14a.initialize_output()
    with (p14a.REPORTS / "command_log.txt").open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(sys.argv) + "\n")
    device = p14a.p12.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = p14a.run_stage_a_blind_inference(Path(args.output_dir), device_name=args.device, shard_size=args.shard_size)
    audit = p14a.blind_integrity_audit()
    status = "PHASE1_4A_COMPLETE" if result.get("status") == "BLIND_INFERENCE_COMPLETE" and audit.get("status") == "PASS" else "BLOCKED_PHASE1_4A"
    runtime = {
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "device": str(device),
    }
    p14a.save_json(p14a.REPORTS / "runtime_and_memory_blind_runner.json", runtime)
    impl = {
        "phase": "Phase1.4A",
        "status": status,
        "blockers": [] if status == "PHASE1_4A_COMPLETE" else ["BLIND_INFERENCE_INTEGRITY_FAILED"],
        "final_blind_inference_completed": status == "PHASE1_4A_COMPLETE",
        "final_candidates_generated": status == "PHASE1_4A_COMPLETE",
        "final_truth_metrics_computed": False,
        "final_scoring_completed": False,
        "runtime_seconds": runtime["runtime_seconds"],
        "peak_gpu_memory_bytes": runtime["peak_gpu_memory_bytes"],
    }
    p14a.save_json(p14a.REPORTS / "implementation_status_phase1_4a.json", impl)
    print(json.dumps(p14a.json_safe({"result": result, "audit": audit, "implementation_status": impl}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
