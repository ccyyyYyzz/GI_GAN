from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
OUT = ROOT / "outputs" / "compatibility" / "phase1_4a_final_freeze_and_blind"
SCORING = OUT / "final_scoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4B one-time final scorer. Not run by Phase 1.4A.")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--incident-override", default="")
    parser.add_argument("--output-dir", default=str(OUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    if args.confirm != "FINAL_LOCKED_SCORING_ONCE":
        print("REFUSING: pass --confirm FINAL_LOCKED_SCORING_ONCE to run the one-time final scorer.")
        return 2
    frozen = out / "freeze_bundle" / "FINAL_EVAL_FROZEN.json"
    complete = out / "blind_inference" / "BLIND_INFERENCE_COMPLETE.json"
    if not frozen.exists() or not complete.exists():
        print("REFUSING: FINAL_EVAL_FROZEN.json and BLIND_INFERENCE_COMPLETE.json are required.")
        return 2
    scoring = out / "final_scoring"
    done = scoring / "FINAL_SCORING_COMPLETE.json"
    started = scoring / "FINAL_SCORING_STARTED.json"
    if done.exists():
        print("REFUSING: final scoring already completed.")
        return 2
    if started.exists() and not args.incident_override:
        print("REFUSING: previous scoring start exists; incident override required.")
        return 2
    scoring.mkdir(parents=True, exist_ok=True)
    started.write_text(json.dumps({"status": "FINAL_SCORING_STARTED"}), encoding="utf-8")
    print("Phase 1.4B scorer skeleton is armed, but metric implementation is intentionally not executed in Phase 1.4A.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
