import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "round59", ROOT / "run_frozen_fohi_raw_fiber_once.py"
)
round59 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(round59)


def test_diagnostic_command_requires_raw_y_target(tmp_path: Path) -> None:
    command = round59.diagnostic_command(
        repo_root=tmp_path,
        cache=tmp_path / "cache.pt",
        frozen_rate={
            "config": "/content/config.yaml",
            "structural_checkpoint": "/content/structural.pt",
            "proposal_checkpoint": "/content/proposal.pt",
        },
        lane_index=2,
        rate="10",
        output_dir=tmp_path / "out",
    )
    assert command[command.index("--final-target") + 1] == "raw_y"
    assert command[command.index("--seed") + 1] == str(20260719 + 200 + 10)
    assert "--primary-val" in command and "--control-val" in command


def test_lane_from_archive_rejects_noncampaign_lane(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", json.dumps({"seed": 7}))
    with pytest.raises(RuntimeError, match="INVALID_LANE_INDEX:7"):
        round59.lane_from_archive(archive)


def test_cache_receipt_rejects_overlap(tmp_path: Path) -> None:
    cache_dir = tmp_path / "lane0" / "rate05" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "test_cache.pt").write_bytes(b"cache")
    (cache_dir / "test_cache_manifest.json").write_text(
        json.dumps({"test_images": 6740, "included_development_raw_hash_overlap": 1}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="ROUND56_CACHE_DEVELOPMENT_OVERLAP:05"):
        round59.cache_receipt(source_lane=tmp_path / "lane0", rate="05", expected_images=6740)
