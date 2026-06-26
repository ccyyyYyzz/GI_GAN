from __future__ import annotations

import gan_high_quality_gi_matched as matched


def test_image_level_bootstrap_aggregates_seed_repeats() -> None:
    rows = []
    for seed in [0, 1, 2]:
        for sample in [0, 1]:
            rows.append({"method": "matched_no_gan", "train_seed": seed, "sample_ordinal": sample, "lpips": 1.0})
            rows.append({"method": "matched_gan", "train_seed": seed, "sample_ordinal": sample, "lpips": 0.9 if sample == 0 else 0.8})
    out = matched.paired_image_bootstrap(
        rows,
        "matched_gan",
        "matched_no_gan",
        "lpips",
        higher_is_better=False,
        reps=20,
        seed=123,
    )
    assert out["status"] == "PASS"
    assert out["n_images"] == 2
    assert out["n_seed_image_pairs"] == 6
    assert out["mean_delta"] < 0
    assert len(out["seed_summary"]) == 3


def test_stable_hash_changes_with_tensor_value() -> None:
    import torch

    a = {"w": torch.tensor([1.0, 2.0])}
    b = {"w": torch.tensor([1.0, 3.0])}
    assert matched.stable_hash(a) != matched.stable_hash(b)
