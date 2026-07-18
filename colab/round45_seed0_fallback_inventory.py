from pathlib import Path


paths = [
    Path("/content/GI_GAN/diagnose_fiber_residual_frequency_fusion.py"),
    Path("/content/data_primary/seed0_val.pt"),
    Path("/content/data_control/seed1_val.pt"),
    Path("/content/data_primary/config_used.yaml"),
]
for path in paths:
    print("PRESENT" if path.exists() else "MISSING", path, path.stat().st_size if path.exists() else -1)
