import torch

import gan_gauge_aligned_nsgan as ga


def test_haar_bands_shape_and_finiteness():
    x = torch.randn(2, 1, 16, 16)
    bands = ga.haar_bands(x)
    assert bands.shape == (2, 4, 8, 8)
    assert torch.isfinite(bands).all()


def test_feature_patch_discriminator_returns_score_and_features():
    disc = ga.FeaturePatchDiscriminator(in_channels=2, base=8)
    x = torch.randn(2, 2, 32, 32)
    score, features = disc(x, return_features=True)
    assert score.shape == (2,)
    assert len(features) == 4
    assert all(torch.isfinite(feat).all() for feat in features)
