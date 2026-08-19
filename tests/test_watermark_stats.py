import torch

from paul_resampler.watermark import _score_g_values, parse_keys


def test_parse_keys():
    assert parse_keys("1, 2,3") == (1, 2, 3)


def test_mean_and_weighted_mean_scores():
    # batch=1, positions=2, depth=2
    g = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.float32)
    mask = torch.tensor([[1, 1]], dtype=torch.bool)
    mean, weighted = _score_g_values(g, mask)
    assert mean == 0.75

    # Default depth weights are [10, 1], normalized to sum to depth=2.
    weights = torch.tensor([10.0, 1.0]) * (2.0 / 11.0)
    expected = ((g * weights.view(1, 1, -1)).sum() / 4.0).item()
    assert abs(weighted - expected) < 1e-7


def test_mask_excludes_positions():
    g = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.float32)
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    mean, weighted = _score_g_values(g, mask)
    assert mean == 1.0
    assert abs(weighted - 1.0) < 1e-7
